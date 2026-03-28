"""
Voice Coach 分析模块
定时汇总有效片段，发送给 LLM API 做教练分析，
并通过邮件推送报告。
"""

import smtplib
import logging
from datetime import datetime, date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from openai import OpenAI

import config
import database

logger = logging.getLogger("voice_coach.analyzer")


def run_analysis(since: str | None = None, until: str | None = None) -> int | None:
    """
    执行一次分析
    返回 analysis_id，如果没有有效片段则返回 None
    """
    # 增量分析：从上次分析结束时间开始，避免重复
    today_str = date.today().isoformat()
    if since is None:
        last_end = database.get_last_analysis_end()
        if last_end:
            since = last_end  # 从上次分析结束的地方接着
            logger.info("增量模式：从上次分析结束时间 %s 开始", since)
        else:
            since = f"{today_str} 00:00:00"  # 首次分析，取今天全部
    if until is None:
        until = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("开始分析: %s ~ %s", since, until)

    # 获取有效片段
    segments = database.get_valid_segments(since, until)
    if not segments:
        logger.info("该时间段内没有有效片段，跳过分析")
        return None

    # 拼装上下文
    context_parts = []
    total_chars = 0
    segment_ids = []

    for seg in segments:
        segment_ids.append(seg["id"])
        total_chars += seg["char_count"]
        time_str = seg["started_at"]
        duration = seg["duration_s"]
        context_parts.append(
            f"--- 片段 [{time_str}] (时长 {duration}s, {seg['char_count']}字) ---\n"
            f"{seg['transcript']}\n"
        )

    context = "\n".join(context_parts)

    user_message = (
        f"以下是今天 ({today_str}) 截至目前的对话记录，"
        f"共 {len(segments)} 个有效片段，合计 {total_chars} 字。\n\n"
        f"请按照你的分析框架，给出完整的教练反馈报告。\n\n"
        f"{context}"
    )

    # 调用 LLM API（火山引擎 / OpenAI 兼容接口）
    # 内容过多时分批分析再合并，避免超时
    MAX_CHARS_PER_BATCH = 8000  # 每批最大字数

    try:
        client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            timeout=300.0,  # 5分钟超时
        )

        if total_chars <= MAX_CHARS_PER_BATCH:
            # 内容不多，直接一次分析
            logger.info("调用 LLM API (片段数=%d, 总字数=%d)...", len(segments), total_chars)
            report = _call_llm(client, user_message)
        else:
            # 内容太多，分批分析再合并
            logger.info("内容较多 (%d字)，分批分析...", total_chars)
            report = _batch_analyze(client, context_parts, segments, today_str, total_chars)

        logger.info("LLM API 返回成功 (报告长度=%d字)", len(report))
    except Exception as e:
        logger.error("LLM API 调用失败: %s", e, exc_info=True)
        return None

    # 保存到数据库
    analysis_id = database.save_analysis(
        period_start=since,
        period_end=until,
        segment_ids=segment_ids,
        total_segments=len(segments),
        total_chars=total_chars,
        report=report,
    )

    # 发送邮件
    if config.EMAIL_ENABLED:
        try:
            _send_email(analysis_id, report, len(segments), total_chars, since, until)
            database.mark_emailed(analysis_id)
        except Exception as e:
            logger.error("邮件发送失败: %s", e, exc_info=True)
    else:
        logger.info("邮件发送已禁用，跳过")

    # MCP 主动推送通知
    try:
        from mcp_server import notify_report_generated
        summary = report[:200] + "..." if len(report) > 200 else report
        notify_report_generated(analysis_id, summary)
    except ImportError:
        pass  # MCP Server 未启动时忽略
    except Exception as e:
        logger.warning("MCP 通知推送失败: %s", e)

    return analysis_id


def _call_llm(client, user_message: str) -> str:
    """调用一次 LLM API"""
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": config.COACH_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def _batch_analyze(client, context_parts: list, segments: list, today_str: str, total_chars: int) -> str:
    """
    分批分析：将大量片段分成多批，每批单独分析，最后合并。
    """
    MAX_CHARS_PER_BATCH = 8000
    batches = []
    current_batch = []
    current_chars = 0

    for part, seg in zip(context_parts, segments):
        seg_chars = seg["char_count"]
        if current_chars + seg_chars > MAX_CHARS_PER_BATCH and current_batch:
            batches.append(current_batch)
            current_batch = [part]
            current_chars = seg_chars
        else:
            current_batch.append(part)
            current_chars += seg_chars

    if current_batch:
        batches.append(current_batch)

    logger.info("分为 %d 批进行分析", len(batches))

    # 每批单独分析，得到局部报告
    batch_reports = []
    for i, batch in enumerate(batches, 1):
        batch_text = "\n".join(batch)
        batch_msg = (
            f"以下是今天 ({today_str}) 对话记录的第 {i}/{len(batches)} 部分，"
            f"请先对这部分内容做要点提炼，包括：\n"
            f"1. 对话全景（核心议题、人物关系、走向）\n"
            f"2. 做得好的地方（引用原文 + 为什么好）\n"
            f"3. 值得改善的地方（引用原文 + 对方反应 + 问题本质 + 改写示范）\n"
            f"4. 对话对象的隐性信号和关切\n"
            f"5. 错失的追问机会\n\n"
            f"{batch_text}"
        )
        logger.info("分析第 %d/%d 批...", i, len(batches))
        partial = _call_llm(client, batch_msg)
        batch_reports.append(f"=== 第 {i} 部分分析 ===\n{partial}")

    # 合并所有局部报告，生成最终综合报告
    merge_msg = (
        f"以下是今天 ({today_str}) 全天对话的分批分析结果，"
        f"共 {len(segments)} 个有效片段，合计 {total_chars} 字，分 {len(batches)} 批分析完成。\n\n"
        f"请将这些分批分析结果合并为一份完整的教练反馈报告，"
        f"按照你的分析框架（议题地图、时间分配、表达模式、互动质量、决策质量、洞见、改进建议）输出。\n\n"
        + "\n\n".join(batch_reports)
    )
    logger.info("合并 %d 批分析结果，生成最终报告...", len(batches))
    final_report = _call_llm(client, merge_msg)

    return final_report


def _send_email(
    analysis_id: int,
    report: str,
    segment_count: int,
    total_chars: int,
    period_start: str,
    period_end: str,
):
    """发送分析报告邮件"""
    today_str = date.today().isoformat()
    stats = database.get_today_stats()

    # 计算有效时长的可读格式
    total_duration = stats["total_duration_s"]
    dur_min = int(total_duration // 60)
    dur_sec = int(total_duration % 60)
    duration_str = f"{dur_min}分{dur_sec}秒"

    # 将 markdown 风格的报告转为简单 HTML
    report_html = _markdown_to_html(report)

    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, 'Segoe UI', sans-serif; max-width: 720px; margin: 0 auto; padding: 20px; color: #333; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 24px; border-radius: 12px; margin-bottom: 24px; }}
            .header h1 {{ margin: 0 0 12px 0; font-size: 22px; }}
            .stats {{ display: flex; gap: 16px; flex-wrap: wrap; }}
            .stat {{ background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 8px; }}
            .stat-value {{ font-size: 20px; font-weight: bold; }}
            .stat-label {{ font-size: 12px; opacity: 0.9; }}
            .report {{ background: #f8f9fa; padding: 24px; border-radius: 12px; line-height: 1.8; }}
            .report h2 {{ color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 6px; }}
            .report h3 {{ color: #555; }}
            .footer {{ margin-top: 24px; padding-top: 16px; border-top: 1px solid #ddd; color: #999; font-size: 12px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 语音教练日报 - {today_str}</h1>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{segment_count}</div>
                    <div class="stat-label">有效片段</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{total_chars}</div>
                    <div class="stat-label">总字数</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{duration_str}</div>
                    <div class="stat-label">有效时长</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{stats['total_segments']}</div>
                    <div class="stat-label">今日总片段</div>
                </div>
            </div>
        </div>

        <div class="report">
            {report_html}
        </div>

        <div class="footer">
            由语音教练系统自动生成 | 分析区间: {period_start} ~ {period_end}
        </div>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"语音教练日报 - {today_str}"
    msg["From"] = config.EMAIL_FROM
    msg["To"] = config.EMAIL_TO

    # 纯文本备选
    msg.attach(MIMEText(f"语音教练日报 - {today_str}\n\n{report}", "plain", "utf-8"))
    # HTML 正文
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
        server.starttls()
        server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
        server.send_message(msg)

    logger.info("分析报告邮件已发送至 %s", config.EMAIL_TO)


def _markdown_to_html(text: str) -> str:
    """简单的 Markdown 转 HTML（处理标题、加粗、列表）"""
    import re

    lines = text.split("\n")
    html_lines = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        # 标题
        if stripped.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = stripped[2:]
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            html_lines.append(f"<li>{content}</li>")
        elif stripped.startswith(tuple(f"{i}." for i in range(1, 10))):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = re.sub(r"^\d+\.\s*", "", stripped)
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            html_lines.append(f"<li>{content}</li>")
        elif stripped == "":
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("<br>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", stripped)
            html_lines.append(f"<p>{content}</p>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)
