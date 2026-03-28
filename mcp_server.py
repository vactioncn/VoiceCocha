"""
Voice Coach MCP Server
为 OpenClaw 等 MCP 客户端提供语音教练系统的查询、控制和主动推送能力。

启动方式：
    python mcp_server.py                  # stdio 模式（标准 MCP 传输）
    python mcp_server.py --sse --port 8765  # SSE 模式（HTTP 传输）

提供的 Tools：
    - voice_coach_status       查看今日统计
    - voice_coach_segments     查询转录片段
    - voice_coach_analyze      触发一次 AI 分析
    - voice_coach_report       读取最新/指定分析报告
    - voice_coach_reports_list 列出所有历史报告
    - voice_coach_search       搜索对话内容
    - voice_coach_control      启停录音系统
"""

import json
import asyncio
import logging
import argparse
import threading
from queue import Queue
from datetime import datetime, date

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import config
import database

logger = logging.getLogger("voice_coach.mcp")

# 全局引用：录音和转录实例（由 control 工具管理）
_recorder = None
_transcriber = None
_audio_queue = None
_pipeline_running = False

# 通知回调列表
_notification_callbacks = []

app = Server("voice-coach")


# ============================================================
# Tools 定义
# ============================================================

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="voice_coach_status",
            description="查看语音教练系统今日统计：总片段数、有效片段数、总字数、有效时长、分析次数",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="voice_coach_segments",
            description="查询转录片段。可指定日期和是否只看有效片段",
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "查询日期，格式 YYYY-MM-DD，默认今天",
                    },
                    "valid_only": {
                        "type": "boolean",
                        "description": "是否只返回有效片段，默认 true",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最多返回几条，默认 20",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="voice_coach_analyze",
            description="立即触发一次 AI 教练分析，分析今日所有有效对话片段并生成报告",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="voice_coach_report",
            description="读取分析报告。默认返回最新一份，也可指定报告 ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "report_id": {
                        "type": "integer",
                        "description": "报告 ID，不传则返回最新报告",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="voice_coach_reports_list",
            description="列出所有历史分析报告的摘要（ID、时间、片段数、字数）",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "最多返回几条，默认 10",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="voice_coach_search",
            description="搜索对话内容，在所有转录文本中全文检索关键词",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最多返回几条，默认 10",
                    },
                },
                "required": ["keyword"],
            },
        ),
        Tool(
            name="voice_coach_control",
            description="控制录音系统：启动或停止录音+转录流水线",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "status"],
                        "description": "start=启动录音, stop=停止录音, status=查看运行状态",
                    },
                },
                "required": ["action"],
            },
        ),
    ]


# ============================================================
# Tools 实现
# ============================================================

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    database.init_db()

    if name == "voice_coach_status":
        return await _tool_status()
    elif name == "voice_coach_segments":
        return await _tool_segments(arguments)
    elif name == "voice_coach_analyze":
        return await _tool_analyze()
    elif name == "voice_coach_report":
        return await _tool_report(arguments)
    elif name == "voice_coach_reports_list":
        return await _tool_reports_list(arguments)
    elif name == "voice_coach_search":
        return await _tool_search(arguments)
    elif name == "voice_coach_control":
        return await _tool_control(arguments)
    else:
        return [TextContent(type="text", text=f"未知工具: {name}")]


async def _tool_status():
    stats = database.get_today_stats()
    dur_min = int(stats["total_duration_s"] // 60)
    dur_sec = int(stats["total_duration_s"] % 60)

    result = (
        f"📊 语音教练今日统计 ({stats['date']})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"总片段数:   {stats['total_segments']}\n"
        f"有效片段:   {stats['valid_segments']}\n"
        f"总字数:     {stats['total_chars']}\n"
        f"有效时长:   {dur_min}分{dur_sec}秒\n"
        f"分析次数:   {stats['analyses_count']}\n"
        f"录音状态:   {'🟢 运行中' if _pipeline_running else '🔴 已停止'}"
    )
    return [TextContent(type="text", text=result)]


async def _tool_segments(args: dict):
    query_date = args.get("date", date.today().isoformat())
    valid_only = args.get("valid_only", True)
    limit = args.get("limit", 20)

    since = f"{query_date} 00:00:00"
    until = f"{query_date} 23:59:59"

    import sqlite3
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        if valid_only:
            rows = conn.execute(
                """SELECT id, started_at, duration_s, char_count, info_density,
                          is_valid, transcript
                   FROM segments
                   WHERE is_valid = 1 AND started_at >= ? AND started_at <= ?
                   ORDER BY started_at LIMIT ?""",
                (since, until, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, started_at, duration_s, char_count, info_density,
                          is_valid, transcript
                   FROM segments
                   WHERE started_at >= ? AND started_at <= ?
                   ORDER BY started_at LIMIT ?""",
                (since, until, limit),
            ).fetchall()
    finally:
        conn.close()

    if not rows:
        return [TextContent(type="text", text=f"📭 {query_date} 没有{'有效' if valid_only else ''}片段")]

    parts = [f"📋 {query_date} 的对话片段 (共 {len(rows)} 条)\n"]
    for r in rows:
        status = "✅" if r["is_valid"] else "❌"
        parts.append(
            f"\n{status} #{r['id']} [{r['started_at']}] "
            f"时长{r['duration_s']}s | {r['char_count']}字 | 密度{r['info_density']:.2f}\n"
            f"{r['transcript'][:200]}{'...' if len(r['transcript']) > 200 else ''}"
        )

    return [TextContent(type="text", text="\n".join(parts))]


async def _tool_analyze():
    from analyzer import run_analysis

    # 在线程中运行避免阻塞
    loop = asyncio.get_event_loop()
    analysis_id = await loop.run_in_executor(None, run_analysis)

    if analysis_id is None:
        return [TextContent(type="text", text="⚠️ 没有有效片段，无法生成分析报告。")]

    # 获取报告内容
    import sqlite3
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT report, total_segments, total_chars FROM analyses WHERE id = ?",
            (analysis_id,),
        ).fetchone()
    finally:
        conn.close()

    result = (
        f"✅ 分析完成！报告 ID: #{analysis_id}\n"
        f"片段数: {row['total_segments']} | 总字数: {row['total_chars']}\n"
        f"{'━' * 40}\n\n"
        f"{row['report']}"
    )
    return [TextContent(type="text", text=result)]


async def _tool_report(args: dict):
    import sqlite3
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        report_id = args.get("report_id")
        if report_id:
            row = conn.execute(
                "SELECT * FROM analyses WHERE id = ?", (report_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM analyses ORDER BY id DESC LIMIT 1"
            ).fetchone()
    finally:
        conn.close()

    if not row:
        return [TextContent(type="text", text="📭 暂无分析报告")]

    result = (
        f"📊 分析报告 #{row['id']}\n"
        f"时间范围: {row['period_start']} ~ {row['period_end']}\n"
        f"片段数: {row['total_segments']} | 总字数: {row['total_chars']}\n"
        f"生成时间: {row['created_at']}\n"
        f"{'━' * 40}\n\n"
        f"{row['report']}"
    )
    return [TextContent(type="text", text=result)]


async def _tool_reports_list(args: dict):
    import sqlite3
    limit = args.get("limit", 10)

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT id, period_start, period_end, total_segments,
                      total_chars, emailed, created_at
               FROM analyses ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return [TextContent(type="text", text="📭 暂无分析报告")]

    parts = [f"📋 历史报告列表 (最近 {len(rows)} 份)\n"]
    for r in rows:
        email_tag = " 📧" if r["emailed"] else ""
        parts.append(
            f"  #{r['id']} | {r['created_at']} | "
            f"片段{r['total_segments']} | {r['total_chars']}字{email_tag}"
        )

    return [TextContent(type="text", text="\n".join(parts))]


async def _tool_search(args: dict):
    import sqlite3
    keyword = args["keyword"]
    limit = args.get("limit", 10)

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT id, started_at, duration_s, char_count, transcript
               FROM segments
               WHERE transcript LIKE ?
               ORDER BY started_at DESC LIMIT ?""",
            (f"%{keyword}%", limit),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return [TextContent(type="text", text=f"🔍 未找到包含「{keyword}」的对话")]

    parts = [f"🔍 搜索「{keyword}」找到 {len(rows)} 条结果\n"]
    for r in rows:
        # 高亮关键词上下文
        text = r["transcript"]
        idx = text.find(keyword)
        start = max(0, idx - 30)
        end = min(len(text), idx + len(keyword) + 30)
        snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")

        parts.append(
            f"\n#{r['id']} [{r['started_at']}] {r['duration_s']}s\n"
            f"  {snippet}"
        )

    return [TextContent(type="text", text="\n".join(parts))]


async def _tool_control(args: dict):
    global _recorder, _transcriber, _audio_queue, _pipeline_running

    action = args["action"]

    if action == "status":
        status = "🟢 录音系统运行中" if _pipeline_running else "🔴 录音系统已停止"
        return [TextContent(type="text", text=status)]

    elif action == "start":
        if _pipeline_running:
            return [TextContent(type="text", text="⚠️ 录音系统已在运行中")]

        from recorder import VoiceRecorder
        from transcriber import Transcriber

        _audio_queue = Queue()
        _recorder = VoiceRecorder(_audio_queue)
        _transcriber = Transcriber(_audio_queue)
        _recorder.start()
        _transcriber.start()
        _pipeline_running = True

        return [TextContent(type="text", text="🟢 录音系统已启动")]

    elif action == "stop":
        if not _pipeline_running:
            return [TextContent(type="text", text="⚠️ 录音系统未在运行")]

        if _recorder:
            _recorder.stop()
        if _transcriber:
            _transcriber.stop()
        _pipeline_running = False

        return [TextContent(type="text", text="🔴 录音系统已停止")]

    return [TextContent(type="text", text=f"未知操作: {action}")]


# ============================================================
# 主动推送：报告生成后通知
# ============================================================

def notify_report_generated(analysis_id: int, report_summary: str):
    """当报告生成后调用此函数，通知所有连接的客户端"""
    logger.info("推送报告通知: #%d", analysis_id)
    for callback in _notification_callbacks:
        try:
            callback(analysis_id, report_summary)
        except Exception as e:
            logger.warning("通知推送失败: %s", e)


# ============================================================
# 入口
# ============================================================

async def run_stdio():
    """以 stdio 模式运行 MCP Server"""
    database.init_db()
    logger.info("Voice Coach MCP Server 启动 (stdio 模式)")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main():
    parser = argparse.ArgumentParser(description="Voice Coach MCP Server")
    parser.add_argument("--sse", action="store_true", help="使用 SSE 模式启动")
    parser.add_argument("--port", type=int, default=8765, help="SSE 模式端口号")
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-7s %(name)s - %(message)s",
    )

    if args.sse:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        import uvicorn

        sse = SseServerTransport("/messages/")
        database.init_db()

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        starlette_app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ]
        )

        logger.info("Voice Coach MCP Server 启动 (SSE 模式, 端口=%d)", args.port)
        uvicorn.run(starlette_app, host="0.0.0.0", port=args.port)
    else:
        asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
