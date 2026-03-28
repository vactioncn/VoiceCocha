#!/usr/bin/env python3
"""
Voice Coach 语音教练系统 - 主入口
全天候办公室语音监控 + AI 教练反馈系统

使用方式：
    python main.py              # 默认：启动全部（录音 + 转录 + 定时分析）
    python main.py --analyze    # 手动触发一次分析
    python main.py --status     # 打印今日统计
    python main.py --test-mic   # 测试麦克风 + VAD
"""

import sys
import signal
import logging
import argparse
import time
import threading
from queue import Queue
from datetime import datetime

import schedule

import config
import database
from recorder import VoiceRecorder, test_mic
from transcriber import Transcriber
from analyzer import run_analysis
from voiceprint import register_voiceprint_interactive, is_registered


# ============================================================
# 日志配置
# ============================================================
def setup_logging():
    """配置日志：同时输出到终端和文件"""
    root_logger = logging.getLogger("voice_coach")
    root_logger.setLevel(logging.DEBUG)

    # 日志格式
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 终端输出（INFO 级别）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # 文件输出（DEBUG 级别）
    file_handler = logging.FileHandler(str(config.LOG_PATH), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    return root_logger


logger = None  # 延迟初始化


# ============================================================
# 命令: --status
# ============================================================
def cmd_status():
    """打印今日统计"""
    database.init_db()
    stats = database.get_today_stats()

    dur_min = int(stats["total_duration_s"] // 60)
    dur_sec = int(stats["total_duration_s"] % 60)

    print(f"\n📊 Voice Coach 今日统计 ({stats['date']})")
    print("=" * 45)
    print(f"  总片段数:   {stats['total_segments']}")
    print(f"  有效片段:   {stats['valid_segments']}")
    print(f"  总字数:     {stats['total_chars']}")
    print(f"  有效时长:   {dur_min}分{dur_sec}秒")
    print(f"  分析次数:   {stats['analyses_count']}")
    print("=" * 45)


# ============================================================
# 命令: --analyze
# ============================================================
def cmd_analyze():
    """手动触发一次分析"""
    database.init_db()
    print("\n🧠 手动触发分析...")

    analysis_id = run_analysis()
    if analysis_id is None:
        print("⚠️  没有有效片段，无法生成分析报告。")
    else:
        print(f"✅ 分析完成，报告 ID: #{analysis_id}")


# ============================================================
# 命令: --test-mic
# ============================================================
def cmd_test_mic():
    """测试麦克风"""
    test_mic()


# ============================================================
# MCP Server 后台启动
# ============================================================
def _start_mcp_server():
    """在后台线程中启动 MCP SSE Server"""
    def _run_mcp():
        try:
            from mcp.server.sse import SseServerTransport
            from starlette.applications import Starlette
            from starlette.routing import Route, Mount
            import uvicorn
            from mcp_server import app as mcp_app

            sse = SseServerTransport("/messages/")

            async def handle_sse(request):
                async with sse.connect_sse(
                    request.scope, request.receive, request._send
                ) as streams:
                    await mcp_app.run(
                        streams[0], streams[1], mcp_app.create_initialization_options()
                    )

            starlette_app = Starlette(
                routes=[
                    Route("/sse", endpoint=handle_sse),
                    Mount("/messages/", app=sse.handle_post_message),
                ]
            )

            logger.info("MCP Server 启动 (SSE, 端口=%d)", config.MCP_PORT)
            uvicorn.run(
                starlette_app,
                host="0.0.0.0",
                port=config.MCP_PORT,
                log_level="warning",  # 减少 uvicorn 日志噪音
            )
        except Exception as e:
            logger.error("MCP Server 启动失败: %s", e, exc_info=True)

    mcp_thread = threading.Thread(target=_run_mcp, daemon=True)
    mcp_thread.start()
    logger.info("MCP Server 已在后台启动 → http://localhost:%d/sse", config.MCP_PORT)


# ============================================================
# 命令: 默认模式（全流水线）
# ============================================================
def cmd_run():
    """启动完整流水线：录音 + 转录 + 定时分析"""
    global logger
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("Voice Coach 语音教练系统启动")
    logger.info("=" * 60)

    # 初始化数据库
    database.init_db()

    # 音频队列（recorder → transcriber）
    audio_queue = Queue()

    # 启动录音器
    recorder = VoiceRecorder(audio_queue)
    recorder.start()

    # 启动转录器
    transcriber = Transcriber(audio_queue)
    transcriber.start()

    # 启动 MCP Server（后台线程）
    if config.MCP_ENABLED:
        _start_mcp_server()

    # 配置定时分析
    for hour in config.ANALYSIS_HOURS:
        schedule_time = f"{hour:02d}:00"
        schedule.every().day.at(schedule_time).do(_scheduled_analysis)
        logger.info("已设置定时分析: 每天 %s", schedule_time)

    # 配置定时状态打印
    schedule.every(config.STATUS_INTERVAL_MIN).minutes.do(_print_status)

    # 配置定时清理旧音频文件（每天凌晨 3 点）
    schedule.every().day.at("03:00").do(_cleanup_old_audio)
    logger.info("已设置定时清理: 每天 03:00 清理 %d 天前的音频文件", config.AUDIO_RETENTION_DAYS)

    # 优雅退出处理
    shutdown_requested = [False]

    def signal_handler(signum, frame):
        if shutdown_requested[0]:
            return
        shutdown_requested[0] = True
        logger.info("\n收到退出信号，正在关闭...")

        # 停止录音
        recorder.stop()
        logger.info("录音已停止")

        # 停止转录
        transcriber.stop()
        logger.info("转录已停止")

        # 做一次最终分析
        logger.info("执行最终分析...")
        try:
            analysis_id = run_analysis()
            if analysis_id:
                logger.info("最终分析完成，报告 ID: #%d", analysis_id)
            else:
                logger.info("没有新的有效片段需要分析")
        except Exception as e:
            logger.error("最终分析失败: %s", e)

        logger.info("Voice Coach 已安全退出")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 打印初始状态
    _print_status()

    logger.info("系统就绪，按 Ctrl+C 退出")
    logger.info("-" * 60)

    # 主线程运行调度器
    while not shutdown_requested[0]:
        schedule.run_pending()
        time.sleep(1)


def _cleanup_old_audio():
    """清理超过保留天数的旧音频文件"""
    import os
    from pathlib import Path

    cutoff = time.time() - config.AUDIO_RETENTION_DAYS * 86400
    audio_dir = config.AUDIO_DIR
    removed = 0
    freed_bytes = 0

    try:
        for f in Path(audio_dir).glob("*.wav"):
            if f.stat().st_mtime < cutoff:
                size = f.stat().st_size
                f.unlink()
                removed += 1
                freed_bytes += size

        if removed > 0 and logger:
            freed_mb = freed_bytes / (1024 * 1024)
            logger.info("🧹 清理完成: 删除 %d 个旧音频文件，释放 %.1fMB", removed, freed_mb)
        elif logger:
            logger.debug("🧹 没有需要清理的旧音频文件")
    except Exception as e:
        if logger:
            logger.error("音频清理失败: %s", e)


def _scheduled_analysis():
    """定时分析任务"""
    if logger:
        logger.info("⏰ 触发定时分析...")
    try:
        analysis_id = run_analysis()
        if analysis_id and logger:
            logger.info("定时分析完成，报告 ID: #%d", analysis_id)
    except Exception as e:
        if logger:
            logger.error("定时分析失败: %s", e, exc_info=True)


def _print_status():
    """打印今日状态摘要"""
    try:
        stats = database.get_today_stats()
        dur_min = int(stats["total_duration_s"] // 60)
        dur_sec = int(stats["total_duration_s"] % 60)

        now = datetime.now().strftime("%H:%M:%S")
        msg = (
            f"[{now}] 📊 今日: "
            f"片段 {stats['valid_segments']}/{stats['total_segments']} | "
            f"字数 {stats['total_chars']} | "
            f"时长 {dur_min}m{dur_sec}s | "
            f"分析 {stats['analyses_count']}次"
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)
    except Exception as e:
        if logger:
            logger.warning("状态查询失败: %s", e)


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Voice Coach 语音教练系统 - 全天候办公室语音监控 + AI 教练反馈",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式:
  python main.py              启动全部（录音 + 转录 + 定时分析）
  python main.py --analyze    手动触发一次分析
  python main.py --status     打印今日统计
  python main.py --test-mic   测试麦克风 + VAD
  python main.py --register   注册你的声纹（首次使用必须执行）
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--analyze", action="store_true", help="手动触发一次分析")
    group.add_argument("--status", action="store_true", help="打印今日统计")
    group.add_argument("--test-mic", action="store_true", help="测试麦克风 + VAD")
    group.add_argument("--register", action="store_true", help="注册你的声纹")

    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.analyze:
        setup_logging()
        cmd_analyze()
    elif args.test_mic:
        cmd_test_mic()
    elif args.register:
        register_voiceprint_interactive()
    else:
        cmd_run()


if __name__ == "__main__":
    main()
