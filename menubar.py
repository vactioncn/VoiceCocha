#!/usr/bin/env python3
"""
Voice Coach 菜单栏应用
macOS 顶部菜单栏，动态显示录音状态，提供一键操作
"""

import os
import subprocess
import threading
import time
import sqlite3
from pathlib import Path
from datetime import date

import rumps
import pyaudio

# 项目路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "voice_coach.db"
LOG_PATH = DATA_DIR / "voice_coach.log"
VENV_PYTHON = BASE_DIR / "venv" / "bin" / "python"
MAIN_PY = BASE_DIR / "main.py"
DEVICE_FILE = DATA_DIR / "audio_device.txt"
STATE_FILE = DATA_DIR / "recorder_state"

# LaunchAgent 标识
SERVICE_LABEL = "com.voicecoach.service"

# 环境变量
ENV = os.environ.copy()
ENV["ARK_API_KEY"] = os.environ.get("ARK_API_KEY", "")
ENV["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# ============================================================
# 菜单栏图标状态（使用纯文字，确保 macOS 兼容性）
# ============================================================
# 录音中：交替显示，产生闪烁效果
ICON_RECORDING_A = "● REC"
ICON_RECORDING_B = "○ REC"
# 空闲监听中
ICON_IDLE = "◉ VC"
# 服务已停止
ICON_STOPPED = "◎ VC"


def is_service_running() -> bool:
    """检查后台服务是否在运行"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "voice-coach/main.py"],
            capture_output=True, text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_recorder_state() -> str:
    """读取录音器状态文件: idle / recording / stopped"""
    try:
        if STATE_FILE.exists():
            return STATE_FILE.read_text().strip()
    except Exception:
        pass
    return "unknown"


def get_audio_devices() -> list[dict]:
    """获取所有可用的音频输入设备"""
    pa = pyaudio.PyAudio()
    devices = []
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                })
    finally:
        pa.terminate()
    return devices


def get_selected_device() -> int | None:
    """获取当前选中的音频设备索引"""
    if DEVICE_FILE.exists():
        try:
            return int(DEVICE_FILE.read_text().strip())
        except (ValueError, OSError):
            pass
    return None


def save_selected_device(index: int):
    """保存选中的音频设备索引"""
    DEVICE_FILE.write_text(str(index))


def get_today_stats() -> dict:
    """从数据库获取今日统计"""
    default = {"total_segments": 0, "valid_segments": 0, "total_chars": 0,
               "total_duration_s": 0, "analyses_count": 0}
    if not DB_PATH.exists():
        return default
    today_str = date.today().isoformat()
    since = f"{today_str} 00:00:00"
    until = f"{today_str} 23:59:59"
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM segments WHERE started_at >= ? AND started_at <= ?",
            (since, until)).fetchone()["cnt"]
        valid = conn.execute(
            "SELECT COUNT(*) as cnt FROM segments WHERE is_valid=1 AND started_at >= ? AND started_at <= ?",
            (since, until)).fetchone()["cnt"]
        chars = conn.execute(
            "SELECT COALESCE(SUM(char_count),0) as c FROM segments WHERE is_valid=1 AND started_at >= ? AND started_at <= ?",
            (since, until)).fetchone()["c"]
        dur = conn.execute(
            "SELECT COALESCE(SUM(duration_s),0) as d FROM segments WHERE is_valid=1 AND started_at >= ? AND started_at <= ?",
            (since, until)).fetchone()["d"]
        analyses = conn.execute(
            "SELECT COUNT(*) as cnt FROM analyses WHERE period_start >= ?",
            (since,)).fetchone()["cnt"]
        return {"total_segments": total, "valid_segments": valid,
                "total_chars": chars, "total_duration_s": round(dur, 1),
                "analyses_count": analyses}
    except Exception:
        return default
    finally:
        conn.close()


def get_latest_report() -> str | None:
    """获取最新的分析报告"""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute(
            "SELECT report, created_at FROM analyses ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            return f"[{row[1]}]\n\n{row[0]}"
        return None
    except Exception:
        return None
    finally:
        conn.close()


class VoiceCoachApp(rumps.App):
    """语音教练菜单栏应用 - 增强版"""

    def __init__(self):
        super().__init__(
            name="Voice Coach",
            title=ICON_STOPPED,
            quit_button=None,
        )

        # 动态图标状态
        self._icon_toggle = False  # 用于录音时图标交替
        self._last_state = "unknown"

        # ========== 菜单项 ==========

        # 状态区域
        self.status_item = rumps.MenuItem("检查中...", callback=None)
        self.recording_info = rumps.MenuItem("", callback=None)

        # 今日数据概览
        self.stats_header = rumps.MenuItem("── 今日数据 ──", callback=None)
        self.stat_segments = rumps.MenuItem("  片段: -", callback=None)
        self.stat_chars = rumps.MenuItem("  字数: -", callback=None)
        self.stat_duration = rumps.MenuItem("  时长: -", callback=None)
        self.stat_analyses = rumps.MenuItem("  分析: -", callback=None)

        # 操作按钮
        self.toggle_item = rumps.MenuItem("▶ 启动服务", callback=self.on_toggle)
        self.analyze_item = rumps.MenuItem("🧠 生成分析报告", callback=self.on_analyze)
        self.report_item = rumps.MenuItem("📋 查看最新报告", callback=self.on_view_report)

        # 音频设备子菜单
        self.device_menu = rumps.MenuItem("🎤 录音设备")
        self._build_device_menu()

        # 工具
        self.log_item = rumps.MenuItem("📄 打开日志", callback=self.on_open_log)
        self.folder_item = rumps.MenuItem("📁 打开项目目录", callback=self.on_open_folder)
        self.quit_item = rumps.MenuItem("退出菜单栏", callback=self.on_quit)
        self.quit_service_item = rumps.MenuItem("⏹ 停止服务并退出", callback=self.on_quit_all)

        self.menu = [
            self.status_item,
            self.recording_info,
            None,
            self.stats_header,
            self.stat_segments,
            self.stat_chars,
            self.stat_duration,
            self.stat_analyses,
            None,
            self.toggle_item,
            self.analyze_item,
            self.report_item,
            None,
            self.device_menu,
            None,
            self.log_item,
            self.folder_item,
            None,
            self.quit_item,
            self.quit_service_item,
        ]

        # 快速刷新定时器（2秒，用于动态图标）
        self._icon_timer = rumps.Timer(self._update_icon, 2)
        self._icon_timer.start()

        # 慢速刷新定时器（15秒，用于统计数据）
        self._stats_timer = rumps.Timer(self._refresh_stats, 15)
        self._stats_timer.start()

        # 立即刷新一次
        self._update_icon(None)
        self._refresh_stats(None)

    # ========== 动态图标 ==========

    def _update_icon(self, _):
        """每2秒更新图标和录音状态"""
        running = is_service_running()
        rec_state = get_recorder_state() if running else "stopped"

        if not running:
            self.title = ICON_STOPPED
            self.status_item.title = "○ 服务未运行"
            self.recording_info.title = ""
            self.toggle_item.title = "▶ 启动服务"
        elif rec_state == "recording":
            # 录音中：图标交替闪烁
            self._icon_toggle = not self._icon_toggle
            self.title = ICON_RECORDING_A if self._icon_toggle else ICON_RECORDING_B
            self.status_item.title = "🔴 正在录音..."
            self.recording_info.title = "  检测到语音，录制中"
            self.toggle_item.title = "⏹ 停止服务"
        elif rec_state == "idle":
            self.title = ICON_IDLE
            self.status_item.title = "● 监听中"
            self.recording_info.title = "  等待语音输入..."
            self.toggle_item.title = "⏹ 停止服务"
        else:
            self.title = ICON_IDLE if running else ICON_STOPPED
            self.status_item.title = "● 运行中" if running else "○ 已停止"
            self.recording_info.title = ""
            self.toggle_item.title = "⏹ 停止服务" if running else "▶ 启动服务"

    # ========== 统计数据刷新 ==========

    def _refresh_stats(self, _):
        """刷新今日统计数据"""
        stats = get_today_stats()
        dur_min = int(stats["total_duration_s"] // 60)
        dur_sec = int(stats["total_duration_s"] % 60)

        self.stat_segments.title = f"  📝 片段: {stats['valid_segments']} 个有效 / {stats['total_segments']} 个总计"
        self.stat_chars.title = f"  📊 字数: {stats['total_chars']:,} 字"
        self.stat_duration.title = f"  ⏱ 时长: {dur_min}分{dur_sec}秒"
        self.stat_analyses.title = f"  🧠 分析: {stats['analyses_count']} 次"

    # ========== 音频设备菜单 ==========

    def _build_device_menu(self):
        """构建音频设备子菜单"""
        try:
            self.device_menu.clear()
        except (AttributeError, Exception):
            pass

        devices = get_audio_devices()
        selected = get_selected_device()

        for dev in devices:
            idx = dev["index"]
            name = dev["name"]
            prefix = "✅ " if idx == selected else "    "
            item = rumps.MenuItem(f"{prefix}{name}", callback=self._make_device_callback(idx, name))
            self.device_menu[item.title] = item

        sep = rumps.MenuItem("──────────", callback=None)
        self.device_menu[sep.title] = sep
        refresh = rumps.MenuItem("🔄 刷新设备列表", callback=self.on_refresh_devices)
        self.device_menu[refresh.title] = refresh

    def _make_device_callback(self, device_index, device_name):
        def callback(_):
            save_selected_device(device_index)
            rumps.notification("Voice Coach", "录音设备已切换",
                             f"已选择: {device_name}\n重启服务后生效")
            self._build_device_menu()
        return callback

    def on_refresh_devices(self, _):
        self._build_device_menu()
        rumps.notification("Voice Coach", "设备列表已刷新", "")

    # ========== 操作回调 ==========

    def on_toggle(self, _):
        """启动/停止服务"""
        if is_service_running():
            # 停止服务
            subprocess.run(["launchctl", "stop", SERVICE_LABEL], capture_output=True)
            subprocess.run(["pkill", "-f", "voice-coach/main.py"], capture_output=True)
            time.sleep(1)
            self.title = ICON_STOPPED
            self.status_item.title = "○ 服务未运行"
            self.toggle_item.title = "▶ 启动服务"
            rumps.notification("Voice Coach", "服务已停止", "录音和转录已暂停")
        else:
            # 启动服务
            self.title = "⏳"
            self.status_item.title = "正在启动..."
            subprocess.run(["launchctl", "start", SERVICE_LABEL], capture_output=True)
            time.sleep(3)
            if is_service_running():
                self.title = ICON_IDLE
                self.status_item.title = "● 监听中"
                self.toggle_item.title = "⏹ 停止服务"
                rumps.notification("Voice Coach", "✅ 服务已启动", "开始录音和转录")
            else:
                self.title = ICON_STOPPED
                rumps.notification("Voice Coach", "❌ 启动失败", "请检查日志")
        self._refresh_stats(None)

    def on_analyze(self, _):
        """触发分析报告"""
        rumps.notification("Voice Coach", "🧠 正在生成报告...", "这需要一点时间")

        def _run():
            try:
                result = subprocess.run(
                    [str(VENV_PYTHON), str(MAIN_PY), "--analyze"],
                    capture_output=True, text=True,
                    cwd=str(BASE_DIR), env=ENV, timeout=300,
                )
                output = result.stdout + result.stderr
                if "分析完成" in output:
                    rumps.notification("Voice Coach", "✅ 报告已生成",
                                     "点击菜单栏 → 查看最新报告")
                elif "没有有效片段" in output:
                    rumps.notification("Voice Coach", "⚠️ 没有有效片段",
                                     "需要更多对话内容才能生成报告")
                else:
                    rumps.notification("Voice Coach", "⚠️ 生成结果",
                                     output[-100:] if output else "请查看日志")
            except subprocess.TimeoutExpired:
                rumps.notification("Voice Coach", "❌ 生成超时", "请稍后重试")
            except Exception as e:
                rumps.notification("Voice Coach", "❌ 生成失败", str(e)[:100])
            self._refresh_stats(None)

        threading.Thread(target=_run, daemon=True).start()

    def on_view_report(self, _):
        """查看最新报告"""
        report = get_latest_report()
        if report:
            tmp_path = DATA_DIR / "latest_report.md"
            tmp_path.write_text(report, encoding="utf-8")
            subprocess.run(["open", str(tmp_path)])
        else:
            rumps.notification("Voice Coach", "暂无报告", "还没有生成过分析报告")

    def on_open_log(self, _):
        if LOG_PATH.exists():
            subprocess.run(["open", "-a", "Console", str(LOG_PATH)])
        else:
            rumps.notification("Voice Coach", "日志不存在", "服务尚未运行过")

    def on_open_folder(self, _):
        subprocess.run(["open", str(BASE_DIR)])

    def on_quit(self, _):
        """退出菜单栏（不停止后台服务）"""
        rumps.quit_application()

    def on_quit_all(self, _):
        """停止服务并退出"""
        if is_service_running():
            subprocess.run(["launchctl", "stop", SERVICE_LABEL], capture_output=True)
            subprocess.run(["pkill", "-f", "voice-coach/main.py"], capture_output=True)
        rumps.quit_application()


if __name__ == "__main__":
    VoiceCoachApp().run()
