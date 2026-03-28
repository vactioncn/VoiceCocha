"""
Voice Coach 录音模块
使用 PyAudio 持续采集音频，通过 Silero VAD 检测人声，
将有效语音片段保存为 .wav 文件并推入队列。
"""

import io
import wave
import time
import logging
import threading
from queue import Queue
from datetime import datetime

import numpy as np
import torch
import pyaudio

import config

logger = logging.getLogger("voice_coach.recorder")


# ============================================================
# 系统音频输出检测（AVAudioEngine tap）
# 实时测量扬声器 PCM 电平，判断电脑是否在播放音频
# ============================================================

class _SystemAudioMonitor:
    """
    通过 AVAudioEngine 在主混音节点安装 tap，
    实时测量输出 PCM 峰值电平来判断系统是否在播放音频。
    后台线程每 100ms 更新一次结果，供主线程查询。
    """

    # 输出电平高于此值则认为系统在播放音频
    LEVEL_THRESHOLD = 0.005

    def __init__(self):
        self._available = False
        self._is_playing = False
        self._peak = 0.0
        self._lock = threading.Lock()
        self._engine = None

        try:
            from AVFoundation import AVAudioEngine
            self._engine = AVAudioEngine.alloc().init()
            mixer = self._engine.mainMixerNode()

            def tap_block(buf, when):
                try:
                    n = int(buf.frameLength())
                    raw = buf.floatChannelData()
                    if raw is None or n == 0:
                        return
                    # 采样前 256 帧计算峰值（足够检测有无音频）
                    peak = max(abs(raw[0][i]) for i in range(min(n, 256)))
                    with self._lock:
                        self._peak = peak
                        self._is_playing = peak > self.LEVEL_THRESHOLD
                except Exception:
                    pass

            out_fmt = mixer.outputFormatForBus_(0)
            mixer.installTapOnBus_bufferSize_format_block_(0, 2048, out_fmt, tap_block)
            ok, err = self._engine.startAndReturnError_(None)
            if ok:
                self._available = True
                logger.info("系统音频监测已启用（AVAudioEngine tap，阈值=%.3f）", self.LEVEL_THRESHOLD)
            else:
                logger.warning("AVAudioEngine 启动失败: %s", err)
        except Exception as e:
            logger.warning("系统音频监测初始化失败，过滤功能不可用: %s", e)

    def is_system_playing(self) -> bool:
        """返回 True 表示系统正在输出音频"""
        if not self._available:
            return False
        with self._lock:
            return self._is_playing

    def stop(self):
        """停止 tap 和引擎"""
        if self._engine and self._available:
            try:
                self._engine.mainMixerNode().removeTapOnBus_(0)
                self._engine.stop()
            except Exception:
                pass


# 全局单例（模块加载时初始化）
_sys_audio_monitor = _SystemAudioMonitor()


def is_system_playing_audio() -> bool:
    """对外接口：检测系统是否正在播放音频"""
    return _sys_audio_monitor.is_system_playing()


class VoiceRecorder:
    """
    语音录制器
    状态机：空闲态 → 录音态 → 切断 → 空闲态
    """

    # 状态文件路径，menubar 读取此文件判断录音状态
    STATE_FILE = config.DATA_DIR / "recorder_state"

    def __init__(self, output_queue: Queue):
        self.queue = output_queue
        self._running = False
        self._thread: threading.Thread | None = None

        # 初始状态
        self._write_state("idle")

        # 加载 Silero VAD 模型
        logger.info("加载 Silero VAD 模型...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            skip_validation=True,
        )
        logger.info("Silero VAD 模型加载完成")

    def _write_state(self, state: str):
        """写入状态文件供菜单栏读取: idle / recording / stopped"""
        try:
            self.STATE_FILE.write_text(state)
        except Exception:
            pass

    def _reset_vad_state(self):
        """重置 VAD 模型的内部状态"""
        self.vad_model.reset_states()

    def _get_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """对一帧音频计算语音概率"""
        tensor = torch.from_numpy(audio_chunk).float()
        # Silero VAD 要求输入范围 [-1, 1]
        if tensor.abs().max() > 1.0:
            tensor = tensor / 32768.0
        prob = self.vad_model(tensor, config.SAMPLE_RATE).item()
        return prob

    def start(self):
        """启动后台录音线程"""
        if self._running:
            logger.warning("录音器已在运行")
            return
        self._running = True
        self._thread = threading.Thread(target=self._recording_loop, daemon=True)
        self._thread.start()
        logger.info("录音线程已启动")

    def stop(self):
        """停止录音"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("录音线程已停止")

    def _recording_loop(self):
        """录音主循环"""
        pa = pyaudio.PyAudio()
        stream = None

        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=config.CHANNELS,
                rate=config.SAMPLE_RATE,
                input=True,
                frames_per_buffer=config.FRAME_SIZE,
            )
            logger.info("音频流已打开 (采样率=%d, 帧大小=%d)",
                        config.SAMPLE_RATE, config.FRAME_SIZE)

            # 状态变量
            is_recording = False
            audio_frames: list[bytes] = []
            segment_start: datetime | None = None
            silence_frames = 0
            frames_per_second = config.SAMPLE_RATE / config.FRAME_SIZE
            silence_threshold = int(config.SILENCE_TIMEOUT_S * frames_per_second)
            max_frames = int(config.MAX_SEGMENT_S * frames_per_second)

            self._reset_vad_state()

            while self._running:
                try:
                    raw_data = stream.read(config.FRAME_SIZE, exception_on_overflow=False)
                except IOError as e:
                    logger.warning("音频读取错误: %s", e)
                    continue

                # 转为 numpy 数组
                audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

                # VAD 判断
                try:
                    prob = self._get_speech_prob(audio_np)
                except Exception as e:
                    logger.warning("VAD 处理错误: %s", e)
                    continue

                is_speech = prob >= config.VAD_THRESHOLD

                # 系统音频过滤：电脑正在播放音频时跳过，避免录到视频/音乐声
                if is_speech and config.FILTER_SYSTEM_AUDIO and is_system_playing_audio():
                    logger.debug("系统正在播放音频，跳过此帧（过滤电脑声音）")
                    is_speech = False

                if not is_recording:
                    # 空闲态：检测到人声则开始录音
                    if is_speech:
                        is_recording = True
                        audio_frames = [raw_data]
                        segment_start = datetime.now()
                        silence_frames = 0
                        self._write_state("recording")
                        logger.debug("检测到语音，开始录制")
                else:
                    # 录音态：持续缓存音频
                    audio_frames.append(raw_data)

                    if is_speech:
                        silence_frames = 0
                    else:
                        silence_frames += 1

                    # 检查切断条件
                    should_cut = False
                    cut_reason = ""

                    if silence_frames >= silence_threshold:
                        should_cut = True
                        cut_reason = "静默超时"
                    elif len(audio_frames) >= max_frames:
                        should_cut = True
                        cut_reason = "达到最大时长"

                    if should_cut:
                        segment_end = datetime.now()
                        duration_s = (segment_end - segment_start).total_seconds()

                        if duration_s < config.MIN_SEGMENT_S:
                            logger.debug("片段过短 (%.1fs)，丢弃", duration_s)
                        else:
                            # 保存为 .wav 文件
                            audio_path = self._save_wav(audio_frames, segment_start)
                            logger.info("片段保存: %s (%.1fs, 原因=%s)",
                                       audio_path, duration_s, cut_reason)

                            # 推入队列
                            self.queue.put({
                                "audio_path": audio_path,
                                "started_at": segment_start.strftime("%Y-%m-%d %H:%M:%S"),
                                "ended_at": segment_end.strftime("%Y-%m-%d %H:%M:%S"),
                                "duration_s": round(duration_s, 1),
                            })

                        # 重置状态
                        is_recording = False
                        audio_frames = []
                        segment_start = None
                        silence_frames = 0
                        self._write_state("idle")
                        self._reset_vad_state()

        except Exception as e:
            logger.error("录音循环异常: %s", e, exc_info=True)
        finally:
            self._write_state("stopped")
            if stream:
                stream.stop_stream()
                stream.close()
            pa.terminate()
            logger.info("音频资源已释放")

    def _save_wav(self, frames: list[bytes], start_time: datetime) -> str:
        """将音频帧保存为 .wav 文件"""
        filename = f"seg_{start_time.strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = config.AUDIO_DIR / filename

        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(config.CHANNELS)
            wf.setsampwidth(config.SAMPLE_WIDTH)
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(b"".join(frames))

        return str(filepath)


def list_audio_devices():
    """列出所有可用的音频输入设备"""
    pa = pyaudio.PyAudio()
    devices = []
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "sample_rate": int(info["defaultSampleRate"]),
                })
    finally:
        pa.terminate()
    return devices


def test_mic():
    """
    测试麦克风 + VAD
    实时显示语音检测结果，用文本进度条可视化
    """
    print("\n🎤 麦克风测试模式")
    print("=" * 60)

    # 列出设备
    devices = list_audio_devices()
    if not devices:
        print("❌ 未找到任何音频输入设备！")
        return

    print("\n可用音频输入设备：")
    for d in devices:
        print(f"  [{d['index']}] {d['name']} "
              f"(声道={d['channels']}, 采样率={d['sample_rate']})")

    print(f"\n加载 VAD 模型...")
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
        skip_validation=True,
    )
    print("VAD 模型就绪")
    print("\n开始实时检测（Ctrl+C 退出）：")
    print("-" * 60)

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=config.CHANNELS,
        rate=config.SAMPLE_RATE,
        input=True,
        frames_per_buffer=config.FRAME_SIZE,
    )

    try:
        while True:
            raw_data = stream.read(config.FRAME_SIZE, exception_on_overflow=False)
            audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

            tensor = torch.from_numpy(audio_np).float()
            prob = model(tensor, config.SAMPLE_RATE).item()

            # 绘制进度条
            bar_width = 40
            filled = int(prob * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)

            # 颜色标识
            if prob >= config.VAD_THRESHOLD:
                status = "🗣️  说话中"
            else:
                status = "🔇 静默  "

            print(f"\r  {status} [{bar}] {prob:.2f}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n测试结束。")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        model.reset_states()
