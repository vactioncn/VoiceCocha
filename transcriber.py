"""
Voice Coach 转录模块
从队列取音频文件，用 Qwen3-ASR 转录为文字，
进行过滤后存入数据库。

Qwen3-ASR 是通义千问团队开源的语音识别模型，
原生支持四川话等 22 种中国方言，Apache 2.0 协议。
首次运行会自动从 HuggingFace 下载模型到 ~/.cache/huggingface/。
"""

import logging
import threading
from queue import Queue, Empty

import torch

import config
import database
import voiceprint

logger = logging.getLogger("voice_coach.transcriber")


def _detect_device() -> str:
    """自动检测最佳推理设备"""
    if config.ASR_DEVICE != "auto":
        return config.ASR_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _detect_dtype(device: str):
    """根据设备选择最佳数据类型"""
    if device == "cuda":
        # GPU 用 bfloat16 加速
        return torch.bfloat16
    # CPU 用 float32 保证兼容性
    return torch.float32


def _calculate_info_density(text: str) -> float:
    """
    计算信息密度 = (总字数 - 语气词字数) / 总字数
    """
    total_chars = len(text)
    if total_chars == 0:
        return 0.0

    filler_chars = 0
    for filler in config.FILLER_WORDS:
        count = text.count(filler)
        filler_chars += count * len(filler)

    density = (total_chars - filler_chars) / total_chars
    return max(0.0, min(1.0, density))


def _is_valid_segment(text: str) -> tuple[bool, int, float]:
    """
    判断片段是否有效
    返回: (is_valid, char_count, info_density)
    """
    # 去除空白后计算字数
    clean_text = text.strip()
    char_count = len(clean_text)
    info_density = _calculate_info_density(clean_text)

    # 硬性规则：字数门槛
    if char_count < config.MIN_CHAR_COUNT:
        return False, char_count, info_density

    # 软性规则：信息密度
    if info_density < config.MIN_INFO_DENSITY:
        return False, char_count, info_density

    return True, char_count, info_density


class Transcriber:
    """转录器：从队列消费音频文件，用 Qwen3-ASR 转录并存储"""

    def __init__(self, audio_queue: Queue):
        self.queue = audio_queue
        self._running = False
        self._thread: threading.Thread | None = None
        self._model = None
        # 声纹验证：加载已注册的声纹
        self._voiceprint = voiceprint.load_voiceprint()
        if self._voiceprint is not None:
            logger.info("声纹验证已启用 - 将标记说话人身份")
        else:
            logger.warning("未注册声纹 - 将处理所有语音片段（运行 --register 注册声纹）")

    def _load_model(self):
        """延迟加载 Qwen3-ASR 模型"""
        if self._model is not None:
            return

        from qwen_asr import Qwen3ASRModel

        device = _detect_device()
        dtype = _detect_dtype(device)
        device_map = f"{device}:0" if device == "cuda" else device

        logger.info("加载 Qwen3-ASR 模型: %s (设备=%s, 精度=%s)",
                     config.ASR_MODEL, device, dtype)
        logger.info("首次运行会自动下载模型，请耐心等待...")

        self._model = Qwen3ASRModel.from_pretrained(
            config.ASR_MODEL,
            dtype=dtype,
            device_map=device_map,
            max_new_tokens=config.ASR_MAX_NEW_TOKENS,
        )
        logger.info("Qwen3-ASR 模型加载完成")

    def start(self):
        """启动后台转录线程"""
        if self._running:
            logger.warning("转录器已在运行")
            return
        self._running = True
        self._thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._thread.start()
        logger.info("转录线程已启动")

    def stop(self):
        """停止转录"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("转录线程已停止")

    def _transcribe_loop(self):
        """转录主循环"""
        self._load_model()

        while self._running:
            try:
                item = self.queue.get(timeout=1.0)
            except Empty:
                continue

            try:
                self._process_segment(item)
            except Exception as e:
                logger.error("转录处理异常: %s", e, exc_info=True)

    def _process_segment(self, item: dict):
        """处理单个音频片段"""
        audio_path = item["audio_path"]

        # 声纹识别：标记说话人身份（全部保留，不丢弃任何片段）
        speaker_tag = ""
        if self._voiceprint is not None:
            is_me, similarity = voiceprint.verify_speaker(audio_path, self._voiceprint)
            if is_me:
                speaker_tag = "[灏哥]"
                logger.info("声纹匹配 (相似度=%.3f): %s", similarity, audio_path)
            else:
                speaker_tag = "[其他人]"
                logger.info("声纹不匹配 (相似度=%.3f): %s", similarity, audio_path)

        logger.info("开始转录: %s", audio_path)

        # 使用 Qwen3-ASR 执行转录
        # language=None 表示自动检测语言/方言
        results = self._model.transcribe(
            audio=audio_path,
            language=None,
        )

        # 提取转录文本
        transcript = ""
        if results and len(results) > 0:
            detected_lang = results[0].language
            transcript = results[0].text.strip()
            logger.debug("检测到语言: %s", detected_lang)

        if not transcript:
            logger.info("转录结果为空，跳过: %s", audio_path)
            database.save_segment(
                audio_path=audio_path,
                started_at=item["started_at"],
                ended_at=item["ended_at"],
                duration_s=item["duration_s"],
                transcript="",
                char_count=0,
                info_density=0.0,
                is_valid=False,
            )
            return

        logger.debug("转录原文: %s", transcript[:100])

        # 在转录文本前加上说话人标记
        if speaker_tag:
            transcript = f"{speaker_tag} {transcript}"

        # 过滤判断
        is_valid, char_count, info_density = _is_valid_segment(transcript)

        # 存入数据库
        seg_id = database.save_segment(
            audio_path=audio_path,
            started_at=item["started_at"],
            ended_at=item["ended_at"],
            duration_s=item["duration_s"],
            transcript=transcript,
            char_count=char_count,
            info_density=info_density,
            is_valid=is_valid,
        )

        status = "有效" if is_valid else "无效"
        logger.info(
            "转录完成 #%d [%s] %s: 字数=%d, 密度=%.2f, 时长=%.1fs",
            seg_id, status, speaker_tag, char_count, info_density, item["duration_s"],
        )
