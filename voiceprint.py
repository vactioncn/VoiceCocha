"""
Voice Coach 声纹识别模块
使用 resemblyzer 进行说话人验证：
1. 注册模式：录制用户声音，生成声纹嵌入并保存
2. 验证模式：对比音频片段与已注册声纹，判断是否为目标用户
"""

import wave
import logging
import time
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

import config

logger = logging.getLogger("voice_coach.voiceprint")

# 声纹文件路径
VOICEPRINT_PATH = config.DATA_DIR / "voiceprint.npy"

# 相似度阈值：高于此值认为是同一人
SPEAKER_THRESHOLD = 0.75


def _get_encoder() -> VoiceEncoder:
    """获取声纹编码器（单例缓存）"""
    if not hasattr(_get_encoder, "_instance"):
        logger.info("加载声纹编码器模型...")
        _get_encoder._instance = VoiceEncoder()
        logger.info("声纹编码器就绪")
    return _get_encoder._instance


def is_registered() -> bool:
    """检查是否已注册声纹"""
    return VOICEPRINT_PATH.exists()


def load_voiceprint() -> np.ndarray | None:
    """加载已保存的声纹嵌入"""
    if not VOICEPRINT_PATH.exists():
        return None
    embed = np.load(str(VOICEPRINT_PATH))
    logger.debug("已加载声纹嵌入 (维度=%s)", embed.shape)
    return embed


def save_voiceprint(embedding: np.ndarray):
    """保存声纹嵌入到文件"""
    np.save(str(VOICEPRINT_PATH), embedding)
    logger.info("声纹已保存至 %s", VOICEPRINT_PATH)


def compute_embedding(wav_path: str) -> np.ndarray | None:
    """从 wav 文件计算声纹嵌入"""
    try:
        wav = preprocess_wav(wav_path)
        if len(wav) < config.SAMPLE_RATE * 0.5:
            # 音频太短，不足以生成可靠嵌入
            logger.warning("音频太短，无法生成声纹: %s", wav_path)
            return None
        encoder = _get_encoder()
        embedding = encoder.embed_utterance(wav)
        return embedding
    except Exception as e:
        logger.error("声纹计算失败: %s", e, exc_info=True)
        return None


def verify_speaker(wav_path: str, voiceprint: np.ndarray) -> tuple[bool, float]:
    """
    验证音频是否匹配已注册声纹
    返回: (is_match, similarity)
    """
    embedding = compute_embedding(wav_path)
    if embedding is None:
        return False, 0.0

    # 余弦相似度
    similarity = np.dot(embedding, voiceprint) / (
        np.linalg.norm(embedding) * np.linalg.norm(voiceprint)
    )
    similarity = float(similarity)

    is_match = similarity >= SPEAKER_THRESHOLD
    logger.debug("声纹验证: 相似度=%.3f, 匹配=%s (%s)", similarity, is_match, wav_path)
    return is_match, similarity


def register_voiceprint_interactive():
    """
    交互式声纹注册
    引导用户录制多段语音，合成声纹嵌入
    """
    import pyaudio

    print("\n🎤 声纹注册")
    print("=" * 60)
    print("为了准确识别你的声音，需要录制 3 段你的语音。")
    print("每段约 8 秒，请正常说话（如介绍自己、念一段话）。")
    print("=" * 60)

    # 预先加载模型（首次会下载，显示进度）
    print("\n⏳ 正在加载声纹识别模型（首次可能需要下载，请稍候）...")
    encoder = _get_encoder()
    print("✅ 模型加载完成\n")

    pa = pyaudio.PyAudio()
    recordings = []

    for i in range(3):
        print(f"\n📍 第 {i+1}/3 段")
        prompts = [
            "请介绍一下你自己和你的公司",
            "请随便说一段你日常工作中会说的话",
            "请再说一段任意内容",
        ]
        print(f"   {prompts[i]}")
        input("   准备好后按回车开始录音...")

        # 录音倒计时
        duration_s = 8
        total_frames = int(config.SAMPLE_RATE / config.FRAME_SIZE * duration_s)
        frames_per_sec = config.SAMPLE_RATE / config.FRAME_SIZE

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=config.CHANNELS,
            rate=config.SAMPLE_RATE,
            input=True,
            frames_per_buffer=config.FRAME_SIZE,
        )

        frames = []
        for f_idx in range(total_frames):
            data = stream.read(config.FRAME_SIZE, exception_on_overflow=False)
            frames.append(data)
            # 每秒更新一次倒计时
            elapsed = (f_idx + 1) / frames_per_sec
            remaining = duration_s - elapsed
            if f_idx % int(frames_per_sec) == 0:
                print(f"\r   🔴 录音中... 剩余 {remaining:.0f} 秒  ", end="", flush=True)

        stream.stop_stream()
        stream.close()
        print(f"\r   ✅ 录音完成 ({duration_s}秒)           ")

        # 保存临时 wav
        tmp_path = config.DATA_DIR / f"voiceprint_reg_{i}.wav"
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(config.CHANNELS)
            wf.setsampwidth(config.SAMPLE_WIDTH)
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(b"".join(frames))

        # 计算声纹嵌入
        print("   ⏳ 正在提取声纹特征...", end="", flush=True)
        wav = preprocess_wav(str(tmp_path))
        embed = encoder.embed_utterance(wav)
        recordings.append(embed)
        print(" 完成 ✅")

        # 清理临时文件
        tmp_path.unlink()

    pa.terminate()

    # 取 3 段嵌入的平均值作为最终声纹
    voiceprint = np.mean(recordings, axis=0)

    # 归一化
    voiceprint = voiceprint / np.linalg.norm(voiceprint)

    # 保存
    save_voiceprint(voiceprint)

    # 验证各段的相似度
    print("\n📊 声纹质量检查:")
    for i, embed in enumerate(recordings):
        sim = float(np.dot(embed, voiceprint) / (
            np.linalg.norm(embed) * np.linalg.norm(voiceprint)
        ))
        print(f"   第 {i+1} 段相似度: {sim:.3f} ✅")

    print(f"\n✅ 声纹注册完成！文件保存在: {VOICEPRINT_PATH}")
    print(f"   识别阈值: {SPEAKER_THRESHOLD}")
    print("   之后录音时，系统会自动标记哪些是你说的 [灏哥]，哪些是别人说的 [其他人]。")
    print("   所有对话都会保留，AI 分析时会结合完整上下文给你反馈。")
