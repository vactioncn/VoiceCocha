# Voice Coach 语音教练系统

全天候办公室语音监控 + AI 教练反馈系统。

系统通过麦克风持续监听办公环境，智能识别有效表达片段（开会、讲解、讨论），离线转录为文字，定时通过 Claude API 进行深度分析，并将教练反馈报告发送到邮箱。

## 系统架构

```
麦克风 → VAD语音检测 → Whisper本地转录 → 过滤 → Claude AI分析 → 邮件推送
```

**五个阶段：**
1. **音频采集 + VAD** - Silero VAD 检测人声，自动切分片段
2. **语音转文字** - faster-whisper 完全离线转录，零成本
3. **片段过滤** - 字数门槛（≥300字）+ 信息密度过滤
4. **AI 分析** - Claude API 做沟通教练分析
5. **邮件推送** - HTML 格式报告发送到邮箱

## 安装

### 1. 安装系统依赖

**macOS：**
```bash
brew install portaudio
```

**Ubuntu/Debian：**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**CentOS/RHEL：**
```bash
sudo yum install portaudio-devel
```

### 2. 创建虚拟环境

```bash
cd voice-coach
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> 如果有 NVIDIA GPU，可安装 CUDA 版 PyTorch 以加速 Whisper 转录：
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

## 配置

### 环境变量

创建 `.env` 文件或在 shell 中导出：

```bash
# Claude API（必需）
export ANTHROPIC_API_KEY="sk-ant-..."

# 邮件（可选，不配置则不发送）
export EMAIL_ENABLED="true"
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
export EMAIL_FROM="your-email@gmail.com"
export EMAIL_TO="your-email@gmail.com"
```

> Gmail 用户需要生成「应用专用密码」：Google 账号 → 安全性 → 两步验证 → 应用专用密码。

### 可调参数

所有参数在 `config.py` 中集中管理：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VAD_THRESHOLD` | 0.5 | 语音检测阈值（0~1），越低越灵敏 |
| `SILENCE_TIMEOUT_S` | 3.0 | 静默多少秒后切断片段 |
| `MAX_SEGMENT_S` | 300 | 单片段最大时长（秒） |
| `MIN_SEGMENT_S` | 1.0 | 过短片段丢弃阈值 |
| `WHISPER_MODEL` | medium | Whisper 模型：tiny/base/small/medium/large-v3 |
| `MIN_CHAR_COUNT` | 300 | 有效片段最低字数 |
| `MIN_INFO_DENSITY` | 0.3 | 有效片段最低信息密度 |
| `ANALYSIS_HOURS` | [12, 18] | 每天自动分析的时间点 |
| `STATUS_INTERVAL_MIN` | 5 | 状态打印间隔（分钟） |

## 使用

### 启动完整系统

```bash
python main.py
```

启动后系统会：
- 后台线程持续录音 + VAD 检测
- 后台线程自动转录有效片段
- 每天 12:00 和 18:00 自动触发 AI 分析
- 每 5 分钟打印今日状态摘要
- `Ctrl+C` 优雅退出（会做一次最终分析）

### 手动触发分析

```bash
python main.py --analyze
```

### 查看今日统计

```bash
python main.py --status
```

### 测试麦克风

```bash
python main.py --test-mic
```

会列出所有音频输入设备，并实时显示 VAD 检测结果（进度条可视化），方便调试。

## 项目结构

```
voice-coach/
├── main.py           # 主入口，4种运行模式
├── recorder.py       # 音频采集 + Silero VAD
├── transcriber.py    # faster-whisper 转录 + 过滤
├── analyzer.py       # Claude API 分析 + 邮件发送
├── database.py       # SQLite 数据层
├── config.py         # 集中配置
├── requirements.txt  # Python 依赖
├── README.md         # 本文件
└── data/             # 运行时自动创建
    ├── audio/        # 音频片段 (.wav)
    ├── voice_coach.db  # SQLite 数据库
    └── voice_coach.log # 日志文件
```

## 数据库

SQLite 数据库 `data/voice_coach.db` 包含两张表：

- **segments** - 所有转录片段（含有效/无效标记）
- **analyses** - AI 分析报告

## 注意事项

- 首次运行会自动下载 Silero VAD 和 Whisper 模型，需要网络连接
- Whisper `medium` 模型约 1.5GB，`large-v3` 约 3GB
- 系统设计为长时间运行，所有错误都会被捕获并记录日志，不会导致崩溃
- 音频文件保存在 `data/audio/`，可定期清理
- 日志文件在 `data/voice_coach.log`
