# Qwen3-TTS API 快速启动指南

## 1. 环境准备

确保系统已安装：
- Python 3.10+
- macOS (Apple Silicon M1/M2/M3/M4)
- ffmpeg: `brew install ffmpeg`

## 2. 安装依赖

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install fastapi uvicorn python-multipart mlx numpy soundfile
pip install transformers sentencepiece huggingface-hub tqdm
pip install git+https://github.com/Blaizzy/mlx-audio.git
```

或者使用 requirements.txt:
```bash
pip install -r requirements.txt
pip install git+https://github.com/Blaizzy/mlx-audio.git
```

## 3. 下载/移动模型（可选）

如果没有模型文件，API 会返回错误但可以测试端点。

### 方式 1: 直接下载到项目

使用以下命令将模型直接下载到本项目文件夹下（不会提交到 Git）:

```bash
# 下载 0.6B 快速模型（推荐）
huggingface-cli download \
  --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit \
  LosEcher/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit

# 或者使用 Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='LosEcher/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit',
    local_dir='models/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit'
)
"
```

### 方式 2: 移动已下载的模型

如果你已在其他位置下载了模型:

```bash
# 使用脚本自动查找并移动
./setup_models.sh

# 或手动移动
mv /path/to/downloaded/model models/
```

模型文件会存放在 `models/` 目录下，该目录已被 `.gitignore` 排除，不会提交到 Git。
```bash
# 使用 huggingface-cli
huggingface-cli download --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit LosEcher/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit

# 或者使用 Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='LosEcher/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit', local_dir='models/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit')
"
```

可用模型:
- `LosEcher/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` - 快速，适合实时应用
- `LosEcher/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` - 高质量
- `LosEcher/Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit` - 语音设计
- `LosEcher/Qwen3-TTS-12Hz-0.6B-Base-8bit` - 语音克隆

## 4. 启动服务器

```bash
# 默认端口 8825
python server.py

# 或使用不同端口
PORT=8830 python server.py

# 后台运行
python server.py &
```

服务器启动后:
- API 地址: http://localhost:8825
- 文档地址: http://localhost:8825/docs

## 5. 测试 API

### 使用测试脚本
```bash
python test_api.py
```

### 使用 curl
```bash
# 健康检查
curl http://localhost:8825/health

# 列出模型
curl http://localhost:8825/v1/models

# 列出声音
curl http://localhost:8825/v1/voices

# 文本转语音
curl -X POST http://localhost:8825/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom-0.6b",
    "input": "你好，世界",
    "voice": "vivian",
    "language": "chinese"
  }' \
  --output output.wav

# 语音设计
curl -X POST http://localhost:8825/v1/audio/design \
  -H "Content-Type: application/json" \
  -d '{
    "input": "这是一段测试文本",
    "voice_description": "深沉的男性旁白声音"
  }' \
  --output designed.wav
```

### 使用 Python
```python
import requests

# 文本转语音
response = requests.post(
    "http://localhost:8825/v1/audio/speech",
    json={
        "model": "custom-0.6b",
        "input": "你好，这是中文语音合成测试",
        "voice": "vivian",
        "language": "chinese"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)

print("音频已保存到 output.wav")
```

## 6. 与 Calligraph-TTS 集成

在 Calligraph-TTS 项目的 backend 中修改配置:

```javascript
// backend/config.js
const config = {
  provider: 'openai',
  ttsUrl: 'http://localhost:8825/v1/audio/speech',
  ttsApiKey: '',  // 本地服务不需要 API key
  ttsModel: 'qwen3-tts',  // 或 custom-0.6b
  ttsVoice: 'vivian',
  ttsLanguage: 'Chinese'
};
```

## 7. 目录结构

```
qwen3-tts-apple-silicon/
├── server.py           # API 服务器
├── test_api.py         # 测试脚本
├── API.md              # 完整 API 文档
├── models/             # TTS 模型文件
├── outputs/            # 生成的音频文件
├── cache/              # 缓存文件
├── voices/             # 保存的语音克隆
└── uploads/            # 临时上传文件
```

## 8. 常见问题

### 端口被占用
```bash
# 使用不同端口
PORT=8830 python server.py
```

### 模型未找到错误
下载模型文件到 `models/` 目录，或检查模型路径配置。

### 内存不足
- 使用 0.6B 模型而非 1.7B 模型
- 关闭其他应用程序

## 9. API 端点速查

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/models` | GET | 列出模型 |
| `/v1/voices` | GET | 列出声音 |
| `/v1/audio/speech` | POST | 文本转语音 |
| `/v1/audio/design` | POST | 语音设计 |
| `/v1/audio/clone` | POST | 语音克隆 |
| `/v1/queue/status` | GET | 队列状态 |
| `/v1/cache/stats` | GET | 缓存统计 |

更多详细信息请查看 `API.md` 文件。
