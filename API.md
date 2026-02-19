# Qwen3-TTS API 文档

基于 Calligraph-TTS 需求增强的 Qwen3-TTS API 服务，支持队列管理、音频缓存和多语言语音合成。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务器
python server.py

# 或使用环境变量配置
PORT=8825 HOST=0.0.0.0 python server.py
```

服务器将在 http://localhost:8825 启动，API 文档可在 http://localhost:8825/docs 查看。

## 目录结构

```
qwen3-tts-apple-silicon/
├── server.py           # API 服务器主文件
├── main.py             # CLI 工具
├── outputs/            # 生成的音频文件
├── cache/              # 缓存的音频文件
├── voices/             # 保存的语音克隆
└── uploads/            # 临时上传文件
```

## API 端点概览

### 基础端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | API 信息和可用端点列表 |
| `/health` | GET | 健康检查和队列状态 |
| `/docs` | GET | Swagger UI 文档 |

### OpenAI 兼容端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/models` | GET | 列出可用模型 |
| `/v1/voices` | GET | 列出可用声音 |
| `/v1/audio/speech` | POST | 文本转语音 |

### 队列管理端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/queue/status` | GET | 获取队列状态 |
| `/v1/queue/details` | GET | 获取详细队列信息 |
| `/v1/queue/item/{id}` | GET | 获取特定队列项状态 |
| `/v1/queue/clear` | POST | 清除已完成和失败的任务 |

### 语音合成端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/audio/design` | POST | 语音设计合成 |
| `/v1/audio/clone` | POST | 语音克隆合成 |
| `/tts` | GET | 简单 TTS 接口（测试用） |

### 语音管理端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/voices/saved` | GET | 列出保存的语音 |
| `/v1/voices/save` | POST | 保存新语音 |
| `/v1/voices/saved/{name}` | DELETE | 删除保存的语音 |

### 缓存管理端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/cache/stats` | GET | 获取缓存统计 |
| `/v1/cache/clear` | POST | 清除缓存 |

### 音频文件管理端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/audio/files` | GET | 列出生成的音频文件 |
| `/v1/audio/files/{filename}` | GET | 下载音频文件 |
| `/v1/audio/files/{filename}` | DELETE | 删除音频文件 |

---

## API 详细说明

### 1. 健康检查

```bash
GET /health
```

响应：
```json
{
  "status": "ok",
  "models_loaded": 2,
  "queue": {
    "pending": 0,
    "processing": 1,
    "completed": 5,
    "failed": 0,
    "is_processing": true
  }
}
```

### 2. 列出模型

```bash
GET /v1/models
```

响应：
```json
{
  "object": "list",
  "data": [
    {
      "id": "custom-0.6b",
      "object": "model",
      "created": 1704067200,
      "owned_by": "qwen3-tts"
    },
    ...
  ]
}
```

### 3. 列出声音

```bash
GET /v1/voices
GET /v1/voices?language=chinese  # 按语言筛选
```

响应：
```json
{
  "object": "list",
  "data": [
    {
      "voice_id": "vivian",
      "name": "Vivian",
      "language": "chinese"
    },
    ...
  ]
}
```

**可用语言**: `chinese`, `english`, `japanese`, `korean`

**可用声音**:
- Chinese: `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`
- English: `ryan`, `aiden`, `ethan`, `chelsie`, `serena`, `vivian`
- Japanese: `ono_anna`
- Korean: `sohee`

### 4. 文本转语音（主端点）

```bash
POST /v1/audio/speech
Content-Type: application/json

{
  "model": "custom-0.6b",
  "input": "你好，世界",
  "voice": "vivian",
  "response_format": "wav",
  "speed": 1.0,
  "instruct": "开心且兴奋的语气",
  "language": "chinese"
}
```

**请求参数**:
| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| model | string | 否 | 模型 ID，默认 `custom-0.6b` |
| input | string | 是 | 要合成的文本，最大 5000 字符 |
| voice | string | 否 | 声音 ID，默认 `vivian` |
| response_format | string | 否 | 响应格式: `wav`, `mp3`, `opus`, `aac`, `flac`, `pcm` |
| speed | float | 否 | 语速 (0.5-2.0)，默认 1.0 |
| instruct | string | 否 | 情感指令，如 "excited and happy" |
| language | string | 否 | 语言提示: `chinese`, `english`, `japanese`, `korean` |

**支持的模型**:
- `custom-0.6b` / `custom-1.7b` - 预设声音（带情感控制）
- `design-0.6b` / `design-1.7b` - 语音设计
- `clone-0.6b` / `clone-1.7b` - 语音克隆

**模型别名**:
- `qwen3-tts` → `custom-0.6b`
- `tts-1` → `custom-0.6b`
- `tts-1-hd` → `custom-1.7b`

**响应**: 音频文件 (WAV 格式)，HTTP 头包含 `X-Cache: HIT/MISS`

### 5. 语音设计

```bash
POST /v1/audio/design
Content-Type: application/json

{
  "model": "design-0.6b",
  "input": "这是一段测试文本",
  "voice_description": "深沉的男性旁白声音，专业且温暖",
  "response_format": "wav"
}
```

**请求参数**:
| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| model | string | 否 | 设计模型 ID |
| input | string | 是 | 要合成的文本 |
| voice_description | string | 是 | 声音特征描述 |
| response_format | string | 否 | 响应格式 |

### 6. 语音克隆

```bash
POST /v1/audio/clone
Content-Type: multipart/form-data

{
  "text": "你好，这是克隆的声音",
  "reference_audio": <文件>,
  "reference_text": "参考音频的转录文本（可选）",
  "model": "clone-0.6b"
}
```

**请求参数**:
| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| text | string | 是 | 要合成的文本 |
| reference_audio | file | 是 | 参考音频文件 (wav, mp3, m4a 等) |
| reference_text | string | 否 | 参考音频的转录文本 |
| model | string | 否 | 克隆模型 ID |

**支持格式**: WAV, MP3, M4A, OGG, FLAC

### 7. 队列状态

```bash
GET /v1/queue/status
```

响应：
```json
{
  "pending": 3,
  "processing": 1,
  "completed": 10,
  "failed": 0,
  "is_processing": true
}
```

### 8. 队列详情

```bash
GET /v1/queue/details
```

响应：
```json
{
  "pending": [...],
  "processing": [...],
  "completed": [...],
  "failed": [...]
}
```

### 9. 保存语音

```bash
POST /v1/voices/save
Content-Type: multipart/form-data

{
  "name": "my_voice",
  "transcript": "参考音频的文本",
  "audio": <文件>
}
```

### 10. 缓存统计

```bash
GET /v1/cache/stats
```

响应：
```json
{
  "size": 10485760,
  "size_mb": 10.0,
  "files": 25
}
```

### 11. 列出音频文件

```bash
GET /v1/audio/files?limit=50
```

响应：
```json
{
  "files": [
    {
      "filename": "20240201_120000_你好世界.wav",
      "path": "/path/to/file",
      "size": 102400,
      "size_mb": 0.1,
      "created": "2024-02-01T12:00:00"
    }
  ]
}
```

### 12. 简单 TTS（GET 方式）

```bash
GET /tts?text=你好世界&voice=vivian&model=custom-0.6b&speed=1.0
```

---

## 使用示例

### Python 示例

```python
import requests

# 基础 TTS
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

# 语音设计
response = requests.post(
    "http://localhost:8825/v1/audio/design",
    json={
        "input": "这是一段专业的新闻播报",
        "voice_description": "专业的新闻主播声音，清晰且正式"
    }
)

with open("designed.wav", "wb") as f:
    f.write(response.content)

# 语音克隆
with open("reference.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8825/v1/audio/clone",
        data={
            "text": "这是用克隆的声音说的话",
            "reference_text": "参考音频的文本内容"
        },
        files={"reference_audio": f}
    )

with open("cloned.wav", "wb") as f:
    f.write(response.content)
```

### cURL 示例

```bash
# 基础 TTS
curl -X POST http://localhost:8825/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom-0.6b",
    "input": "你好，世界",
    "voice": "vivian"
  }' \
  --output output.wav

# 语音克隆
curl -X POST http://localhost:8825/v1/audio/clone \
  -F "text=这是克隆的声音" \
  -F "reference_audio=@/path/to/reference.wav" \
  -F "reference_text=参考音频的文本" \
  --output cloned.wav

# 检查队列状态
curl http://localhost:8825/v1/queue/status
```

### JavaScript 示例

```javascript
// 基础 TTS
const response = await fetch('http://localhost:8825/v1/audio/speech', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'custom-0.6b',
    input: '你好，世界',
    voice: 'vivian',
    language: 'chinese'
  })
});

const blob = await response.blob();
const url = URL.createObjectURL(blob);
const audio = new Audio(url);
audio.play();

// 语音克隆
const formData = new FormData();
formData.append('text', '这是克隆的声音');
formData.append('reference_audio', fileInput.files[0]);
formData.append('reference_text', '参考音频文本');

const response = await fetch('http://localhost:8825/v1/audio/clone', {
  method: 'POST',
  body: formData
});
```

---

## 队列管理

API 使用队列系统管理 TTS 请求：

- **最大并发**: 3 个同时处理的请求
- **超时时间**: 120 秒
- **缓存**: 相同文本+声音+模型 的组合会被缓存
- **自动重试**: 失败的请求不会自动重试，需要客户端重新请求

### 队列状态说明

- `pending`: 等待处理
- `processing`: 正在处理
- `completed`: 已完成
- `failed`: 失败

### 缓存机制

- 缓存键: `MD5(text|voice|model)`
- 缓存位置: `cache/` 目录
- 缓存命中时直接返回，不进入队列

---

## 与 Calligraph-TTS 集成

此 API 设计完全兼容 Calligraph-TTS 项目的需求：

1. **OpenAI 兼容端点**: `/v1/audio/speech` 与 OpenAI TTS API 格式一致
2. **队列管理**: 支持高并发请求的队列处理
3. **多语言支持**: 支持中文、英文、日文、韩文
4. **多种声音**: 提供多种预设声音选择
5. **缓存机制**: 减少重复生成，提高响应速度

### Calligraph-TTS 配置示例

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

---

## 性能优化

1. **使用缓存**: 相同请求会命中缓存，立即返回
2. **选择合适模型**: 0.6B 模型比 1.7B 模型快 2-3 倍
3. **批量处理**: 使用队列管理大量请求
4. **清理缓存**: 定期调用 `/v1/cache/clear` 清理旧缓存

---

## 故障排除

### 模型未找到错误

```
Model not found: Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit
```

解决方案：
1. 确保模型文件已下载到 `models/` 目录
2. 检查模型文件夹名称是否正确

### 端口占用

```
Address already in use
```

解决方案：
```bash
# 使用不同端口
PORT=8830 python server.py
```

### 内存不足

对于 1.7B 模型，确保系统至少有 6GB 可用内存。

---

## 许可证

MIT License