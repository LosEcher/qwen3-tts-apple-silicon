# 模型管理指南

## 目录
- [模型下载](#模型下载)
- [模型目录结构](#模型目录结构)
- [移动现有模型](#移动现有模型)
- [Git 忽略设置](#git-忽略设置)

## 模型下载

### 方式 1: 使用 huggingface-cli (推荐)

```bash
# 安装 huggingface-cli
pip install huggingface-hub

# 下载模型到项目目录
huggingface-cli download \
  --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit \
  LosEcher/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit
```

### 方式 2: 使用 Python

```python
from huggingface_hub import snapshot_download

# 下载模型
snapshot_download(
    repo_id='LosEcher/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit',
    local_dir='models/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit'
)
```

### 方式 3: 使用 setup_models.sh 脚本

```bash
# 运行模型设置脚本
./setup_models.sh
```

## 可用模型

| 模型名称 | 参数 | 用途 | 推荐场景 |
|---------|------|------|----------|
| `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` | 0.6B | 预设声音+情感 | 实时应用，快速响应 |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | 1.7B | 预设声音+情感 | 高质量需求 |
| `Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit` | 0.6B | 语音设计 | 自定义声音特征 |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | 1.7B | 语音设计 | 高质量语音设计 |
| `Qwen3-TTS-12Hz-0.6B-Base-8bit` | 0.6B | 语音克隆 | 快速语音克隆 |
| `Qwen3-TTS-12Hz-1.7B-Base-8bit` | 1.7B | 语音克隆 | 高质量语音克隆 |

## 模型目录结构

下载后的模型应该位于:
```
models/
├── Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit/
│   └── snapshots/
│       └── <commit-hash>/
│           ├── config.json
│           ├── model.safetensors
│           └── ...
├── Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/
│   └── ...
└── ...
```

服务器会自动识别 `snapshots/` 子目录中的模型文件。

## 移动现有模型

如果你已经在其他位置下载了模型，可以移动到本项目:

### 方法 1: 手动移动

```bash
# 假设模型下载在 ~/.cache/huggingface/hub/
# 或者 ~/Downloads/

# 创建模型目录
mkdir -p models

# 移动模型 (示例)
mv ~/Downloads/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit models/
mv ~/Downloads/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit models/

# 验证
ls -la models/
```

### 方法 2: 使用脚本

```bash
./setup_models.sh
```

脚本会自动搜索常见位置的模型并询问是否移动。

### 方法 3: 创建符号链接 (节省空间)

```bash
# 如果模型在其他位置且不想复制
ln -s /path/to/existing/model models/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit
```

## Git 忽略设置

`.gitignore` 文件已配置为忽略模型文件:

```gitignore
# Large model files (users download separately)
models/
```

这意味着:
- ✅ `models/` 目录不会被提交到 Git
- ✅ 其他开发者需要自行下载模型
- ✅ 项目仓库保持小巧

### 保留目录结构

为了让其他开发者知道模型应该放在哪里，我们使用 `.gitkeep`:

```bash
# 这些文件会被提交，但目录内容被忽略
models/.gitkeep
voices/.gitkeep
outputs/.gitkeep
cache/.gitkeep
uploads/.gitkeep
```

## 验证模型安装

启动服务器后，检查日志:

```
Qwen3-TTS API Server starting...
Models directory: /path/to/models
Available models: ['custom-1.7b', 'custom-0.6b', 'design-1.7b', 'design-0.6b', 'clone-1.7b', 'clone-0.6b']
```

或通过 API 检查:
```bash
curl http://localhost:8825/v1/models
```

如果模型未找到，API 会返回:
```json
{"detail": "Model not found: Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit. Please download it from HuggingFace."}
```

## 存储需求

| 模型 | 大小 (8bit) | 推荐 RAM |
|------|-------------|----------|
| 0.6B 模型 | ~600 MB | 3-4 GB |
| 1.7B 模型 | ~1.7 GB | 6-8 GB |

建议至少保留 10GB 磁盘空间用于模型和生成的音频文件。
