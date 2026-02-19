#!/bin/bash
# 模型设置脚本 - 将下载的模型移动到项目目录

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$PROJECT_DIR/models"

echo "=========================================="
echo "Qwen3-TTS 模型设置"
echo "=========================================="
echo ""
echo "项目目录: $PROJECT_DIR"
echo "模型目录: $MODELS_DIR"
echo ""

# 创建模型目录
mkdir -p "$MODELS_DIR"

# 检查是否有外部模型需要移动
echo "检查外部模型..."

# 常见的外部模型位置
COMMON_LOCATIONS=(
    "$HOME/.cache/huggingface/hub"
    "$HOME/Downloads/Qwen3-TTS*"
    "$HOME/models/Qwen3-TTS*"
)

found_models=()

for location in "${COMMON_LOCATIONS[@]}"; do
    if [ -d "$location" ]; then
        echo "  发现: $location"
        found_models+=("$location")
    fi
done

if [ ${#found_models[@]} -eq 0 ]; then
    echo ""
    echo "未发现外部模型。"
    echo ""
    echo "请先下载模型:"
    echo "  方法1: 使用 huggingface-cli"
    echo "    huggingface-cli download --local-dir $MODELS_DIR/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit LosEcher/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
    echo ""
    echo "  方法2: 使用 Python"
    echo "    python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='LosEcher/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit', local_dir='$MODELS_DIR/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit')\""
    echo ""
    echo "可用模型列表:"
    echo "  - LosEcher/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit  (快速，推荐)"
    echo "  - LosEcher/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit  (高质量)"
    echo "  - LosEcher/Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit  (语音设计)"
    echo "  - LosEcher/Qwen3-TTS-12Hz-0.6B-Base-8bit         (语音克隆)"
    exit 1
fi

echo ""
echo "是否要将这些模型移动到项目目录? (y/n)"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    for model_path in "${found_models[@]}"; do
        model_name=$(basename "$model_path")
        target_path="$MODELS_DIR/$model_name"

        if [ -d "$target_path" ]; then
            echo "  跳过: $model_name (已存在)"
        else
            echo "  移动: $model_name"
            mv "$model_path" "$target_path"
        fi
    done
    echo ""
    echo "✅ 模型移动完成!"
else
    echo "取消移动。"
fi

echo ""
echo "当前模型列表:"
ls -la "$MODELS_DIR"
