#!/usr/bin/env bash
# ============================================================
# SCRIPT PARA RUNPOD - Ministral 3B com SGLang (UV Edition)
# GPU: NVIDIA A40 (Ampere SM 8.6, 48GB VRAM)
# Repositório: https://github.com/waltagan/start_comand_modelo
# Versão: UV-v13.0 - INSTALAÇÃO OFICIAL SIMPLES
# Baseado na documentação oficial: https://docs.sglang.ai
# ============================================================

set -e

export CUDA_VISIBLE_DEVICES="0"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_LINK_MODE=copy

echo "============================================================"
echo "[BOOT] SGLang Server - Ministral 3B (UV Edition v13)"
echo "[INFO] Instalação OFICIAL simplificada"
echo "[INFO] Seguindo: https://docs.sglang.ai/get_started/install.html"
echo "[DATA] $(date)"
echo "============================================================"

# --- ETAPA 1: DEPENDÊNCIAS DE SISTEMA + UV ---
echo "[1/5] Instalando dependências do sistema e uv..."
apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-pip python3-dev \
    git wget curl \
    build-essential cmake ninja-build \
    libnuma1 libnuma-dev \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv (recomendado pela documentação oficial)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

# --- ETAPA 2: AMBIENTE VIRTUAL ---
echo "[2/5] Criando ambiente virtual..."
uv venv /workspace/venv --python 3.10 --seed
export PATH="/workspace/venv/bin:$PATH"
source /workspace/venv/bin/activate

# --- ETAPA 3: INSTALAÇÃO OFICIAL DO SGLANG ---
echo "[3/5] Instalando SGLang (método oficial)..."
# Documentação: uv pip install "sglang" --extra-index-url para CUDA específico
uv pip install --no-cache "sglang[all]" \
    --extra-index-url https://download.pytorch.org/whl/cu124

# --- ETAPA 4: DEPENDÊNCIAS EXTRAS PARA MISTRAL ---
echo "[4/5] Instalando suporte a modelos Mistral..."
uv pip install --no-cache \
    "mistral_common>=1.5.0" \
    "tiktoken>=0.7.0" \
    protobuf sentencepiece \
    hf_transfer

# --- ETAPA 5: VERIFICAÇÃO ---
echo "[5/5] Verificação..."
python3 << 'VERIFY_EOF'
import sys
print("\n" + "="*60)
print("VERIFICAÇÃO - UV Edition v13.0 (Instalação Oficial)")
print("="*60)

errors = []

# PyTorch
import torch
print(f"✓ PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# sgl_kernel
try:
    import sgl_kernel
    print(f"✓ sgl_kernel: instalado")
except ImportError:
    print(f"⚠ sgl_kernel: não instalado")

# vLLM
try:
    import vllm
    print(f"✓ vLLM: {vllm.__version__}")
except Exception as e:
    print(f"⚠ vLLM: {e}")

# SGLang
try:
    import sglang
    print(f"✓ SGLang: {sglang.__version__}")
except Exception as e:
    errors.append(f"SGLang: {e}")

# Transformers
import transformers
print(f"✓ Transformers: {transformers.__version__}")

# Numpy
import numpy
print(f"✓ Numpy: {numpy.__version__}")

print("-"*60)
if errors:
    print("ERROS CRÍTICOS:")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("✓ AMBIENTE PRONTO!")
print("="*60 + "\n")
VERIFY_EOF

# Limpeza
rm -rf ~/.cache/huggingface/hub/.locks/* 2>/dev/null || true

# --- EXECUÇÃO DO SERVIDOR ---
export MODEL_ID="mistralai/Ministral-3-3B-Instruct-2512"
export SERVED_MODEL_NAME="ministral-3b"
export HOST="0.0.0.0"
export PORT="30000"

echo "============================================================"
echo "[BOOT] Launching SGLang Server..."
echo "[INFO] Modelo: ${MODEL_ID}"
echo "[INFO] Endpoint: http://${HOST}:${PORT}"
echo "============================================================"

python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}" \
    --mem-fraction-static 0.90 \
    --max-running-requests 64 \
    --log-level info
