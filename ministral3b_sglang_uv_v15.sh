#!/usr/bin/env bash
# ============================================================
# SCRIPT PARA RUNPOD - Ministral 3B com SGLang (UV Edition)
# GPU: NVIDIA A40 (Ampere SM 8.6, 48GB VRAM)
# Repositório: https://github.com/waltagan/start_comand_modelo
# Versão: UV-v15.0 - SOLUÇÃO DEFINITIVA para SM86 (A40)
# 
# PROBLEMA: sgl_kernel mais recente só tem binários para SM100
#           GPU A40 é SM86 (Ampere) - incompatível
#
# SOLUÇÃO: Usar SGLang 0.4.6.post5 (última versão estável para SM86)
#          com stack de dependências compatível
#
# Referências:
# - https://github.com/sgl-project/sglang/issues/11763
# - https://github.com/sgl-project/sglang/issues/7070
# ============================================================

set -e

# --- VARIÁVEIS DE AMBIENTE ---
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="0"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_LINK_MODE=copy

echo "============================================================"
echo "[BOOT] SGLang Server - Ministral 3B (UV Edition v15)"
echo "[INFO] SOLUÇÃO DEFINITIVA: SGLang 0.4.6.post5 para SM86 (A40)"
echo "[INFO] CUDA_HOME: $CUDA_HOME"
echo "[DATA] $(date)"
echo "============================================================"

# --- ETAPA 1: DEPENDÊNCIAS DE SISTEMA ---
echo "[1/7] Instalando dependências do sistema..."
apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-pip python3-dev \
    git wget curl \
    build-essential cmake ninja-build \
    libnuma1 libnuma-dev \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

# --- ETAPA 2: AMBIENTE VIRTUAL ---
echo "[2/7] Criando ambiente virtual..."
uv venv /workspace/venv --python 3.10 --seed
export PATH="/workspace/venv/bin:$PATH"
source /workspace/venv/bin/activate

# --- ETAPA 3: PYTORCH 2.6.0 (versão compatível com SGLang 0.4.6) ---
echo "[3/7] Instalando PyTorch 2.6.0+cu124..."
uv pip install --no-cache \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Adicionar torch/lib ao LD_LIBRARY_PATH
TORCH_LIB_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib
export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$LD_LIBRARY_PATH"

python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"

# --- ETAPA 4: FLASHINFER COMPATÍVEL ---
echo "[4/7] Instalando FlashInfer (versão compatível com torch 2.6)..."
uv pip install --no-cache flashinfer-python \
    --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6

# --- ETAPA 5: SGLANG 0.4.6.post5 (última versão estável para SM86) ---
echo "[5/7] Instalando SGLang 0.4.6.post5..."
# Esta versão usa sgl_kernel 0.1.x que tem suporte a SM86
uv pip install --no-cache "sglang[all]==0.4.6.post5"

# --- ETAPA 6: DEPENDÊNCIAS EXTRAS ---
echo "[6/7] Instalando dependências extras..."
uv pip install --no-cache \
    "numpy<2.0.0" \
    "transformers>=4.45.0,<4.50.0" \
    "tokenizers>=0.19.0,<0.22.0" \
    "huggingface_hub>=0.20.0" \
    accelerate \
    "mistral_common>=1.5.0" \
    "tiktoken>=0.7.0" \
    protobuf sentencepiece \
    hf_transfer

# --- ETAPA 7: VERIFICAÇÃO ---
echo "[7/7] Verificação..."

export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

python3 << 'VERIFY_EOF'
import sys
import os
print("\n" + "="*60)
print("VERIFICAÇÃO - UV Edition v15.0 (SGLang 0.4.6.post5)")
print("="*60)

errors = []

# PyTorch
import torch
ver = torch.__version__
print(f"✓ PyTorch: {ver}")
if not ver.startswith("2.6"):
    errors.append(f"PyTorch deveria ser 2.6.x, mas é {ver}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")

# sgl_kernel (agora deve funcionar com 0.1.x)
try:
    import sgl_kernel
    print(f"✓ sgl_kernel: {getattr(sgl_kernel, '__version__', 'instalado')}")
except ImportError as e:
    print(f"⚠ sgl_kernel: {e}")
    print("  (Pode ser opcional para modelos não-FP8)")

# SGLang
try:
    import sglang
    ver = sglang.__version__
    print(f"✓ SGLang: {ver}")
    if not ver.startswith("0.4.6"):
        errors.append(f"SGLang deveria ser 0.4.6.x, mas é {ver}")
except Exception as e:
    errors.append(f"SGLang: {e}")

# Transformers
import transformers
print(f"✓ Transformers: {transformers.__version__}")

# AutoProcessor
try:
    from transformers import AutoProcessor
    print(f"✓ AutoProcessor: disponível")
except ImportError as e:
    errors.append(f"AutoProcessor: {e}")

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
export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export MODEL_ID="mistralai/Ministral-3-3B-Instruct-2512"
export SERVED_MODEL_NAME="ministral-3b"
export HOST="0.0.0.0"
export PORT="30000"

echo "============================================================"
echo "[BOOT] Launching SGLang Server..."
echo "[INFO] Modelo: ${MODEL_ID}"
echo "[INFO] Endpoint: http://${HOST}:${PORT}"
echo "[INFO] SGLang 0.4.6.post5 - Compatível com SM86 (A40)"
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
