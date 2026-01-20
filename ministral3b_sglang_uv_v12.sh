#!/usr/bin/env bash
# ============================================================
# SCRIPT PARA RUNPOD - Ministral 3B com SGLang (UV Edition)
# GPU: NVIDIA A40 (Ampere SM 8.6, 48GB VRAM)
# Repositório: https://github.com/waltagan/start_comand_modelo
# Versão: UV-v12.0 - STACK MODERNA (PyTorch 2.5 + SGLang recente)
# ============================================================

set -e

export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_LINK_MODE=copy

echo "============================================================"
echo "[BOOT] SGLang Server - Ministral 3B (UV Edition v12)"
echo "[INFO] STACK MODERNA - Versões compatíveis entre si"
echo "[INFO] PyTorch 2.5.1 | SGLang latest | CUDA 12.4"
echo "[DATA] $(date)"
echo "============================================================"

# --- ETAPA 1: DEPENDÊNCIAS DE SISTEMA + UV ---
echo "[1/8] Instalando dependências do sistema e uv..."
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

# --- ETAPA 2: AMBIENTE VIRTUAL COM UV ---
echo "[2/8] Criando ambiente virtual com uv..."
uv venv /workspace/venv --python 3.10 --seed
export PATH="/workspace/venv/bin:$PATH"
source /workspace/venv/bin/activate

# --- ETAPA 3: PYTORCH 2.5.1 + CUDA 12.4 ---
echo "[3/8] Instalando PyTorch 2.5.1+cu124..."
uv pip install --no-cache \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"

# --- ETAPA 4: DEPENDÊNCIAS BASE ---
echo "[4/8] Instalando dependências base..."
uv pip install --no-cache \
    "numpy<2.0.0" \
    "transformers>=4.45.0" \
    "tokenizers>=0.19.0" \
    "huggingface_hub>=0.20.0" \
    accelerate

python3 -c "from transformers import AutoProcessor, AutoImageProcessor; print('✓ AutoProcessor disponível')"

# --- ETAPA 5: SGLANG + TODAS DEPENDÊNCIAS ---
echo "[5/8] Instalando SGLang (versão mais recente compatível)..."
# SGLang 0.4.1.post7 é a versão mais recente de Jan/2026 que funciona bem
uv pip install --no-cache "sglang[all]>=0.4.1"

# --- ETAPA 6: FLASHINFER COMPATÍVEL ---
echo "[6/8] Instalando FlashInfer para PyTorch 2.5..."
uv pip install --no-cache flashinfer-python \
    --extra-index-url https://flashinfer.ai/whl/cu124/torch2.5

# --- ETAPA 7: DEPENDÊNCIAS EXTRAS DO MODELO ---
echo "[7/8] Instalando suporte a modelos Mistral..."
uv pip install --no-cache \
    "mistral_common>=1.5.0" \
    "tiktoken>=0.7.0" \
    protobuf sentencepiece \
    hf_transfer

# Garantir numpy < 2.0
uv pip install --no-cache "numpy<2.0.0"

# --- ETAPA 8: VERIFICAÇÃO FINAL ---
echo "[8/8] Verificação..."
python3 << 'VERIFY_EOF'
import sys
print("\n" + "="*60)
print("VERIFICAÇÃO FINAL - UV Edition v12.0 (Stack Moderna)")
print("="*60)

errors = []

# PyTorch
import torch
ver = torch.__version__
print(f"✓ PyTorch: {ver}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Transformers
import transformers
print(f"✓ Transformers: {transformers.__version__}")

# AutoProcessor
try:
    from transformers import AutoProcessor, AutoImageProcessor
    print(f"✓ AutoProcessor: disponível")
except ImportError as e:
    errors.append(f"AutoProcessor: {e}")

# sgl_kernel
try:
    import sgl_kernel
    print(f"✓ sgl_kernel: {getattr(sgl_kernel, '__version__', 'instalado')}")
except ImportError as e:
    print(f"⚠ sgl_kernel: não instalado (pode ser opcional)")

# vLLM
try:
    import vllm
    print(f"✓ vLLM: {vllm.__version__}")
except Exception as e:
    errors.append(f"vLLM: {e}")

# SGLang
try:
    import sglang
    print(f"✓ SGLang: {sglang.__version__}")
except Exception as e:
    errors.append(f"SGLang: {e}")

# Numpy
import numpy
print(f"✓ Numpy: {numpy.__version__}")

print("-"*60)
if errors:
    print("ERROS:")
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
