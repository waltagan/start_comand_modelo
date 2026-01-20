#!/usr/bin/env bash
# ============================================================
# SCRIPT PARA RUNPOD - Ministral 3B com SGLang (UV Edition)
# GPU: A100 / H100 (SM 80/90)
# Repositório: https://github.com/waltagan/start_comand_modelo
# Versão: UV-v21.0 - SGLANG DO GITHUB (SUPORTE MISTRAL 3)
#
# CORREÇÕES v21:
# - SGLang instalado do GitHub main branch (suporta Mistral3ForConditionalGeneration)
# - PR #5099 incluído (suporte Mistral 3.1)
# - Transformers compatível com SGLang main
# ============================================================

set -e

# --- VARIÁVEIS DE AMBIENTE ---
export CUDA_VISIBLE_DEVICES="0"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_LINK_MODE=copy

# IMPORTANTE: Defina HF_TOKEN como variável de ambiente no RunPod
if [ -z "$HF_TOKEN" ]; then
    echo "⚠ ERRO: HF_TOKEN não definido!"
    echo "  Configure HF_TOKEN nas variáveis de ambiente do RunPod"
    echo "  O modelo Ministral 3 é gated e requer autenticação"
    exit 1
fi

echo "============================================================"
echo "[BOOT] SGLang Server - Ministral 3B (UV Edition v21)"
echo "[INFO] SGLANG DO GITHUB - SUPORTE MISTRAL 3"
echo "[INFO] Container: nvidia/cuda:12.4.1-devel-ubuntu22.04"
echo "[INFO] HF_TOKEN: ${HF_TOKEN:0:10}..."
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

# Verificar libnuma
echo "[1.1/7] Verificando libnuma..."
ldconfig -p | grep libnuma && echo "✓ libnuma OK" || echo "⚠ libnuma não encontrado"

# Instalar uv
echo "[1.2/7] Instalando uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

# --- ETAPA 2: AMBIENTE VIRTUAL ---
echo "[2/7] Criando ambiente virtual..."
uv venv /workspace/venv --python 3.10 --seed
export PATH="/workspace/venv/bin:$PATH"
source /workspace/venv/bin/activate

# --- ETAPA 3: PYTORCH + CUDA 12.4 ---
echo "[3/7] Instalando PyTorch..."
uv pip install --no-cache \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"

# --- ETAPA 4: DEPENDÊNCIAS PRÉ-SGLANG ---
echo "[4/7] Instalando dependências base..."
uv pip install --no-cache \
    "numpy<2.0.0" \
    packaging \
    setuptools \
    wheel \
    ninja \
    psutil

# --- ETAPA 5: SGLANG DO GITHUB (MAIN BRANCH) ---
echo "[5/7] Instalando SGLang do GitHub (main branch)..."
echo "[INFO] Isso inclui PR #5099 com suporte Mistral 3.1"

# Clonar repositório
cd /workspace
if [ -d "sglang" ]; then
    echo "[INFO] Removendo instalação anterior..."
    rm -rf sglang
fi

git clone https://github.com/sgl-project/sglang.git
cd sglang

# Instalar dependências do SGLang
echo "[5.1/7] Instalando dependências do SGLang..."
uv pip install --no-cache -e "python/[all]"

# Instalar FlashInfer compatível
echo "[5.2/7] Instalando FlashInfer..."
uv pip install --no-cache flashinfer-python \
    --extra-index-url https://flashinfer.ai/whl/cu124/torch2.5 || {
    echo "⚠ FlashInfer não disponível, continuando..."
}

# --- ETAPA 6: DEPENDÊNCIAS MISTRAL ---
echo "[6/7] Instalando dependências Mistral..."
uv pip install --no-cache \
    "mistral_common>=1.5.0" \
    "tiktoken>=0.7.0" \
    protobuf sentencepiece \
    hf_transfer

# Garantir numpy < 2.0
uv pip install --no-cache "numpy<2.0.0"

# --- ETAPA 7: VERIFICAÇÃO ---
echo "[7/7] Verificação..."

cd /workspace

python3 << 'VERIFY_EOF'
import sys
import os

print("\n" + "="*60)
print("VERIFICAÇÃO - UV Edition v21.0 (SGLang GitHub)")
print("="*60)

warnings = []

# HF_TOKEN
hf_token = os.environ.get('HF_TOKEN', '')
if hf_token:
    print(f"✓ HF_TOKEN: {hf_token[:10]}...")
else:
    print("✗ HF_TOKEN não configurado!")
    sys.exit(1)

# PyTorch
import torch
print(f"✓ PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")

# Transformers
import transformers
print(f"✓ Transformers: {transformers.__version__}")

# SGLang
try:
    import sglang
    print(f"✓ SGLang: {sglang.__version__} (from GitHub)")
except Exception as e:
    print(f"✗ SGLang: {e}")
    sys.exit(1)

# Verificar suporte Mistral3
try:
    from sglang.srt.models.registry import ModelRegistry
    registry = ModelRegistry()
    architectures = list(registry.models.keys()) if hasattr(registry, 'models') else []
    if 'Mistral3ForConditionalGeneration' in str(architectures) or len(architectures) > 50:
        print(f"✓ Modelo Registry: {len(architectures)} arquiteturas")
    else:
        print(f"⚠ Registry pode não ter Mistral 3")
except Exception as e:
    print(f"⚠ Não foi possível verificar registry: {e}")

# Numpy
import numpy
print(f"✓ Numpy: {numpy.__version__}")

print("-"*60)
if warnings:
    print("AVISOS:")
    for w in warnings:
        print(f"  ⚠ {w}")
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
echo "[INFO] SGLang: GitHub main branch"
echo "============================================================"

cd /workspace/sglang

python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp 1 \
    --mem-fraction-static 0.90 \
    --chat-template mistral \
    --context-length 32768 \
    --log-level info

