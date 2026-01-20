#!/usr/bin/env bash
# ============================================================
# SCRIPT PARA RUNPOD - Ministral 3B com SGLang (UV Edition)
# GPU: A100 / H100 (SM 80/90)
# Repositório: https://github.com/waltagan/start_comand_modelo
# Versão: UV-v19.0 - SUPORTE MINISTRAL 3 (modelo novo)
#
# CORREÇÕES v19:
# - Transformers 4.50.0 (suporta arquitetura 'mistral3')
# - HF_TOKEN configurado via variável de ambiente
# - libnuma1 + libnuma-dev instalados
# - Container recomendado: lmsysorg/sglang:v0.4.3.post2-cu124
# ============================================================

set -e

# --- VARIÁVEIS DE AMBIENTE ---
export CUDA_VISIBLE_DEVICES="0"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_LINK_MODE=copy

# IMPORTANTE: Defina HF_TOKEN como variável de ambiente no RunPod
# O modelo Ministral 3 é gated e requer autenticação
# No RunPod: Environment Variables > HF_TOKEN = seu_token
if [ -z "$HF_TOKEN" ]; then
    echo "⚠ ERRO: HF_TOKEN não definido!"
    echo "  Configure HF_TOKEN nas variáveis de ambiente do RunPod"
    echo "  O modelo Ministral 3 é gated e requer autenticação"
    exit 1
fi

echo "============================================================"
echo "[BOOT] SGLang Server - Ministral 3B (UV Edition v19)"
echo "[INFO] SUPORTE MINISTRAL 3 - Transformers 4.50+"
echo "[INFO] Container recomendado: lmsysorg/sglang:v0.4.3.post2-cu124"
echo "[INFO] HF_TOKEN: ${HF_TOKEN:0:10}..."
echo "[DATA] $(date)"
echo "============================================================"

# --- ETAPA 1: DEPENDÊNCIAS DE SISTEMA (CRÍTICO: libnuma) ---
echo "[1/6] Instalando dependências do sistema..."
apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-pip python3-dev \
    git wget curl \
    build-essential cmake ninja-build \
    libnuma1 libnuma-dev \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Verificar libnuma
echo "[1.1/6] Verificando libnuma..."
ldconfig -p | grep libnuma && echo "✓ libnuma OK" || echo "⚠ libnuma não encontrado"

# Instalar uv
echo "[1.2/6] Instalando uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

# --- ETAPA 2: AMBIENTE VIRTUAL ---
echo "[2/6] Criando ambiente virtual..."
uv venv /workspace/venv --python 3.10 --seed
export PATH="/workspace/venv/bin:$PATH"
source /workspace/venv/bin/activate

# --- ETAPA 3: PYTORCH + CUDA 12.4 ---
echo "[3/6] Instalando PyTorch..."
uv pip install --no-cache \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"

# --- ETAPA 4: TRANSFORMERS 4.50 (SUPORTA MINISTRAL 3) ---
echo "[4/6] Instalando Transformers 4.50.0 (suporte Ministral 3)..."
uv pip install --no-cache \
    "transformers==4.50.0" \
    "tokenizers>=0.19.0" \
    "huggingface_hub>=0.20.0" \
    accelerate \
    "numpy<2.0.0"

# Verificar se Ministral 3 é reconhecido
python3 -c "
from transformers import AutoConfig
print('Testando reconhecimento do Ministral 3...')
try:
    # Isso vai testar se a arquitetura é reconhecida
    config = AutoConfig.from_pretrained('mistralai/Ministral-3-3B-Instruct-2512', trust_remote_code=True, token='$HF_TOKEN')
    print(f'✓ Ministral 3 reconhecido: {config.model_type}')
except Exception as e:
    print(f'⚠ Aviso: {e}')
    print('  Continuando mesmo assim...')
"

# --- ETAPA 5: SGLANG + FLASHINFER + MISTRAL ---
echo "[5/6] Instalando SGLang e dependências..."

# FlashInfer primeiro
uv pip install --no-cache flashinfer-python \
    --extra-index-url https://flashinfer.ai/whl/cu124/torch2.5

# SGLang com todas dependências
uv pip install --no-cache "sglang[all]"

# Suporte Mistral
uv pip install --no-cache \
    "mistral_common>=1.5.0" \
    "tiktoken>=0.7.0" \
    protobuf sentencepiece \
    hf_transfer

# Garantir numpy < 2.0 no final
uv pip install --no-cache "numpy<2.0.0"

# --- ETAPA 6: VERIFICAÇÃO ---
echo "[6/6] Verificação..."

python3 << 'VERIFY_EOF'
import sys
import os

print("\n" + "="*60)
print("VERIFICAÇÃO - UV Edition v19.0 (Ministral 3 Support)")
print("="*60)

errors = []

# HF_TOKEN
hf_token = os.environ.get('HF_TOKEN', '')
if hf_token:
    print(f"✓ HF_TOKEN: {hf_token[:10]}...")
else:
    errors.append("HF_TOKEN não configurado!")

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
ver = transformers.__version__
print(f"✓ Transformers: {ver}")
if not ver.startswith("4.50"):
    errors.append(f"Transformers deveria ser 4.50.x, mas é {ver}")

# sgl_kernel
try:
    import sgl_kernel
    print(f"✓ sgl_kernel: instalado")
except ImportError as e:
    print(f"⚠ sgl_kernel: {e} (pode ser opcional)")

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
    print("ERROS CRÍTICOS:")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("✓ AMBIENTE PRONTO PARA MINISTRAL 3!")
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
echo "[INFO] HF_TOKEN configurado para modelo gated"
echo "============================================================"

python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp 1 \
    --mem-fraction-static 0.90 \
    --attention-backend flashinfer \
    --chat-template mistral \
    --context-length 32768 \
    --log-level info

