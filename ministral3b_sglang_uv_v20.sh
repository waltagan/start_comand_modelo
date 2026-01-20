#!/usr/bin/env bash
# ============================================================
# SCRIPT PARA RUNPOD - Ministral 3B com SGLang (UV Edition)
# GPU: A100 / H100 (SM 80/90)
# Repositório: https://github.com/waltagan/start_comand_modelo
# Versão: UV-v20.0 - SUPORTE MINISTRAL 3 (modelo novo)
#
# CORREÇÕES v20:
# - Aceita Transformers 4.50+ (não só 4.50.x)
# - SGLang 0.5.7 traz Transformers 4.57.1 automaticamente
# - Removida verificação restritiva que causava erro
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
echo "[BOOT] SGLang Server - Ministral 3B (UV Edition v20)"
echo "[INFO] SUPORTE MINISTRAL 3 - Transformers 4.50+"
echo "[INFO] Container: nvidia/cuda:12.4.1-devel-ubuntu22.04"
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

# --- ETAPA 4: SGLANG (TRAZ TRANSFORMERS COMPATÍVEL) ---
echo "[4/6] Instalando SGLang e dependências..."

# FlashInfer primeiro
uv pip install --no-cache flashinfer-python \
    --extra-index-url https://flashinfer.ai/whl/cu124/torch2.5

# SGLang com todas dependências (vai instalar Transformers compatível)
uv pip install --no-cache "sglang[all]"

# Suporte Mistral
uv pip install --no-cache \
    "mistral_common>=1.5.0" \
    "tiktoken>=0.7.0" \
    protobuf sentencepiece \
    hf_transfer

# Garantir numpy < 2.0
uv pip install --no-cache "numpy<2.0.0"

# --- ETAPA 5: VERIFICAR COMPATIBILIDADE MINISTRAL 3 ---
echo "[5/6] Verificando suporte ao Ministral 3..."

python3 << 'CHECK_EOF'
import sys
from packaging import version

# Verificar versão do Transformers
import transformers
ver = version.parse(transformers.__version__)
min_ver = version.parse("4.50.0")
max_ver = version.parse("5.0.0")

print(f"Transformers instalado: {transformers.__version__}")

if ver >= min_ver and ver < max_ver:
    print(f"✓ Versão compatível com Ministral 3!")
else:
    print(f"⚠ Versão fora do range esperado (4.50.0 - 5.0.0)")
    print("  Tentando atualizar para versão compatível...")
    sys.exit(1)
CHECK_EOF

# Se a verificação falhar, tentar ajustar
if [ $? -ne 0 ]; then
    echo "[5.1/6] Ajustando versão do Transformers..."
    uv pip install --no-cache "transformers>=4.50.0,<5.0.0"
fi

# --- ETAPA 6: VERIFICAÇÃO FINAL ---
echo "[6/6] Verificação final..."

python3 << 'VERIFY_EOF'
import sys
import os
from packaging import version

print("\n" + "="*60)
print("VERIFICAÇÃO - UV Edition v20.0 (Ministral 3 Support)")
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

# Transformers (aceita 4.50+ até <5.0)
import transformers
ver = version.parse(transformers.__version__)
min_ver = version.parse("4.50.0")
max_ver = version.parse("5.0.0")

if ver >= min_ver and ver < max_ver:
    print(f"✓ Transformers: {transformers.__version__} (compatível com Ministral 3)")
else:
    warnings.append(f"Transformers {transformers.__version__} pode ter problemas")

# sgl_kernel
try:
    import sgl_kernel
    print(f"✓ sgl_kernel: instalado")
except ImportError as e:
    print(f"⚠ sgl_kernel: não disponível (opcional)")

# SGLang
try:
    import sglang
    print(f"✓ SGLang: {sglang.__version__}")
except Exception as e:
    print(f"✗ SGLang: {e}")
    sys.exit(1)

# Numpy
import numpy
print(f"✓ Numpy: {numpy.__version__}")

# Testar reconhecimento do modelo
print("\n[Testando reconhecimento do Ministral 3...]")
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        'mistralai/Ministral-3-3B-Instruct-2512',
        trust_remote_code=True,
        token=hf_token
    )
    print(f"✓ Ministral 3 reconhecido! (model_type: {config.model_type})")
except Exception as e:
    warnings.append(f"Reconhecimento do modelo: {e}")

print("-"*60)
if warnings:
    print("AVISOS (não críticos):")
    for w in warnings:
        print(f"  ⚠ {w}")
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

