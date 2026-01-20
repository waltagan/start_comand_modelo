#!/usr/bin/env bash
# ============================================================
# SCRIPT PARA RUNPOD - Ministral 3B com SGLang (UV Edition)
# GPU: NVIDIA A40 (Ampere SM 8.6, 48GB VRAM)
# Repositório: https://github.com/waltagan/start_comand_modelo
# Versão: UV-v14.0 - FIX sgl_kernel sm86 common_ops
# 
# Correções baseadas em:
# - https://github.com/sgl-project/sglang/issues/11421
# - https://github.com/sgl-project/sglang/issues/11335
# ============================================================

set -e

# --- VARIÁVEIS DE AMBIENTE CRÍTICAS (FIX Issues #11421, #11335) ---
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="0"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_LINK_MODE=copy

echo "============================================================"
echo "[BOOT] SGLang Server - Ministral 3B (UV Edition v14)"
echo "[INFO] FIX: sgl_kernel sm86 common_ops (Issues #11421, #11335)"
echo "[INFO] CUDA_HOME: $CUDA_HOME"
echo "[DATA] $(date)"
echo "============================================================"

# --- ETAPA 1: DEPENDÊNCIAS DE SISTEMA (CRÍTICO: libnuma1) ---
echo "[1/6] Instalando dependências do sistema..."
apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-pip python3-dev \
    git wget curl \
    build-essential cmake ninja-build \
    libnuma1 libnuma-dev \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Verificar se libnuma está acessível
echo "[1.1/6] Verificando libnuma..."
ldconfig -p | grep libnuma || echo "WARNING: libnuma não encontrado no cache"

# Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

# --- ETAPA 2: AMBIENTE VIRTUAL ---
echo "[2/6] Criando ambiente virtual..."
uv venv /workspace/venv --python 3.10 --seed
export PATH="/workspace/venv/bin:$PATH"
source /workspace/venv/bin/activate

# --- ETAPA 3: PYTORCH (fix: instalar antes para ter torch/lib) ---
echo "[3/6] Instalando PyTorch..."
uv pip install --no-cache \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# FIX CRÍTICO: Adicionar torch/lib ao LD_LIBRARY_PATH (Issue #11421)
TORCH_LIB_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib
export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$LD_LIBRARY_PATH"
echo "[3.1/6] Torch lib path: $TORCH_LIB_PATH"

python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"

# --- ETAPA 4: SGLANG (instalação oficial) ---
echo "[4/6] Instalando SGLang..."
uv pip install --no-cache "sglang[all]"

# --- ETAPA 5: FLASHINFER + DEPENDÊNCIAS MISTRAL ---
echo "[5/6] Instalando FlashInfer e dependências Mistral..."
# Nota: FlashInfer pode precisar de compilação JIT, então CUDA_HOME é essencial
uv pip install --no-cache flashinfer-python \
    --extra-index-url https://flashinfer.ai/whl/cu124/torch2.5

uv pip install --no-cache \
    "mistral_common>=1.5.0" \
    "tiktoken>=0.7.0" \
    protobuf sentencepiece \
    hf_transfer

# --- ETAPA 6: VERIFICAÇÃO ---
echo "[6/6] Verificação..."

# Re-exportar LD_LIBRARY_PATH antes da verificação
export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

python3 << 'VERIFY_EOF'
import sys
import os
print("\n" + "="*60)
print("VERIFICAÇÃO - UV Edition v14.0 (FIX sgl_kernel)")
print("="*60)

errors = []

# Verificar LD_LIBRARY_PATH
print(f"LD_LIBRARY_PATH includes torch/lib: {'torch' in os.environ.get('LD_LIBRARY_PATH', '')}")

# PyTorch
import torch
print(f"✓ PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")

# sgl_kernel (O TESTE CRÍTICO)
try:
    import sgl_kernel
    print(f"✓ sgl_kernel: carregado com sucesso!")
except ImportError as e:
    # Tentar mostrar mais detalhes
    print(f"⚠ sgl_kernel: {e}")
    print("  Tentando diagnóstico...")
    
    # Verificar arquivos disponíveis
    try:
        import sgl_kernel
        sgl_kernel_path = sgl_kernel.__path__[0] if hasattr(sgl_kernel, '__path__') else "N/A"
    except:
        import importlib.util
        spec = importlib.util.find_spec("sgl_kernel")
        if spec and spec.origin:
            import os
            sgl_kernel_path = os.path.dirname(spec.origin)
        else:
            sgl_kernel_path = "não encontrado"
    
    print(f"  sgl_kernel path: {sgl_kernel_path}")

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
# Re-exportar variáveis de ambiente
export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export MODEL_ID="mistralai/Ministral-3-3B-Instruct-2512"
export SERVED_MODEL_NAME="ministral-3b"
export HOST="0.0.0.0"
export PORT="30000"

echo "============================================================"
echo "[BOOT] Launching SGLang Server..."
echo "[INFO] Modelo: ${MODEL_ID}"
echo "[INFO] Endpoint: http://${HOST}:${PORT}"
echo "[INFO] LD_LIBRARY_PATH configurado para torch/lib + cuda/lib64"
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
