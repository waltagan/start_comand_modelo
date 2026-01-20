#!/usr/bin/env bash
# ============================================================
# SCRIPT PARA RUNPOD - Ministral 3B com SGLang (UV Edition)
# GPU: NVIDIA A40 (Ampere SM 8.6, 48GB VRAM)
# Repositório: https://github.com/waltagan/start_comand_modelo
# Versão: UV-v10.0 - Remove sgl_kernel após instalação
# ============================================================

set -e

export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_LINK_MODE=copy

echo "============================================================"
echo "[BOOT] SGLang Server - Ministral 3B (UV Edition v10)"
echo "[INFO] Usando uv para instalações ultra-rápidas"
echo "[INFO] PyTorch 2.4.1 | SGLang 0.4.3.post2 | CUDA 12.4"
echo "[DATA] $(date)"
echo "============================================================"

# --- ETAPA 1: DEPENDÊNCIAS DE SISTEMA + UV ---
echo "[1/10] Instalando dependências do sistema e uv..."
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
echo "[2/10] Criando ambiente virtual com uv..."
uv venv /workspace/venv --python 3.10 --seed
export PATH="/workspace/venv/bin:$PATH"
source /workspace/venv/bin/activate

# --- ETAPA 3: VLLM PRIMEIRO ---
echo "[3/10] Instalando vLLM..."
uv pip install --no-cache "vllm==0.6.3.post1"

# --- ETAPA 4: FORÇAR PYTORCH 2.4.1+cu124 ---
echo "[4/10] Forçando PyTorch 2.4.1+cu124..."
uv pip uninstall torch torchaudio torchvision 2>/dev/null || true
uv pip install --no-cache \
    torch==2.4.1 \
    torchaudio==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

python3 -c "import torch; v=torch.__version__; assert '2.4.1' in v and 'cu124' in v, f'PyTorch errado: {v}'; print(f'✓ PyTorch: {v}')"

# --- ETAPA 5: DEPENDÊNCIAS BASE ---
echo "[5/10] Instalando dependências base..."
uv pip install --no-cache \
    "numpy<2.0.0" \
    "transformers==4.48.3" \
    "tokenizers>=0.19.0,<0.22.0" \
    "huggingface_hub>=0.20.0,<0.27.0" \
    accelerate

python3 -c "from transformers import AutoProcessor, AutoImageProcessor; print('✓ AutoProcessor disponível')"

# --- ETAPA 6: FLASHINFER ---
echo "[6/10] Instalando FlashInfer..."
uv pip install --no-cache flashinfer-python \
    --extra-index-url https://flashinfer.ai/whl/cu124/torch2.4

# --- ETAPA 7: SGLANG (sem deps) ---
echo "[7/10] Instalando SGLang..."
uv pip install --no-cache --no-deps "sglang[all]==0.4.3.post2"

# --- ETAPA 8: REMOVER sgl_kernel (incompatível com PyTorch 2.4.1) ---
echo "[8/10] Removendo sgl_kernel (incompatível com PyTorch 2.4.1)..."
uv pip uninstall sgl-kernel sgl_kernel 2>/dev/null || true
rm -rf /workspace/venv/lib/python3.10/site-packages/sgl_kernel* 2>/dev/null || true

# --- ETAPA 9: TODAS AS DEPENDÊNCIAS RESTANTES ---
echo "[9/10] Instalando dependências restantes..."
uv pip install --no-cache \
    "mistral_common>=1.5.0" \
    "tiktoken>=0.7.0" \
    protobuf sentencepiece \
    hf_transfer \
    "pybase64>=1.4.0" \
    "orjson>=3.10.0" \
    "msgspec>=0.18.0" \
    "python-multipart>=0.0.9" \
    "pyzmq>=25.1.2" \
    "uvloop>=0.19.0" \
    watchfiles \
    fastapi uvicorn httptools \
    pydantic requests aiohttp \
    "xgrammar>=0.1.0" \
    "outlines>=0.0.44,<1.0.0" \
    interegular lark gguf \
    "opencv-python-headless>=4.9.0,<4.11.0" \
    pillow soundfile imageio einops timm \
    psutil tqdm regex rich numba \
    setproctitle prometheus-client nvidia-ml-py \
    py-spy ninja scipy \
    ipython openai anthropic \
    diskcache cloudpickle rpyc \
    partial-json-parser compressed-tensors \
    dill filelock msgpack blobfile triton \
    modelscope

# Garantir numpy e PyTorch corretos no final
uv pip install --no-cache "numpy<2.0.0"
uv pip install --no-cache \
    torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Remover sgl_kernel novamente (pode ter sido reinstalado)
uv pip uninstall sgl-kernel sgl_kernel 2>/dev/null || true
rm -rf /workspace/venv/lib/python3.10/site-packages/sgl_kernel* 2>/dev/null || true

# --- ETAPA 10: VERIFICAÇÃO FINAL ---
echo "[10/10] Verificação..."
python3 << 'VERIFY_EOF'
import sys
print("\n" + "="*60)
print("VERIFICAÇÃO FINAL - UV Edition v10.0")
print("="*60)

errors = []
warnings = []

# PyTorch
import torch
ver = torch.__version__
print(f"✓ PyTorch: {ver}")
if "2.4.1" not in ver:
    errors.append(f"PyTorch deveria ser 2.4.1, atual: {ver}")
if "cu124" not in ver:
    errors.append(f"PyTorch deveria ser cu124, atual: {ver}")
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

# sgl_kernel - deve estar REMOVIDO
try:
    import sgl_kernel
    warnings.append(f"sgl_kernel ainda está instalado (pode causar erro)")
except ImportError:
    print(f"✓ sgl_kernel: REMOVIDO (correto)")

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
if warnings:
    print("AVISOS:")
    for w in warnings:
        print(f"  ⚠ {w}")
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
    --disable-cuda-graph-padding \
    --log-level info
