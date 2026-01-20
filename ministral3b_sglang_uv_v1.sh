#!/usr/bin/env bash
# ============================================================
# SCRIPT PARA RUNPOD - Ministral 3B com SGLang (UV Edition)
# GPU: NVIDIA A40 (Ampere SM 8.6, 48GB VRAM)
# Repositório: https://github.com/waltagan/start_comand_modelo
# Versão: UV-v1.0 - Usando uv para instalações ultra-rápidas
# ============================================================

set -e

export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "============================================================"
echo "[BOOT] SGLang Server - Ministral 3B (UV Edition)"
echo "[INFO] Usando uv para instalações 10-100x mais rápidas"
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

# Instalar uv (gerenciador de pacotes ultra-rápido)
echo "[1/10] Instalando uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Verificar instalação do uv
uv --version

# --- ETAPA 2: AMBIENTE VIRTUAL COM UV ---
echo "[2/10] Criando ambiente virtual com uv..."
uv venv /workspace/venv --python 3.10
export PATH="/workspace/venv/bin:$PATH"
source /workspace/venv/bin/activate

# --- ETAPA 3: PYTORCH 2.4.1 + CUDA 12.4 ---
echo "[3/10] Instalando PyTorch 2.4.1+cu124..."
uv pip install --no-cache \
    torch==2.4.1 \
    torchaudio==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Verificar versão
python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"

# --- ETAPA 4: DEPENDÊNCIAS BASE ---
echo "[4/10] Instalando dependências base..."
uv pip install --no-cache \
    "numpy<2.0.0" \
    "transformers==4.48.3" \
    "tokenizers>=0.19.0,<0.22.0" \
    "huggingface_hub>=0.20.0,<0.27.0" \
    accelerate

# Verificar AutoProcessor
python3 -c "from transformers import AutoProcessor, AutoImageProcessor; print('✓ AutoProcessor disponível')"

# --- ETAPA 5: FLASHINFER + SGL-KERNEL ---
echo "[5/10] Instalando FlashInfer e sgl-kernel..."
uv pip install --no-cache flashinfer-python \
    --index-url https://flashinfer.ai/whl/cu124/torch2.4

uv pip install --no-cache --reinstall sgl-kernel \
    --index-url https://flashinfer.ai/whl/cu124/torch2.4

# Verificar sgl_kernel
python3 -c "import sgl_kernel; print('✓ sgl_kernel instalado')"

# --- ETAPA 6: VLLM ---
echo "[6/10] Instalando vLLM 0.6.3.post1..."
uv pip install --no-cache "vllm==0.6.3.post1"

# Reinstalar PyTorch caso vLLM tenha alterado
uv pip install --no-cache --reinstall \
    torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# --- ETAPA 7: SGLANG ---
echo "[7/10] Instalando SGLang 0.4.3.post2..."
uv pip install --no-cache --no-deps "sglang[all]==0.4.3.post2"

# --- ETAPA 8: DEPENDÊNCIAS DO MODELO ---
echo "[8/10] Instalando suporte a modelos Mistral..."
uv pip install --no-cache \
    "mistral_common>=1.5.0" \
    "tiktoken>=0.7.0" \
    protobuf sentencepiece \
    hf_transfer

# --- ETAPA 9: RUNTIME & UTILIDADES ---
echo "[9/10] Instalando dependências de runtime..."
uv pip install --no-cache \
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

# Garantir numpy < 2.0 no final
uv pip install --no-cache "numpy<2.0.0"

# --- ETAPA 10: VERIFICAÇÃO FINAL ---
echo "[10/10] Verificação..."
python3 << 'VERIFY_EOF'
import sys
print("\n" + "="*60)
print("VERIFICAÇÃO FINAL - UV Edition v1.0")
print("="*60)

errors = []

# PyTorch
import torch
print(f"✓ PyTorch: {torch.__version__}")
if "2.4.1" not in torch.__version__:
    errors.append(f"PyTorch deveria ser 2.4.1, mas é {torch.__version__}")
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
    print(f"✓ sgl_kernel: instalado")
except ImportError as e:
    errors.append(f"sgl_kernel: {e}")

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

# SGLang Quantization
try:
    from sglang.srt.layers.quantization import QUANTIZATION_METHODS
    print(f"✓ SGLang Quantization: OK")
except Exception as e:
    errors.append(f"SGLang Quantization: {e}")

# FlashInfer
try:
    import flashinfer
    print(f"✓ FlashInfer: OK")
except:
    print(f"⚠ FlashInfer: não carregado")

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
echo "[INFO] --mem-fraction-static 0.90 (43GB de 48GB)"
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
