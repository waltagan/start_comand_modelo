#!/usr/bin/env bash
# ============================================================
# SCRIPT PARA RUNPOD - Ministral 3B com SGLang
# GPU: NVIDIA A40 (Ampere SM 8.6, 48GB VRAM)
# Repositório: https://github.com/waltagan/start_comand_modelo
# Versão: 6.0 - FIX: Adiciona vLLM (dependência do SGLang para quantização)
# ============================================================

set -e

export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "============================================================"
echo "[BOOT] SGLang Server - Ministral 3B (Otimizado A40)"
echo "[INFO] PyTorch 2.4.1 | SGLang 0.4.3.post2 | CUDA 12.4"
echo "[INFO] transformers==4.48.3 | vLLM 0.6.3.post1"
echo "[DATA] $(date)"
echo "============================================================"

# --- ETAPA 1: DEPENDÊNCIAS DE SISTEMA ---
echo "[1/12] Instalando dependências do sistema..."
apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-pip python3-dev \
    git wget curl \
    build-essential cmake ninja-build \
    libnuma1 libnuma-dev \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- ETAPA 2: AMBIENTE VIRTUAL ---
echo "[2/12] Criando ambiente virtual..."
python3 -m venv /workspace/venv
export PATH="/workspace/venv/bin:$PATH"
source /workspace/venv/bin/activate
pip install --upgrade pip setuptools wheel packaging

# --- ETAPA 3: PYTORCH 2.4.1 (Base Estável) ---
echo "[3/12] Instalando PyTorch 2.4.1..."
pip install --no-cache-dir \
    torch==2.4.1 \
    torchaudio==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# --- ETAPA 4: NUMPY < 2.0 (Compatibilidade) ---
echo "[4/12] Instalando numpy < 2.0..."
pip install --no-cache-dir "numpy<2.0.0"

# --- ETAPA 5: FLASHINFER ---
echo "[5/12] Instalando FlashInfer para Torch 2.4 / CUDA 12.4..."
pip install --no-cache-dir flashinfer-python \
    -i https://flashinfer.ai/whl/cu124/torch2.4

# --- ETAPA 6: VLLM (Dependência do SGLang para quantização) ---
# IMPORTANTE: SGLang 0.4.x importa módulos de quantização do vLLM
# Referência: https://github.com/sgl-project/sglang/issues/5843
echo "[6/12] Instalando vLLM 0.6.3.post1 (compatível com PyTorch 2.4)..."
pip install --no-cache-dir "vllm==0.6.3.post1"

# --- ETAPA 7: TRANSFORMERS 4.48.3 ---
echo "[7/12] Instalando transformers==4.48.3..."
pip install --no-cache-dir \
    "transformers==4.48.3" \
    "tokenizers>=0.19.0,<0.22.0" \
    "huggingface_hub>=0.20.0,<0.27.0"

# Verificar AutoProcessor
python3 -c "from transformers import AutoProcessor, AutoImageProcessor; print('✓ AutoProcessor e AutoImageProcessor disponíveis')"

# --- ETAPA 8: SGLANG 0.4.3.post2 ---
echo "[8/12] Instalando SGLang 0.4.3.post2..."
pip install --no-cache-dir --no-deps "sglang[all]==0.4.3.post2"

# sgl-kernel compatível com Torch 2.4
pip install --no-cache-dir sgl-kernel \
    -i https://flashinfer.ai/whl/cu124/torch2.4 2>/dev/null || true

# --- ETAPA 9: OUTRAS DEPENDÊNCIAS DE MODELOS ---
echo "[9/12] Instalando suporte adicional a modelos..."
pip install --no-cache-dir \
    "mistral_common>=1.5.0" \
    "tiktoken>=0.7.0" \
    protobuf sentencepiece accelerate \
    hf_transfer

# --- ETAPA 10: DEPENDÊNCIAS DE RUNTIME ---
echo "[10/12] Instalando dependências de runtime..."

# Grupo 1: Alta Performance & I/O
pip install --no-cache-dir \
    "pybase64>=1.4.0" \
    "orjson>=3.10.0" \
    "msgspec>=0.18.0" \
    "python-multipart>=0.0.9" \
    "pyzmq>=25.1.2" \
    "uvloop>=0.19.0" \
    watchfiles

# Grupo 2: Web Server
pip install --no-cache-dir \
    fastapi uvicorn httptools \
    pydantic requests aiohttp

# Grupo 3: Geração Estruturada
pip install --no-cache-dir \
    "xgrammar>=0.1.0" \
    "outlines>=0.0.44" \
    interegular lark gguf

# Grupo 4: Multimodal
pip install --no-cache-dir \
    "opencv-python-headless>=4.9.0,<4.11.0" \
    pillow soundfile imageio einops timm
pip install --no-cache-dir decord 2>/dev/null || pip install --no-cache-dir av

# Grupo 5: Utilidades
pip install --no-cache-dir \
    psutil tqdm regex rich numba \
    setproctitle prometheus-client nvidia-ml-py \
    py-spy ninja scipy

# --- ETAPA 11: DEPENDÊNCIAS DO SGLANG ---
echo "[11/12] Instalando dependências do SGLang..."
pip install --no-cache-dir \
    ipython openai anthropic \
    diskcache cloudpickle rpyc \
    partial-json-parser compressed-tensors \
    dill filelock msgpack blobfile triton \
    modelscope

# Garantir versões corretas no final
pip install --no-cache-dir "numpy<2.0.0"
pip install --no-cache-dir --force-reinstall --no-deps "transformers==4.48.3"

# --- ETAPA 12: VERIFICAÇÃO FINAL ---
echo "[12/12] Verificação..."
python3 << 'VERIFY_EOF'
import sys
print("\n" + "="*60)
print("VERIFICAÇÃO FINAL")
print("="*60)

errors = []
warnings = []

# PyTorch
import torch
print(f"✓ PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Transformers
import transformers
print(f"✓ Transformers: {transformers.__version__}")

# vLLM
try:
    import vllm
    print(f"✓ vLLM: {vllm.__version__}")
except Exception as e:
    errors.append(f"vLLM: {e}")

# AutoProcessor e AutoImageProcessor
try:
    from transformers import AutoProcessor, AutoImageProcessor
    print(f"✓ AutoProcessor: disponível")
    print(f"✓ AutoImageProcessor: disponível")
except ImportError as e:
    errors.append(f"Auto*Processor: {e}")

# SGLang
try:
    import sglang
    print(f"✓ SGLang: {sglang.__version__}")
except Exception as e:
    errors.append(f"SGLang: {e}")

# Teste de import da camada de quantização
try:
    from sglang.srt.layers.quantization import QUANTIZATION_METHODS
    print(f"✓ SGLang Quantization: OK")
except Exception as e:
    errors.append(f"SGLang Quantization: {e}")

# Dependências críticas
critical = ['pybase64', 'orjson', 'xgrammar', 'flashinfer']
for dep in critical:
    try:
        __import__(dep)
        print(f"✓ {dep}: OK")
    except:
        warnings.append(f"{dep}: não carregado")

# Numpy
import numpy
print(f"✓ Numpy: {numpy.__version__}")

print("-"*60)
if warnings:
    print("AVISOS:")
    for w in warnings:
        print(f"  ⚠ {w}")
if errors:
    print("ERROS CRÍTICOS:")
    for e in errors:
        print(f"  ✗ {e}")
    print("\n❌ Ambiente com erros")
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
