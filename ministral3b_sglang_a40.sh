#!/usr/bin/env bash
# ============================================================
# SCRIPT PARA RUNPOD - Ministral 3B com SGLang
# GPU: NVIDIA A40 (Ampere SM 8.6, 48GB VRAM)
# Repositório: https://github.com/waltagan/start_comand_modelo
# Versão: 2.0 - Dependências completas baseadas no pyproject.toml oficial
# ============================================================

set -e

export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "============================================================"
echo "[BOOT] SGLang Server - Ministral 3B (Otimizado A40)"
echo "[INFO] PyTorch 2.4.1 | FlashInfer | CUDA 12.4"
echo "[DATA] $(date)"
echo "============================================================"

# --- ETAPA 1: DEPENDÊNCIAS DE SISTEMA ---
echo "[1/10] Instalando dependências do sistema..."
apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-pip python3-dev git wget curl \
    build-essential cmake ninja-build \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- ETAPA 2: AMBIENTE VIRTUAL ---
echo "[2/10] Criando ambiente virtual..."
python3 -m venv /workspace/venv
export PATH="/workspace/venv/bin:$PATH"
source /workspace/venv/bin/activate
pip install --upgrade pip setuptools wheel packaging

# --- ETAPA 3: PYTORCH 2.4.1 ---
echo "[3/10] Instalando PyTorch 2.4.1..."
pip install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# --- ETAPA 4: NUMPY < 2.0 ---
echo "[4/10] Instalando numpy < 2.0..."
pip install --no-cache-dir "numpy<2.0.0"

# --- ETAPA 5: FLASHINFER ---
echo "[5/10] Instalando FlashInfer..."
pip install --no-cache-dir flashinfer-python \
    -i https://flashinfer.ai/whl/cu124/torch2.4

# --- ETAPA 6: TRANSFORMERS E TOKENIZER ---
echo "[6/10] Instalando Transformers e Tokenizers..."
pip install --no-cache-dir \
    "transformers>=4.46.0,<5.0.0" "tokenizers>=0.21.0" \
    "huggingface_hub>=0.26.0" "mistral_common>=1.5.0" \
    "tiktoken>=0.7.0" protobuf sentencepiece

# --- ETAPA 7: DEPENDÊNCIAS DE RUNTIME CRÍTICAS ---
echo "[7/10] Instalando dependências de runtime..."

# I/O e Serialização (CRÍTICO - inclui pybase64)
pip install --no-cache-dir \
    "pybase64>=1.4.0" \
    "orjson>=3.10.0" \
    "msgspec>=0.18.0" \
    "python-multipart>=0.0.9" \
    "pyzmq>=25.1.2"

# Web Server
pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "uvloop>=0.19.0" \
    httptools watchfiles \
    "pydantic>=2.0.0"

# Geração Estruturada (CRÍTICO - xgrammar)
pip install --no-cache-dir \
    "xgrammar>=0.1.0" \
    "outlines>=0.0.44" \
    interegular lark gguf

# Multimodal (CRÍTICO - opencv headless e decord)
pip install --no-cache-dir \
    "opencv-python-headless>=4.10.0" \
    "pillow>=10.0.0" \
    "decord>=0.6.0" || pip install --no-cache-dir av
pip install --no-cache-dir \
    soundfile imageio einops timm

# Utilidades
pip install --no-cache-dir \
    psutil requests aiohttp tqdm regex rich \
    setproctitle prometheus-client nvidia-ml-py

# --- ETAPA 8: DEPENDÊNCIAS DO SGLANG ---
echo "[8/10] Instalando dependências do SGLang..."
pip install --no-cache-dir \
    ipython openai anthropic \
    diskcache cloudpickle rpyc \
    partial-json-parser compressed-tensors \
    scipy dill pickle5 filelock msgpack blobfile

# Quantização (torchao - pode falhar em PyTorch 2.4)
pip install --no-cache-dir torchao 2>/dev/null || echo "[WARN] torchao não compatível com PyTorch 2.4"

# --- ETAPA 9: SGLANG CORE ---
echo "[9/10] Instalando SGLang..."
pip install --no-cache-dir --no-deps "sglang[all]>=0.4.3"
pip install --no-cache-dir sgl-kernel \
    -i https://flashinfer.ai/whl/cu124/torch2.4 2>/dev/null || true

# Corrigir numpy após instalações
pip install --no-cache-dir "numpy<2.0.0"

# --- ETAPA 10: VERIFICAÇÃO ---
echo "[10/10] Verificação completa..."
python3 << 'VERIFY_EOF'
import sys
print("\n" + "="*60)
print("VERIFICAÇÃO DE DEPENDÊNCIAS CRÍTICAS")
print("="*60)

errors = []

# PyTorch
import torch
print(f"✓ PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# FlashInfer
try:
    import flashinfer
    print(f"✓ FlashInfer: {flashinfer.__version__}")
except: print("⚠ FlashInfer: não carregado (JIT)")

# pybase64 (CRÍTICO)
try:
    import pybase64
    print(f"✓ pybase64: disponível")
except ImportError:
    errors.append("pybase64 não instalado!")

# xgrammar (CRÍTICO)
try:
    import xgrammar
    print(f"✓ xgrammar: disponível")
except ImportError:
    print("⚠ xgrammar: não disponível")

# OpenCV Headless
try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except ImportError:
    errors.append("opencv-python-headless não instalado!")

# orjson
try:
    import orjson
    print(f"✓ orjson: disponível")
except ImportError:
    errors.append("orjson não instalado!")

# msgspec
try:
    import msgspec
    print(f"✓ msgspec: disponível")
except ImportError:
    print("⚠ msgspec: não disponível")

# Numpy
import numpy
print(f"✓ Numpy: {numpy.__version__}")
if numpy.__version__.startswith("2"):
    errors.append("Numpy deve ser < 2.0!")

# SGLang
try:
    import sglang
    print(f"✓ SGLang: {sglang.__version__}")
except Exception as e:
    errors.append(f"SGLang: {e}")

print("-"*60)
if errors:
    print("ERROS CRÍTICOS:")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("✓ TODAS DEPENDÊNCIAS CRÍTICAS OK!")
print("="*60 + "\n")
VERIFY_EOF

# Limpa locks
rm -rf ~/.cache/huggingface/hub/.locks/* 2>/dev/null || true

# --- EXECUÇÃO DO SERVIDOR ---
export MODEL_ID="mistralai/Ministral-3-3B-Instruct-2512"
export SERVED_MODEL_NAME="ministral-3b"

echo "============================================================"
echo "[BOOT] Iniciando servidor..."
echo "[INFO] Modelo: ${MODEL_ID}"
echo "[INFO] Endpoint: http://0.0.0.0:30000"
echo "============================================================"

python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.90 \
    --max-running-requests 64 \
    --disable-cuda-graph-padding \
    --log-level info
