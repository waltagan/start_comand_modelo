#!/usr/bin/env bash
# ============================================================
# SCRIPT PARA RUNPOD - Ministral 3B com SGLang
# GPU: NVIDIA A40 (Ampere SM 8.6, 48GB VRAM)
# Repositório: https://github.com/waltagan/start_comand_modelo
# Versão: 4.0 - Corrigido: SGLang 0.4.3.post2 + PyTorch 2.4.1
# ============================================================

set -e

export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "============================================================"
echo "[BOOT] SGLang Server - Ministral 3B (Otimizado A40)"
echo "[INFO] PyTorch 2.4.1 | SGLang 0.4.3.post2 | CUDA 12.4"
echo "[DATA] $(date)"
echo "============================================================"

# --- ETAPA 1: DEPENDÊNCIAS DE SISTEMA ---
echo "[1/10] Instalando dependências do sistema..."
apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-pip python3-dev \
    git wget curl \
    build-essential cmake ninja-build \
    libnuma1 libnuma-dev \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- ETAPA 2: AMBIENTE VIRTUAL ---
echo "[2/10] Criando ambiente virtual..."
python3 -m venv /workspace/venv
export PATH="/workspace/venv/bin:$PATH"
source /workspace/venv/bin/activate
pip install --upgrade pip setuptools wheel packaging

# --- ETAPA 3: PYTORCH 2.4.1 (Base Estável) ---
echo "[3/10] Instalando PyTorch 2.4.1..."
pip install --no-cache-dir \
    torch==2.4.1 \
    torchaudio==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# --- ETAPA 4: NUMPY < 2.0 (Compatibilidade) ---
echo "[4/10] Instalando numpy < 2.0..."
pip install --no-cache-dir "numpy<2.0.0"

# --- ETAPA 5: FLASHINFER (Kernel de Atenção Otimizado) ---
echo "[5/10] Instalando FlashInfer para Torch 2.4 / CUDA 12.4..."
pip install --no-cache-dir flashinfer-python \
    -i https://flashinfer.ai/whl/cu124/torch2.4

# --- ETAPA 6: SGLANG 0.4.3.post2 (Compatível com PyTorch 2.4.1) ---
echo "[6/10] Instalando SGLang 0.4.3.post2..."
pip install --no-cache-dir --no-deps "sglang[all]==0.4.3.post2"

# sgl-kernel compatível com Torch 2.4
pip install --no-cache-dir sgl-kernel \
    -i https://flashinfer.ai/whl/cu124/torch2.4 2>/dev/null || true

# --- ETAPA 7: TRANSFORMERS & TOKENIZERS ---
echo "[7/10] Instalando suporte a modelos..."
pip install --no-cache-dir \
    "transformers>=4.46.0,<4.57.0" \
    "tokenizers>=0.21.0" \
    "huggingface_hub>=0.26.0" \
    "mistral_common>=1.5.0" \
    "tiktoken>=0.7.0" \
    protobuf sentencepiece accelerate \
    hf_transfer

# --- ETAPA 8: DEPENDÊNCIAS DE RUNTIME ---
echo "[8/10] Instalando dependências de runtime..."

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
    "opencv-python-headless>=4.9.0,<4.13.0" \
    pillow soundfile imageio moviepy einops timm
pip install --no-cache-dir decord 2>/dev/null || pip install --no-cache-dir av

# Grupo 5: Utilidades & Debug
pip install --no-cache-dir \
    psutil tqdm regex rich numba \
    setproctitle prometheus-client nvidia-ml-py \
    py-spy ninja scipy

# --- ETAPA 9: DEPENDÊNCIAS DO SGLANG ---
echo "[9/10] Instalando dependências do SGLang..."
pip install --no-cache-dir \
    ipython openai anthropic \
    diskcache cloudpickle rpyc \
    partial-json-parser compressed-tensors \
    dill filelock msgpack blobfile triton \
    modelscope

# Garantir numpy < 2.0 após todas instalações
pip install --no-cache-dir "numpy<2.0.0"

# --- ETAPA 10: VERIFICAÇÃO FINAL ---
echo "[10/10] Verificação..."
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

# AutoProcessor
try:
    from transformers import AutoProcessor
    print(f"✓ AutoProcessor: disponível")
except ImportError as e:
    errors.append(f"AutoProcessor: {e}")

# SGLang
try:
    import sglang
    print(f"✓ SGLang: {sglang.__version__}")
except Exception as e:
    errors.append(f"SGLang: {e}")

# Dependências críticas
critical = ['pybase64', 'orjson', 'xgrammar', 'flashinfer']
for dep in critical:
    try:
        __import__(dep)
        print(f"✓ {dep}: OK")
    except:
        warnings.append(f"{dep}: não carregado (pode não ser necessário)")

# Numpy
import numpy
print(f"✓ Numpy: {numpy.__version__}")
if numpy.__version__.startswith("2"):
    warnings.append("Numpy >= 2.0 pode causar problemas")

print("-"*60)
if warnings:
    print("AVISOS (não críticos):")
    for w in warnings:
        print(f"  ⚠ {w}")
if errors:
    print("ERROS CRÍTICOS:")
    for e in errors:
        print(f"  ✗ {e}")
    # NÃO sair com erro - tentar executar mesmo assim
    print("\n⚠ Tentando iniciar servidor apesar dos erros...")
else:
    print("✓ AMBIENTE PRONTO!")
print("="*60 + "\n")
VERIFY_EOF

# Limpeza de locks do HuggingFace
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
