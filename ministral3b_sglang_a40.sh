#!/usr/bin/env bash
# ============================================================
# SCRIPT PARA RUNPOD - Ministral 3B com SGLang
# GPU: NVIDIA A40 (Ampere SM 8.6, 48GB VRAM)
# Repositório: https://github.com/waltagan/start_comand_modelo
# 
# CONFIGURAÇÃO DO RUNPOD:
# - Container image: nvidia/cuda:12.4.1-devel-ubuntu22.04
# - Container disk: 50 GB
# - Volume disk: 75 GB
# - Volume mount: /workspace
# - HTTP ports: 30000
# ============================================================

set -e

export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "============================================================"
echo "[BOOT] SGLang Server - Ministral 3B (Otimizado A40)"
echo "[INFO] PyTorch 2.4.1 | FlashInfer 0.2.x | CUDA 12.4"
echo "[DATA] $(date)"
echo "============================================================"

# --- ETAPA 1: DEPENDÊNCIAS DE SISTEMA ---
echo "[1/10] Instalando dependências do sistema..."
apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-pip python3-dev git wget \
    build-essential ffmpeg libsndfile1 libgl1 libglib2.0-0 \
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
echo "[6/10] Instalando Transformers..."
pip install --no-cache-dir \
    "transformers>=4.46.0,<5.0.0" "tokenizers>=0.21.0" \
    "huggingface_hub>=0.26.0" "mistral_common>=1.5.0" \
    "tiktoken>=0.7.0" protobuf sentencepiece

# --- ETAPA 7: DEPENDÊNCIAS DE RUNTIME ---
echo "[7/10] Instalando dependências de runtime..."
pip install --no-cache-dir \
    "pybase64>=1.4.0" "orjson>=3.10.0" "python-multipart>=0.0.9" \
    "pyzmq>=26.0.0" "uvloop>=0.19.0" "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" httptools watchfiles "pydantic>=2.0.0" \
    "outlines>=0.0.44" gguf interegular lark \
    psutil requests aiohttp tqdm regex rich setproctitle prometheus-client

# --- ETAPA 8: DEPENDÊNCIAS DO SGLANG ---
echo "[8/10] Instalando dependências do SGLang..."
pip install --no-cache-dir \
    ipython openai anthropic diskcache cloudpickle rpyc \
    partial-json-parser compressed-tensors scipy

# --- ETAPA 9: SGLANG CORE ---
echo "[9/10] Instalando SGLang..."
pip install --no-cache-dir --no-deps "sglang[all]>=0.4.3"
pip install --no-cache-dir sgl-kernel \
    -i https://flashinfer.ai/whl/cu124/torch2.4 2>/dev/null || true
pip install --no-cache-dir "numpy<2.0.0"

# --- ETAPA 10: VERIFICAÇÃO ---
echo "[10/10] Verificação..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

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

