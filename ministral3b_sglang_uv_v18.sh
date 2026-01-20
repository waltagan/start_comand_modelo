#!/bin/bash
# ============================================================
# SGLang Server - Ministral 3B (UV Edition v18.0)
# SUPORTE RTX 5090 / BLACKWELL (SM 120)
# 
# Esta versão usa PyTorch Nightly com CUDA 12.8 para suportar
# GPUs Blackwell (RTX 50 series) que requerem SM 100/SM 120
# 
# MELHORIAS v18:
# - Removida flag --disable-flashinfer (obsoleta/ambígua)
# - Adicionada verificação de CUDA antes de iniciar
# - Instruções para usar container CUDA 12.8
# - Melhor tratamento de erros CUDA
# ============================================================
set -e

echo "============================================================"
echo "[BOOT] SGLang Server - Ministral 3B (UV Edition v18 - BLACKWELL)"
echo "[INFO] Suporte RTX 5090 / Blackwell SM 120"
echo "[INFO] PyTorch Nightly + CUDA 12.8"
echo "[INFO] IMPORTANTE: Use container CUDA 12.8 no RunPod!"
echo "[DATA] $(date)"
echo "============================================================"

# Verificar versão CUDA do container
echo "[CHECK] Verificando versão CUDA do container..."
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/' || echo "unknown")
echo "[CHECK] CUDA do container: $CUDA_VERSION"

if [[ "$CUDA_VERSION" != "12.8"* ]] && [[ "$CUDA_VERSION" != "unknown" ]]; then
    echo "⚠ AVISO: Container está usando CUDA $CUDA_VERSION, mas RTX 5090 requer CUDA 12.8"
    echo "⚠ Use container: nvidia/cuda:12.8.0-devel-ubuntu22.04 ou runpod/pytorch:2.8.0-py3.11-cuda12.8.1"
fi

# ============================================================
# [1/6] Instalar dependências do sistema e uv
# ============================================================
echo "[1/6] Instalando dependências do sistema e uv..."

# Atualizar repositórios
if ! apt-get update; then
    echo "ERRO: Falha ao atualizar repositórios apt"
    exit 1
fi

# Instalar dependências
if ! apt-get install -y --no-install-recommends \
    python3-venv \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libnuma1 \
    libnuma-dev \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0; then
    echo "ERRO: Falha ao instalar dependências do sistema"
    exit 1
fi

# Limpar cache do apt
rm -rf /var/lib/apt/lists/*

# Instalar uv
echo "[1.1/6] Instalando uv..."
if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
    echo "ERRO: Falha ao instalar uv. Tentando método alternativo..."
    # Método alternativo: instalar via pip
    python3 -m pip install --user uv || {
        echo "ERRO CRÍTICO: Não foi possível instalar uv"
        exit 1
    }
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verificar se uv está no PATH
if ! command -v uv &> /dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "ERRO CRÍTICO: uv não encontrado no PATH"
        exit 1
    fi
fi

echo "uv version: $(uv --version)"

# ============================================================
# [2/6] Criar ambiente virtual
# ============================================================
echo "[2/6] Criando ambiente virtual..."
uv venv /workspace/venv --python 3.10 --clear
source /workspace/venv/bin/activate

# ============================================================
# [3/6] Instalar PyTorch Nightly com CUDA 12.8 (suporte Blackwell)
# ============================================================
echo "[3/6] Instalando PyTorch Nightly cu128 (Blackwell support)..."

# IMPORTANTE: PyTorch Nightly com CUDA 12.8 é necessário para SM 120
if ! uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128; then
    echo "ERRO: Falha ao instalar PyTorch Nightly cu128"
    exit 1
fi

# Verificar instalação do PyTorch e CUDA
echo "[3.1/6] Verificando PyTorch e CUDA..."
python3 << 'PYTORCH_CHECK'
import torch
import sys

print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponível: {torch.cuda.is_available()}')

if not torch.cuda.is_available():
    print("⚠ ERRO: CUDA não está disponível no PyTorch!")
    print("⚠ Isso pode indicar:")
    print("  - Container não tem CUDA 12.8")
    print("  - Driver NVIDIA não está configurado")
    print("  - PyTorch não foi compilado com suporte CUDA")
    sys.exit(1)

print(f'CUDA Version: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')

# Verificar GPU
try:
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f'GPU: {gpu_name}')
    print(f'CUDA Capability: SM {cap[0]}{cap[1]}')
    
    if cap[0] >= 12:
        print('✓ GPU Blackwell detectada (SM 120)')
    elif cap[0] >= 10:
        print('⚠ GPU pode ser Blackwell mas capability não detectada corretamente')
    else:
        print(f'⚠ GPU não é Blackwell (SM {cap[0]}{cap[1]})')
except Exception as e:
    print(f'⚠ Erro ao detectar GPU: {e}')
    sys.exit(1)
PYTORCH_CHECK

# ============================================================
# [4/6] Instalar SGLang e dependências
# ============================================================
echo "[4/6] Instalando SGLang e dependências..."

# Instalar dependências core primeiro
uv pip install \
    numpy>=1.26 \
    packaging \
    setuptools \
    wheel \
    ninja \
    psutil \
    pydantic \
    fastapi \
    uvicorn \
    requests \
    aiohttp \
    msgspec \
    interegular \
    cloudpickle \
    orjson \
    rpyc \
    filelock

# Instalar Transformers versão específica (evitar incompatibilidade)
echo "[4.1/6] Instalando Transformers (versão compatível)..."
uv pip install "transformers>=4.47,<4.50" tokenizers sentencepiece accelerate

# FlashInfer: Não instalar para Blackwell ainda (sem suporte completo)
echo "[4.2/6] Pulando FlashInfer (sem suporte completo para SM 120)"

# Instalar SGLang do PyPI (versão mais recente)
echo "[4.3/6] Instalando SGLang..."
if ! uv pip install sglang; then
    echo "WARN: sglang do PyPI falhou, tentando do source..."
    uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git"
fi

# Verificar compressed_tensors
echo "[4.4/6] Verificando compressed_tensors..."
python3 -c "import compressed_tensors" 2>/dev/null && {
    echo "Atualizando compressed_tensors para versão compatível..."
    uv pip uninstall compressed_tensors -y 2>/dev/null || true
    uv pip install compressed-tensors --upgrade || true
} || echo "compressed_tensors não instalado, OK"

# ============================================================
# [5/6] Instalar suporte Mistral
# ============================================================
echo "[5/6] Instalando suporte a modelos Mistral..."
uv pip install mistral-common protobuf

# ============================================================
# [6/6] Verificação final
# ============================================================
echo "[6/6] Verificação..."

python3 << 'VERIFY_EOF'
import sys

def check(name, cmd):
    try:
        result = eval(cmd)
        print(f"✓ {name}: {result}")
        return True
    except Exception as e:
        print(f"⚠ {name}: {e}")
        return False

print("============================================================")
print("VERIFICAÇÃO - UV Edition v18.0 (Blackwell/SM120)")
print("============================================================")

# PyTorch
check("PyTorch", "__import__('torch').__version__")

# CUDA info detalhada
import torch
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"  Compute Capability: SM {cap[0]}{cap[1]}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    
    # Verificar se é Blackwell
    if cap[0] >= 12:
        print("  ✓ GPU Blackwell detectada (SM 120)")
    else:
        print(f"  ⚠ GPU não é Blackwell (SM {cap[0]}{cap[1]})")
else:
    print("  ✗ CUDA não disponível!")
    sys.exit(1)

# sgl_kernel
check("sgl_kernel", "'instalado' if __import__('importlib').util.find_spec('sgl_kernel') else 'não encontrado'")

# SGLang
check("SGLang", "__import__('sglang').__version__")

# Transformers
check("Transformers", "__import__('transformers').__version__")

# Numpy
check("Numpy", "__import__('numpy').__version__")

print("------------------------------------------------------------")

# Teste de importação crítica
try:
    from sglang.srt.entrypoints.http_server import launch_server
    print("✓ Import sglang.srt.entrypoints.http_server: OK")
except Exception as e:
    print(f"✗ Import falhou: {e}")
    sys.exit(1)

print("============================================================")
print("✓ AMBIENTE PRONTO PARA BLACKWELL!")
print("============================================================")
VERIFY_EOF

# ============================================================
# Iniciar servidor SGLang
# ============================================================
echo "============================================================"
echo "[BOOT] Launching SGLang Server..."
echo "[INFO] Modelo: mistralai/Ministral-3-3B-Instruct-2512"
echo "[INFO] Endpoint: http://0.0.0.0:30000"
echo "============================================================"

# Variáveis de ambiente para Blackwell
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"
export UV_LINK_MODE=copy

# IMPORTANTE: Removida flag --disable-flashinfer (obsoleta/ambígua)
# SGLang vai usar fallback automático se FlashInfer não estiver disponível
exec python3 -m sglang.launch_server \
    --model-path mistralai/Ministral-3-3B-Instruct-2512 \
    --port 30000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --dtype auto \
    --context-length 32768 \
    --disable-cuda-graph

