# Ministral 3B com SGLang

Deploy do modelo `mistralai/Ministral-3-3B-Instruct-2512` usando SGLang como motor de inferência.

## Stack de Tecnologia

### Para GPUs Ampere/Ada (A40, A100, RTX 4090)

| Componente | Versão | Motivo |
|------------|--------|--------|
| **PyTorch** | 2.4.1+cu124 | Base estável, evita conflitos de ABI |
| **CUDA** | 12.4 | Compatível com RunPod e PyTorch 2.4 |
| **FlashInfer** | 0.2.x (torch2.4) | Kernels otimizados para atenção |
| **SGLang** | >= 0.4.3 | Motor de inferência com RadixAttention |
| **Transformers** | >= 4.46.0 | Suporte ao tokenizer Tekken |

### Para GPUs Blackwell (RTX 5090, RTX 5080, RTX 5070 Ti) ⚠️

| Componente | Versão | Motivo |
|------------|--------|--------|
| **PyTorch** | Nightly cu128 | Único com suporte SM 120 |
| **CUDA** | 12.8 | Mínimo para Blackwell |
| **FlashInfer** | Desabilitado | Ainda sem suporte completo SM 120 |
| **SGLang** | Latest | Instalação com flags especiais |
| **Transformers** | 4.47-4.49 | Evita conflito com compressed_tensors |

> **Script para Blackwell**: Use `runpod_start_uv_v16.txt`

## Padrões Importantes

### Estratégia "Goldilocks"

O principal desafio é a **dependência circular** entre versões:

1. SGLang moderno precisa de FlashInfer 0.2.x
2. FlashInfer 0.2.x padrão requer PyTorch 2.5
3. Precisamos manter PyTorch 2.4.1

**Solução**: Instalar FlashInfer do índice específico para PyTorch 2.4:
```bash
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.4
```

### Ordem de Instalação Crítica

1. **PyTorch 2.4.1** (primeiro, define a ABI)
2. **FlashInfer** (do índice torch2.4)
3. **Transformers >= 4.46** (para tokenizer Tekken)
4. **Dependências de runtime** (numpy<2, fastapi, etc)
5. **SGLang com `--no-deps`** (evita upgrade do PyTorch)

### Tokenizer Tekken

O Ministral 3B usa o tokenizer **Tekken** (vocab ~131k tokens). Versões antigas do `transformers` não suportam corretamente, causando:
- Geração de texto sem sentido
- Tokens de parada não reconhecidos

**Requisito**: `transformers >= 4.46.0`

## Arquivos do Projeto

| Arquivo | Descrição |
|---------|-----------|
| `linha_comando_v2.sh` | Script completo para container/RunPod |
| `teste_local.sh` | Script interativo para teste em etapas |
| `start_server.sh` | Inicia apenas o servidor |

## Como Testar Localmente

### Pré-requisitos

- GPU NVIDIA com CUDA 12.4
- Docker (recomendado) ou ambiente Linux com drivers NVIDIA

### Opção 1: Teste em Etapas (Recomendado)

```bash
# Tornar executável
chmod +x teste_local.sh

# Executar interativamente
./teste_local.sh

# Ou executar etapa específica
./teste_local.sh 1  # Verificar CUDA
./teste_local.sh 3  # Instalar PyTorch
./teste_local.sh 9  # Verificação completa
```

### Opção 2: Execução Completa

```bash
# Executar todas as etapas
./teste_local.sh all

# Depois iniciar servidor
./start_server.sh
```

### Opção 3: Docker Local

```bash
# Construir imagem de teste
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -p 30000:30000 \
    nvidia/cuda:12.4.1-devel-ubuntu22.04 \
    bash /workspace/linha_comando_v2.sh
```

## Deploy no RunPod

1. Criar Pod com template **NVIDIA CUDA 12.4**
2. Upload do `linha_comando_v2.sh` para `/workspace`
3. Executar:
```bash
chmod +x /workspace/linha_comando_v2.sh
/workspace/linha_comando_v2.sh
```

## Verificação do Ambiente

Após instalação, execute:

```python
import torch
import transformers
import sglang

print(f"PyTorch: {torch.__version__}")  # Deve ser 2.4.1
print(f"Transformers: {transformers.__version__}")  # >= 4.46
print(f"SGLang: {sglang.__version__}")  # >= 0.4.3

# Teste do tokenizer
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "mistralai/Ministral-3-3B-Instruct-2512",
    trust_remote_code=True
)
print(f"Vocab size: {tok.vocab_size}")  # Deve ser ~131k
```

## Endpoints da API

Após servidor iniciado:

```bash
# Health check
curl http://localhost:30000/health

# Completions (compatível com OpenAI)
curl http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ministral-3b",
        "prompt": "Olá, como você está?",
        "max_tokens": 100
    }'

# Chat completions
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ministral-3b",
        "messages": [{"role": "user", "content": "Olá!"}],
        "max_tokens": 100
    }'
```

## Troubleshooting

### Erro: `undefined symbol: _ZN3c10...`

**Causa**: Incompatibilidade de ABI entre PyTorch e FlashInfer/sgl-kernel.

**Solução**: Reinstalar FlashInfer do índice correto:
```bash
pip uninstall flashinfer-python flashinfer -y
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.4
```

### Erro: Tokenizer com vocab pequeno (~32k)

**Causa**: Transformers antigo carregando tokenizer legado.

**Solução**:
```bash
pip install "transformers>=4.46.0"
```

### Erro: Out of Memory (OOM)

**Solução**: Reduzir fração de memória:
```bash
export MEM_FRACTION=0.75
./start_server.sh
```

### FlashInfer não carrega

Se aparecer warning sobre JIT, é normal - o kernel será compilado na primeira execução. Isso requer `nvcc` disponível (imagem `devel`).

### Erro: SM 120 not compatible (RTX 5090/Blackwell)

**Mensagem**:
```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

**Causa**: PyTorch estável (cu124) não suporta arquitetura Blackwell.

**Solução**: Use o script `runpod_start_uv_v16.txt` que instala PyTorch Nightly com CUDA 12.8:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Erro: No module named 'transformers.masking_utils'

**Causa**: Incompatibilidade entre `compressed_tensors` e Transformers.

**Solução**: Use Transformers 4.47-4.49:
```bash
pip install "transformers>=4.47,<4.50"
```

## Compatibilidade de GPUs

| GPU | Arquitetura | SM | CUDA | Script |
|-----|-------------|----|----- |--------|
| A40 | Ampere | 86 | 12.4 | v13-v15 |
| A100 | Ampere | 80 | 12.4 | v13-v15 |
| RTX 4090 | Ada Lovelace | 89 | 12.4 | v13-v15 |
| RTX 5090 | **Blackwell** | **120** | **12.8** | **v16** |
| RTX 5080 | Blackwell | 120 | 12.8 | v16 |
| RTX 5070 Ti | Blackwell | 120 | 12.8 | v16 |
| H100 | Hopper | 90 | 12.4 | v13-v15 |

