# RTX 5090 (Blackwell SM 120) - Incompatibilidade PyTorch

## Problema

Ao tentar rodar SGLang em uma **RTX 5090** (arquitetura Blackwell), o seguinte erro aparece:

```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

Além disso, ocorre um erro de importação:

```
ModuleNotFoundError: No module named 'transformers.masking_utils'
```

## Causa

1. **PyTorch cu124 não suporta Blackwell**: A RTX 5090 usa arquitetura **Blackwell** com compute capability **SM 120**. PyTorch estável com CUDA 12.4 só suporta até **SM 90**.

2. **compressed_tensors incompatível**: O pacote `compressed_tensors` importa um módulo (`transformers.masking_utils`) que não existe em versões recentes do Transformers.

## Solução

### 1. Usar PyTorch Nightly com CUDA 12.8

O PyTorch Nightly com CUDA 12.8 (cu128) inclui suporte para Blackwell:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 2. Usar Transformers versão compatível

Usar uma versão específica do Transformers que funciona com compressed_tensors:

```bash
pip install "transformers>=4.47,<4.50"
```

### 3. Desabilitar otimizações não suportadas

Ao iniciar o SGLang, usar flags para desabilitar funcionalidades que podem não funcionar em Blackwell:

```bash
python -m sglang.launch_server \
    --model-path mistralai/Ministral-3-3B-Instruct-2512 \
    --disable-cuda-graph \
    --disable-flashinfer
```

## Scripts

- **v16**: `runpod_start_uv_v16.txt` - Script específico para RTX 5090/Blackwell

## Referências

- [PyTorch Issue #159207](https://github.com/pytorch/pytorch/issues/159207) - Add official support for CUDA sm_120
- [PyTorch Forum](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099) - Blackwell support discussion
- [ComfyUI Discussion #6643](https://github.com/Comfy-Org/ComfyUI/discussions/6643) - Nvidia 50 Series support thread

## Informações Técnicas

| GPU | Arquitetura | Compute Capability | CUDA Mínimo |
|-----|-------------|-------------------|-------------|
| RTX 4090 | Ada Lovelace | SM 89 | 12.0 |
| RTX 5090 | Blackwell | SM 120 | 12.8 |
| RTX 5080 | Blackwell | SM 120 | 12.8 |
| RTX 5070 Ti | Blackwell | SM 120 | 12.8 |
| A100 | Ampere | SM 80 | 11.0 |
| H100 | Hopper | SM 90 | 12.0 |

## Data da Solução

2026-01-20

