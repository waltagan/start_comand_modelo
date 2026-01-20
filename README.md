# Start Command - Ministral 3B com SGLang

Script otimizado para executar o modelo **Ministral-3-3B-Instruct-2512** no RunPod com GPU **NVIDIA A40**.

## Configuração do RunPod

| Campo | Valor |
|-------|-------|
| **Container image** | `nvidia/cuda:12.4.1-devel-ubuntu22.04` |
| **Container disk** | 50 GB |
| **Volume disk** | 75 GB |
| **Volume mount** | `/workspace` |
| **HTTP ports** | `30000` |
| **TCP ports** | `22` |

## Start Command

### Para GPUs Ampere/Ada (A40, A100, RTX 4090)

Cole este comando no campo **"Container Start Command"** do RunPod:

```bash
bash -c "curl -sSL https://raw.githubusercontent.com/waltagan/start_comand_modelo/main/ministral3b_sglang_a40.sh | bash"
```

### Para GPUs Blackwell (RTX 5090, RTX 5080, RTX 5070 Ti) ⚠️

**IMPORTANTE**: GPUs Blackwell (SM 120) requerem PyTorch Nightly com CUDA 12.8:

```bash
bash -c 'apt-get update && apt-get install -y curl && curl -sL https://raw.githubusercontent.com/waltagan/start_comand_modelo/main/ministral3b_sglang_uv_v16.sh > /tmp/start.sh && bash /tmp/start.sh'
```

> **Nota**: O script v16 usa PyTorch Nightly cu128 que é o único com suporte para SM 120. Veja [fixes/rtx5090_blackwell_sm120_incompatibility.md](fixes/rtx5090_blackwell_sm120_incompatibility.md) para mais detalhes.

## Componentes Instalados

### Para GPUs Ampere/Ada
- PyTorch 2.4.1 (CUDA 12.4)
- FlashInfer 0.2.x (kernels otimizados para Ampere)
- SGLang 0.4.x
- Transformers 4.46+ (suporte ao tokenizer Tekken)

### Para GPUs Blackwell (v16)
- PyTorch Nightly (CUDA 12.8) - suporte SM 120
- SGLang Latest
- Transformers 4.47-4.49 (compatibilidade)
- FlashInfer desabilitado (ainda sem suporte completo SM 120)

## Endpoint da API

Após inicialização (10-15 min), a API estará disponível em:

```
http://<POD_IP>:30000/v1/chat/completions
```

### Exemplo de uso:

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ministral-3b",
        "messages": [{"role": "user", "content": "Olá!"}],
        "max_tokens": 100
    }'
```

## Otimizações para A40

- `--mem-fraction-static 0.90`: Usa 43GB dos 48GB disponíveis
- `--max-running-requests 64`: Mais requests paralelas
- `TORCH_CUDA_ARCH_LIST=8.6`: Kernels compilados para Ampere

## Compatibilidade de GPUs

| GPU | Arquitetura | SM | CUDA | Script |
|-----|-------------|----|----- |--------|
| A40 | Ampere | 86 | 12.4 | a40.sh, uv_v13-v15 |
| A100 | Ampere | 80 | 12.4 | uv_v13-v15 |
| RTX 4090 | Ada Lovelace | 89 | 12.4 | uv_v13-v15 |
| **RTX 5090** | **Blackwell** | **120** | **12.8** | **uv_v16** |
| RTX 5080 | Blackwell | 120 | 12.8 | uv_v16 |
| RTX 5070 Ti | Blackwell | 120 | 12.8 | uv_v16 |
| H100 | Hopper | 90 | 12.4 | uv_v13-v15 |

