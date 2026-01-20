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

Cole este comando no campo **"Container Start Command"** do RunPod:

```bash
bash -c "curl -sSL https://raw.githubusercontent.com/waltagan/start_comand_modelo/main/ministral3b_sglang_a40.sh | bash"
```

## Componentes Instalados

- PyTorch 2.4.1 (CUDA 12.4)
- FlashInfer 0.2.x (kernels otimizados para Ampere)
- SGLang 0.4.x
- Transformers 4.46+ (suporte ao tokenizer Tekken)

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

