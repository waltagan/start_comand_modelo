# RTX 5090 - Erros CUDA Initialization e Flags Obsoletas

## Problemas Identificados nos Logs (v17)

### 1. Erro de Inicialização CUDA

**Mensagem**:
```
CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero.
```

**Causa**:
- Container está usando **CUDA 12.4** mas precisa de **CUDA 12.8** para Blackwell
- PyTorch Nightly cu128 não consegue inicializar CUDA em container cu124
- Incompatibilidade entre versão do container e versão do PyTorch

**Solução**:
- Usar container com CUDA 12.8 no RunPod:
  - `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu` (recomendado)
  - `nvidia/cuda:12.8.0-devel-ubuntu22.04`

### 2. Flag `--disable-flashinfer` Ambígua

**Mensagem**:
```
launch_server.py: error: ambiguous option: --disable-flashinfer could match --disable-flashinfer-autotune, --disable-flashinfer-cutlass-moe-fp4-allgather
```

**Causa**:
- A flag `--disable-flashinfer` foi removida ou tornou-se ambígua no SGLang mais recente
- Existem múltiplas flags relacionadas a FlashInfer que começam com o mesmo prefixo

**Solução**:
- Remover a flag `--disable-flashinfer` completamente
- SGLang vai usar fallback automático se FlashInfer não estiver disponível
- FlashInfer não tem suporte completo para SM 120 ainda, então não é necessário

### 3. Triton Não Suportado

**Mensagem**:
```
Triton is not supported on current platform, roll back to CPU.
```

**Causa**:
- Triton pode não ter suporte completo para Blackwell ainda
- Fallback para CPU é normal e aceitável

**Solução**:
- Aceitar o fallback para CPU (não crítico)
- SGLang vai funcionar mesmo sem Triton

### 4. Failed to Get Device Capability

**Mensagem**:
```
Failed to get device capability: CUDA unknown error
```

**Causa**:
- Mesmo problema de inicialização CUDA
- PyTorch não consegue acessar a GPU devido à incompatibilidade de versão

**Solução**:
- Usar container CUDA 12.8 (mesma solução do problema 1)

## Correções Aplicadas na v18

1. **Verificação de CUDA do Container**: Script agora verifica a versão CUDA do container antes de iniciar
2. **Remoção da Flag Obsoleta**: Removida `--disable-flashinfer` que causava erro
3. **Validação de CUDA**: Verificação se CUDA está disponível antes de continuar
4. **Instruções de Container**: Documentação clara sobre qual container usar no RunPod
5. **Melhor Tratamento de Erros**: Mensagens de erro mais claras e informativas

## Como Usar a v18

### 1. Configurar Container no RunPod

**IMPORTANTE**: Use um destes containers:

- **Recomendado**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu`
- **Alternativa**: `nvidia/cuda:12.8.0-devel-ubuntu22.04`

**NÃO use**: `nvidia/cuda:12.4.1-devel-ubuntu22.04` (não suporta Blackwell)

### 2. Usar o Comando v18

Cole o comando do arquivo `runpod_start_uv_v18.txt` no campo "Container Start Command" do RunPod.

## Referências

- [RunPod PyTorch 2.8 + CUDA 12.8 Guide](https://www.runpod.io/articles/guides/pytorch-2-8-cuda-12-8)
- [SGLang Issue #1146 - disable-flashinfer](https://github.com/sgl-project/sglang/issues/1146)
- [PyTorch Blackwell Support](https://github.com/pytorch/pytorch/issues/159207)

## Data da Solução

2026-01-20

