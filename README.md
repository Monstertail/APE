# <img src="assets/logo.png" width="40" height="40" align="top">  APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding [ICLR 2025]

### [[Paper](https://arxiv.org/abs/2502.05431)] | [[Project](https://infini-ai-lab.github.io/APE-Page)]

## TL;DR

We introduce APE for context-augmented generation with better efficiency and performance.

## System Support and Benchmark

### Environment Setup
```bash
conda create -yn ape-dev python=3.10
conda activate ape-dev

cd lmcache
pip install -e .

cd lmcache_vllm
pip install -e .
```

### Backend Selection
Set configure for the LMCache in .yaml. More parameters see [here](https://docs.lmcache.ai/configuration/v0/v0_config.html).
See [details here](https://docs.lmcache.ai/examples/v0/backend.html) for more backends of storage.

### Bench RAG
Follow the instruction [here](lmcache/benchmarks/rag/README.md). We adopt "meta-llama/Llama-3.1-8B-Instruct" here.

Start a vLLM server:
```bash
# Rope is not supported(No blend for Llama-3.1-8B-Instruction)
# vllm serve meta-llama/Llama-3.1-8B-Instruct --disable-log-requests
vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests
```

Bench vLLM

```bash
bash launch_vllm.sh
```

Start lm-cache-vLLM server:
```bash
# Rope is not supported(No blend for Llama-3.1-8B-Instruction)
# LMCACHE_CONFIG_FILE=example_blending.yaml python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.7 --port 8000
LMCACHE_CONFIG_FILE=example_blending.yaml python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.7 --port 8000
```

Bench lmcache-vLLM

```bash
bash launch_lmcache.sh
```

Note: For model selection, only those with  rope_scaling=None can blend KV cache. See [here](https://github.com/LMCache/LMCache/issues/242).


## Usage

### Environment Setup

```bash
conda create -yn ape python=3.10
conda activate ape

pip install -r requirements.txt
python setup.py install
```

## Run Context-augmented Question Answering with APE

By default, the temperature and scaling factor are set to 0.9, preserving over 90% performance on few-shot tasks.

```bash
CUDA_VISIBLE_DEVICES=0 python demo_APE.py --model llama3-8b-instruct
```

## Experiments

To reproduce the APE results for retrieval-augmented generation (RAG) and in-context learning (ICL) tasks in Section 5, please follow the instructions and use the code provided in the `experiments` directory.

## TODOs
We will release the code and data in the following order, please stay tuned!

- [x] Release core code of APE, including Llama-3, Llama-3.1, Mistral-v0.3, and Gemma-2.
- [x] Release RAG and ICL evaluation code.
- [x] Release APE context-augmented QA demo
- [ ] Incorporate APE into efficient inference engine

## Citation

If you find APE useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@inproceedings{yang2025ape,
  title={APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding},
  author={Yang, Xinyu and Chen, Tianqi and Chen, Beidi},
  booktitle={ICLR 2025},
  year={2025}
}
```
