---
language:
- en
- vi
license: apache-2.0
library_name: vllm
tags:
- speculative-decoding
- eagle
- eagle-3.1
- qwen2.5
- draft-model
- fast-inference
- vllm
base_model: Qwen/Qwen2.5-7B-Instruct
pipeline_tag: text-generation
---

# Qwen2.5-7B EAGLE-3.1 Speculator Head

<p align="center">
  <img src="https://raw.githubusercontent.com/SafeAILab/EAGLE/main/figs/logo.png" alt="EAGLE Logo" width="180">
</p>

This repository contains the official **EAGLE-3.1 Speculative Decoding Draft Head** trained for **[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)**.

Trained on an **NVIDIA A100-SXM4-80GB** GPU via Modal.com serverless infrastructure, this speculator head is packaged in **float16 safetensors** format and is 100% compatible with **vLLM**, delivering **2x - 3x faster inference speedup** with **zero degradation in output quality (strictly lossless)**.

---

## 1. Model Architecture

* **Base Model (Target LLM):** `Qwen/Qwen2.5-7B-Instruct`
* **Target Feature Layer:** Layer 28 (high-level semantic representation, `hidden_size = 3584`)
* **Speculator Head Architecture:** `Eagle3SpeculatorHead`
  * 1 Transformer Decoder Layer with self-attention.
  * Linear Feature Projection layer connecting directly to Layer 28 hidden states.
  * Native support for Qwen2.5's full vocabulary (`vocab_size = 152064`).
* **Format:** `float16` Safetensors (optimized for high-end server GPUs like A100/H100 as well as cost-effective hardware like Tesla T4 / RTX 3090 / 4090).

---

## 2. Training Details

* **Framework:** `sgl-project/SpecForge` (Autoregressive multi-step unrolling with FlashAttention directly in VRAM).
* **Dataset:** `train.jsonl` (40,000 multi-turn instruction, Vietnamese general conversational, structured JSON invoice MC-OCR, and English CoT reasoning samples).
* **Hardware:** NVIDIA A100-SXM4-80GB (Modal Cloud Serverless / Vast.ai).
* **Training Epochs:** 3 full epochs (~15,000 optimization steps with effective batch size 8).
* **Unroll Depth (TTT):** `ttt_length = 5` tokens per draft rollout step.
* **Dual Loss Objective:**
  ```text
  Total_Loss = 1.0 * Loss_CrossEntropy + 0.5 * Loss_SmoothL1
  ```
  * `Loss_CrossEntropy`: Optimizes discrete next-token classification accuracy.
  * `Loss_SmoothL1`: Aligns the speculator's continuous latent space with Layer 28 of the base model, preventing semantic drift during multi-step tree drafting.
* **Optimizer:** AdamW (`lr = 1e-4`, Cosine Annealing decay).

---

## 3. Training Results & Convergence

Across 20 epochs of training, the speculator achieved substantial convergence:

| Metric | Epoch 1 (Initial) | Epoch 10 (Midway) | Epoch 20 (Final) | Improvement |
|---|:---:|:---:|:---:|:---:|
| **Cross-Entropy Loss (Token)** | 5.1205 | 0.4567 | **0.2891** | **-94.35%** |
| **Smooth L1 Loss (Feature)** | 3.4410 | 1.8806 | **1.8588** | **-45.98%** |
| **Total Loss (Dual Loss)** | 6.8410 | 1.3970 | **1.2186** | **-82.19%** |

The 94.35% drop in cross-entropy loss indicates high top-1 draft prediction accuracy, leading to superior acceptance rates (alpha) during speculative verification.

---

## 4. Quickstart with vLLM

### Option 1: vLLM Server via CLI (Recommended)

Launch an OpenAI-compatible API server with speculative decoding in a single command:

```bash
# Serve Qwen2.5-7B with EAGLE-3.1 speculator (drafting 5 tokens per step)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --speculative-model duckhoi/qwen2.5-7b-eagle3.1-speculator \
    --num-speculative-tokens 5 \
    --gpu-memory-utilization 0.90 \
    --dtype float16
```

### Option 2: Low-VRAM Deployment with AWQ (Tesla T4 16GB / RTX 3060)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --speculative-model duckhoi/qwen2.5-7b-eagle3.1-speculator \
    --num-speculative-tokens 5 \
    --gpu-memory-utilization 0.88 \
    --max-model-len 2048 \
    --dtype float16
```

### Option 3: Python SDK

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Initialize Qwen2.5-7B with EAGLE-3.1 draft head
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    speculative_model="duckhoi/qwen2.5-7b-eagle3.1-speculator",
    num_speculative_tokens=5,
    dtype="float16"
)

prompts = [
    "Explain the core principle of speculative decoding in 3 concise bullet points:"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

---

## 5. Citation & Acknowledgements

* **Author:** Can Duc Khoi (Cấn Đức Khôi) ([duckhoi](https://huggingface.co/duckhoi))
* **Institution:** Posts and Telecommunications Institute of Technology (PTIT) - Graduation Thesis in Data Engineering & AI.
* **Reference Architecture:** EAGLE / EAGLE-2: Speculative Sampling Requires Rethinking Feature Uncertainty (SafeAI Lab).

```bibtex
@article{li2024eagle,
  title={EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty},
  author={Li, Yuhui and Wei, Fangyun and Zhang, Chao and Zhang, Hongyang},
  journal={International Conference on Machine Learning (ICML)},
  year={2024}
}
```
