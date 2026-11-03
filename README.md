---
language:
- en
- vi
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: transformers
tags:
- speculative-decoding
- eagle-3.1
- draft-model
- specforge
- vllm
- sglang
pipeline_tag: text-generation
inference: false
---

# Qwen2.5-7B EAGLE-3.1 Speculator

Official EAGLE-3.1 speculative decoding draft head for **[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)**, developed with `SpecForge`.

This draft head accelerates text generation in high-throughput inference engines (**SGLang** and **vLLM**) via speculative decoding. It produces mathematically identical, lossless outputs compared to standard autoregressive generation from the base target model.

---

## Model Summary

- **Base Model (Target):** [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (7.61B parameters)
- **Draft Head Architecture:** `LlamaForCausalLMEagle3` (EAGLE-3.1 Tree-Attention Draft Model)
- **Decoder Layers:** 1 Transformer decoder layer
- **Hidden Size:** 3,584
- **Draft Vocabulary Size:** 16,000 tokens (canonical frequency-mapped draft vocabulary)
- **Attention Configuration:** Grouped-Query Attention (GQA) — 28 Query Heads, 4 Key/Value Heads
- **Intermediate Capture Layers:** `[1, 13, 24]` from `Qwen/Qwen2.5-7B-Instruct`
- **Feature Normalization:** `fc_norm: true`, `norm_output: true`, `use_post_norm_feedback: true`
- **Precision:** `bfloat16`
- **Trained With:** `SpecForge`

---

## Evaluation & Performance

Evaluated on an independent validation set of **3,000 multi-turn conversations**:

| Metric | Result |
| :--- | :---: |
| **Average Acceptance Length (`simulated_acc_len`)** | **1.58 tokens / step** |
| **Overall Top-1 Drafting Accuracy** | **66.37%** |
| **Validation Loss** | **2.505** |

### Per-Position Drafting Accuracy (5 Speculative Steps)

| Speculative Position | Top-1 Accuracy | Expected Acceptance Rate |
| :---: | :---: | :---: |
| **Position 1 (Token +1)** | **66.37%** | 66.44% |
| **Position 2 (Token +2)** | **62.88%** | 62.92% |
| **Position 3 (Token +3)** | **61.17%** | 60.98% |
| **Position 4 (Token +4)** | **59.91%** | 59.62% |
| **Position 5 (Token +5)** | **58.94%** | 58.54% |

---

## How EAGLE-3.1 Works

EAGLE-3.1 is an advanced speculative decoding architecture designed for low-overhead, high-acceptance drafting:
1. **Feature-Level Conditioning:** Instead of drafting tokens from text token embeddings alone, the draft head leverages intermediate hidden representations extracted from layers `[1, 13, 24]` of the target model.
2. **Tree-Attention Drafting:** The speculator constructs a candidate token tree spanning multiple future time steps.
3. **Single Verification Pass:** The target model (`Qwen2.5-7B-Instruct`) validates the candidate tree in a single forward pass using Tree Attention. Accepted tokens are emitted directly, guaranteeing output distribution equivalence to standard generation.

---

## Serving & Deployment

This draft head is natively compatible with **SGLang** and **vLLM**.

### 1. Serving with SGLang (Recommended)

#### Launch Server (CLI)
```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --speculative-algorithm EAGLE \
  --speculative-draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
  --speculative-num-steps 5 \
  --port 30000
```

#### Client Request (Python / OpenAI-Compatible)
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain speculative decoding in 3 sentences."}
    ],
    temperature=0.0,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

---

### 2. Serving with vLLM

#### Launch Server (CLI)
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --speculative-model duckhoi/qwen2.5-7b-eagle3.1-speculator \
  --num-speculative-tokens 5 \
  --port 8000
```

#### Offline Inference (Python)
```python
from vllm import LLM, SamplingParams

prompts = ["Explain quantum computing in simple terms:"]
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    speculative_model="duckhoi/qwen2.5-7b-eagle3.1-speculator",
    num_speculative_tokens=5,
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

---

## Model Files

| File | Description |
| :--- | :--- |
| `model.safetensors` | Speculator head weights serialized in `bfloat16` format (~1.78 GB) |
| `config.json` | EAGLE-3.1 draft head architecture configuration |
| `README.md` | Model card and deployment documentation |

---

## Citation & License

- **License:** Apache 2.0
- **Base Model License:** Apache 2.0 (Qwen2.5 Community License / Apache 2.0)
- **Author:** Cấn Đức Khôi (`@duckhoi`)
