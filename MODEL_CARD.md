---
language:
- en
- vi
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- speculative-decoding
- eagle-3.1
- draft-model
---

# Qwen2.5-7B EAGLE-3.1 training candidate

This is a training candidate description, not evidence of a trained or released model.
No new M3-04 weights, convergence results, acceptance rate, speedup, or serving
compatibility have been established by the migration.

The frozen target is Qwen/Qwen2.5-7B-Instruct. The draft uses the registered
LlamaForCausalLMEagle3 architecture with one decoder layer, fc_norm and norm_output.
BF16, offline training and offline evaluation are canonical. Current SpecForge
resolves three auxiliary capture layers from the target configuration.
PM selected target/tokenizer vocab_size=152064 and draft_vocab_size=16000,
matching the pinned Qwen2.5-7B EAGLE3 recipe. The draft output head has 16000
entries; deterministic vocabulary mapping is derived from training features only.

Training intent: 40,000 training conversations; 3,000 validation conversations;
3 epochs; batch 2; accumulation 4; learning rate 1e-4; max length 2048; TTT 5.
The 2,000 test conversations remain outside training and checkpoint selection.
All raw turns are preserved by project-local preparation before explicit token truncation.

The runtime is SpecForge at fddcb4ad37733a821708170df28cac2f7bb6ea05.
Its default EAGLE3 objective is teacher-distribution log-softmax distillation,
not the historical CE + SmoothL1 objective. Canonical selection maximizes
eval/simulated_acc_len and publishes <run_id>-best.

Runtime checkpoints must pass native SpecForge export before packaging.
Exported dtype must match actual BF16 tensors. Do not infer vLLM/SGLang serving
compatibility or numerical quality from config/unit tests. Historical logs remain
in the repository and do not establish results for this migrated runtime.

Author: Cấn Đức Khôi. Training operations: Bùi Thiên Đại.
See README.md and docs/m3-04-migration.md for the current Modal workflow.
