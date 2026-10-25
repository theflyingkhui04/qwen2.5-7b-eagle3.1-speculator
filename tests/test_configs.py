import json
from pathlib import Path
from training.train import SPECFORGE_SHA, load_training_config

ROOT = Path(__file__).resolve().parents[1]


def test_eagle31_model_config_integrity():
    data = json.loads((ROOT / "configs/eagle3.1_qwen2.5_7b.json").read_text())
    assert data["architectures"] == ["LlamaForCausalLMEagle3"]
    assert data["model_type"] == "llama"
    assert data["num_hidden_layers"] == 1
    assert data["hidden_size"] == 3584
    assert data["num_attention_heads"] == 28
    assert data["num_key_value_heads"] == 4
    assert data["vocab_size"] == 152064  # Target/tokenizer input vocabulary.
    assert data["draft_vocab_size"] == 16000  # PM-approved draft output head.
    assert data["fc_norm"] and data["norm_output"] and data["use_post_norm_feedback"]
    assert data["torch_dtype"] == "bfloat16"
    assert "target_layer" not in data
    assert "use_fc_norm" not in data


def test_typed_offline_config_contract():
    cfg = load_training_config()
    assert set(cfg) == {"model", "data", "training", "tracking", "deployment", "run_id", "output_dir"}
    assert cfg["model"]["target_model_path"] == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg["model"]["torch_dtype"] == "bfloat16"
    assert cfg["model"].get("aux_hidden_state_layer_ids") is None
    assert cfg["data"]["hidden_states_path"].startswith("/vol/hidden_states/m3-04/train/")
    assert cfg["data"]["eval_hidden_states_path"].startswith("/vol/hidden_states/m3-04/eval/")
    assert not cfg["data"]["eval_hidden_states_path"].endswith(".jsonl")
    assert not cfg["data"].get("train_data_path")
    assert cfg["data"]["chat_template"] == "qwen"
    assert cfg["data"]["is_preformatted"]
    assert cfg["data"]["max_length"] == 2048
    training = cfg["training"]
    assert training["strategy"] == "eagle3"
    assert training["num_epochs"] == 3
    assert training["batch_size"] == 2
    assert training["accumulation_steps"] == 4
    assert training["learning_rate"] == 1e-4
    assert training["ttt_length"] == 5
    assert training["eval_interval"] > 0
    assert cfg["tracking"]["report_to"] == "wandb"
    assert "wandb_key" not in cfg["tracking"]


def test_immutable_dependency_ownership():
    req = (ROOT / "requirements.txt").read_text()
    assert "SpecForge.git@" + SPECFORGE_SHA in req
    deps = [line for line in req.splitlines() if line and not line.startswith("#")]
    assert not any(line.startswith(("torch", "transformers", "sglang")) for line in deps)
