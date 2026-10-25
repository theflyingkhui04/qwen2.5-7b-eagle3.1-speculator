import json
import yaml
from pathlib import Path


def test_eagle31_model_config_integrity():
    root = Path(__file__).parent.parent
    config_path = root / "configs" / "eagle3.1_qwen2.5_7b.json"
    assert config_path.exists(), "configs/eagle3.1_qwen2.5_7b.json must exist"
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["hidden_size"] == 3584
    assert data["intermediate_size"] == 18944
    assert data["num_hidden_layers"] == 1
    assert data["num_attention_heads"] == 28
    assert data["num_key_value_heads"] == 4
    assert data["vocab_size"] == 152064
    assert data["target_layer"] == 28
    assert data["use_fc_norm"] is True
    assert data["norm_output"] is True
    assert data.get("use_post_norm_feedback") is True
    assert data.get("eagle_version") == "3.1"
    assert "Eagle3_1Qwen2ForCausalLM" in data["architectures"]
    assert "EagleQwen2ForCausalLM" in data["architectures"]
    assert data["target_model_name_or_path"] == "Qwen/Qwen2.5-7B-Instruct"


def test_specforge_train_config_integrity():
    root = Path(__file__).parent.parent
    config_path = root / "configs" / "train_config.yaml"
    assert config_path.exists(), "configs/train_config.yaml must exist"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["target_model_path"] == "Qwen/Qwen2.5-7B-Instruct"
    assert data["draft_model_config"] == "configs/eagle3.1_qwen2.5_7b.json"
    assert data["train_data_path"] == "data/processed/train.jsonl"
    assert data["val_data_path"] == "data/processed/val.jsonl"
    assert data["output_dir"] == "checkpoints/qwen2.5-7b-eagle3.1"
    assert data["num_epochs"] == 3
    assert data["batch_size"] == 2
    assert data["gradient_accumulation_steps"] == 4
    assert float(data["learning_rate"]) == 1.0e-4
    assert data["max_seq_len"] == 2048
    assert data["ttt_length"] == 5
    assert data.get("tp_size", 1) == 1
