import json
import yaml
from pathlib import Path

def test_model_config_integrity():
    root = Path(__file__).parent.parent
    config_path = root / "configs" / "eagle3_qwen2.5_7b.json"
    assert config_path.exists(), "configs/eagle3_qwen2.5_7b.json must exist"
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["hidden_size"] == 3584
    assert data["intermediate_size"] == 18944
    assert data["num_hidden_layers"] == 1
    assert data["num_attention_heads"] == 28
    assert data["num_key_value_heads"] == 4
    assert data["target_layer"] == 28
    assert data["use_fc_norm"] is True
    assert data.get("use_post_norm_feedback") is True
    assert data.get("eagle_version") == "3.1"
    assert "Eagle3_1Qwen2ForCausalLM" in data["architectures"]
    assert "EagleQwen2ForCausalLM" in data["architectures"]

    # Kiểm tra cả file eagle3.1_qwen2.5_7b.json
    eagle31_path = root / "configs" / "eagle3.1_qwen2.5_7b.json"
    assert eagle31_path.exists()

def test_train_config_integrity():
    root = Path(__file__).parent.parent
    config_path = root / "configs" / "train_config.yaml"
    assert config_path.exists(), "configs/train_config.yaml must exist"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["training"]["batch_size"] == 16
    assert data["training"]["gradient_accumulation_steps"] == 2
    assert float(data["training"]["learning_rate"]) == 3e-4
    assert data["training"]["epochs"] == 20
    assert data["training"]["warmup_steps"] == 500
    assert data["loss"]["smooth_l1_weight"] == 0.5
    assert data["safeguards"]["min_disk_free_gb"] == 50
    assert data["safeguards"]["max_runtime_hours"] == 6.0
