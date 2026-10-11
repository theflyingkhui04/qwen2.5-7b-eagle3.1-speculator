import json
import pytest
from pathlib import Path
from export.package_and_upload import (
    build_hf_config,
    verify_package_integrity,
    parse_export_args,
)

def test_build_hf_config_for_vllm_float16():
    raw_cfg = {
        "hidden_size": 3584,
        "num_hidden_layers": 1,
        "num_attention_heads": 28,
        "target_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "target_layer": 28,
    }
    hf_cfg = build_hf_config(raw_cfg, precision="float16")
    assert hf_cfg["architectures"] == ["EagleQwen2ForCausalLM"]
    assert hf_cfg["torch_dtype"] == "float16"
    assert hf_cfg["hidden_size"] == 3584
    assert hf_cfg["target_layer"] == 28

def test_verify_package_integrity(tmp_path):
    # Thư mục thiếu file -> False
    assert verify_package_integrity(tmp_path) is False

    # Có đầy đủ config.json và model.safetensors -> True
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["EagleQwen2ForCausalLM"]}))
    (tmp_path / "model.safetensors").write_bytes(b"dummy_weights")
    assert verify_package_integrity(tmp_path) is True

def test_parse_export_args():
    args = parse_export_args(["--repo-id", "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator", "--precision", "float16"])
    assert args.repo_id == "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator"
    assert args.precision == "float16"
