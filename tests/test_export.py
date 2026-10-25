import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from export.package_and_upload import (
    attach_model_card,
    build_hf_config,
    parse_export_args,
    upload_to_huggingface,
    verify_package_integrity,
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
    assert "Eagle3_1Qwen2ForCausalLM" in hf_cfg["architectures"]
    assert "EagleQwen2ForCausalLM" in hf_cfg["architectures"]
    assert hf_cfg["eagle_version"] == "3.1"
    assert hf_cfg["use_post_norm_feedback"] is True
    assert hf_cfg["torch_dtype"] == "float16"
    assert hf_cfg["hidden_size"] == 3584
    assert hf_cfg["target_layer"] == 28


def test_verify_package_integrity(tmp_path):
    # Thư mục thiếu file -> False
    assert verify_package_integrity(tmp_path) is False

    # Có đầy đủ config.json và model.safetensors -> True
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Eagle3_1Qwen2ForCausalLM", "EagleQwen2ForCausalLM"]}),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors").write_bytes(b"dummy_weights")
    assert verify_package_integrity(tmp_path) is True


def test_attach_model_card(tmp_path):
    card_file = tmp_path / "MODEL_CARD.md"
    card_file.write_text("# Test Model Card Header", encoding="utf-8")
    export_dir = tmp_path / "checkpoint_dir"
    export_dir.mkdir()

    attach_model_card(str(export_dir), str(card_file))
    readme_file = export_dir / "README.md"
    assert readme_file.exists()
    assert "# Test Model Card Header" in readme_file.read_text(encoding="utf-8")


def test_parse_export_args():
    args = parse_export_args([
        "--output-dir", "checkpoints/test",
        "--repo-id", "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator",
        "--dry-run",
    ])
    assert args.output_dir == "checkpoints/test"
    assert args.repo_id == "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator"
    assert args.dry_run is True


def test_upload_to_huggingface_dry_run(tmp_path):
    # Chế độ dry-run không cần token thực tế
    res = upload_to_huggingface(
        export_dir=str(tmp_path),
        repo_id="theflyingkhui04/qwen2.5-7b-eagle3.1-speculator",
        dry_run=True,
    )
    assert res is True


def test_upload_to_huggingface_with_mock_api(tmp_path):
    import sys
    mock_api_instance = MagicMock()
    mock_hf_module = MagicMock()
    mock_hf_module.HfApi.return_value = mock_api_instance

    with patch.dict(sys.modules, {"huggingface_hub": mock_hf_module}):
        res = upload_to_huggingface(
            export_dir=str(tmp_path),
            repo_id="theflyingkhui04/qwen2.5-7b-eagle3.1-speculator",
            token="hf_mock_token",
            dry_run=False,
        )
        assert res is True
        mock_hf_module.HfApi.assert_called_once_with(token="hf_mock_token")
        mock_api_instance.create_repo.assert_called_once_with(
            repo_id="theflyingkhui04/qwen2.5-7b-eagle3.1-speculator",
            exist_ok=True,
            repo_type="model",
        )
        mock_api_instance.upload_folder.assert_called_once()
