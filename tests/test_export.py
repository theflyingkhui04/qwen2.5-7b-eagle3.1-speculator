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


def test_packaging_preserves_native_bfloat16_config():
    raw_cfg = {
        "hidden_size": 3584,
        "num_hidden_layers": 1,
        "num_attention_heads": 28,
        "target_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "architectures": ["LlamaForCausalLMEagle3"],
        "torch_dtype": "bfloat16",
    }
    hf_cfg = build_hf_config(raw_cfg)
    assert hf_cfg["architectures"] == ["LlamaForCausalLMEagle3"]
    assert hf_cfg["torch_dtype"] == "bfloat16"
    assert hf_cfg["hidden_size"] == 3584
    assert "target_layer" not in hf_cfg


def test_verify_package_integrity(tmp_path):
    # Thư mục thiếu file -> False
    assert verify_package_integrity(tmp_path) is False

    # Có đầy đủ config.json và model.safetensors -> True
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLMEagle3"], "torch_dtype": "bfloat16"}),
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


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_packaging_native_serialized_dtype(tmp_path, dtype):
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig

    # Native save_pretrained writes dtype, not the legacy torch_dtype key.
    LlamaConfig(architectures=["LlamaForCausalLMEagle3"], dtype=dtype).save_pretrained(tmp_path)
    save_file({"lm_head.weight": torch.zeros(1, dtype=getattr(torch, dtype))},
              str(tmp_path / "model.safetensors"))
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["dtype"] == dtype and "torch_dtype" not in config
    if dtype == "bfloat16":
        assert build_hf_config(config) == config
        assert verify_package_integrity(tmp_path)
    else:
        with pytest.raises(ValueError, match="BF16"):
            build_hf_config(config)
        assert not verify_package_integrity(tmp_path)
