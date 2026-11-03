import json
from pathlib import Path


def test_deprecated_modules_and_tests_removed():
    root = Path(__file__).parent.parent
    deprecated_paths = [
        root / "extraction" / "extract_hidden_states.py",
        root / "extraction" / "__init__.py",
        root / "training" / "model.py",
        root / "training" / "loss.py",
        root / "configs" / "eagle3_qwen2.5_7b.json",
        root / "tests" / "test_extraction.py",
        root / "tests" / "test_loss.py",
        root / "tests" / "test_model.py",
    ]
    for p in deprecated_paths:
        assert not p.exists(), f"File cũ chưa được dọn dẹp: {p}"


def test_eagle3_1_config_pinned():
    root = Path(__file__).parent.parent
    cfg_path = root / "configs" / "eagle3.1_qwen2.5_7b.json"
    assert cfg_path.exists(), "Thiếu file cấu hình chuẩn eagle3.1_qwen2.5_7b.json"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg.get("num_hidden_layers") == 1, "EAGLE-3.1 phải có đúng 1 decoder layer"
    assert cfg.get("fc_norm") is True
    assert cfg.get("norm_output") is True, "EAGLE-3.1 bắt buộc norm_output: true"
    assert cfg.get("use_post_norm_feedback") is True, "EAGLE-3.1 bắt buộc use_post_norm_feedback: true"
    assert "target_layer" not in cfg  # PM-approved canonical three-layer capture
    assert cfg.get("hidden_size") == 3584, "hidden_size phải là 3584 cho Qwen2.5-7B"
