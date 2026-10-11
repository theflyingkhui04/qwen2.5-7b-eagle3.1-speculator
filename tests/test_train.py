import pytest
from pathlib import Path
from training.train import parse_training_args, get_latest_checkpoint, load_training_config

def test_parse_training_args_defaults():
    args = parse_training_args(["--config", "configs/train_config.yaml"])
    assert args.config == "configs/train_config.yaml"
    assert args.resume is False

def test_get_latest_checkpoint_empty(tmp_path):
    assert get_latest_checkpoint(tmp_path) is None

def test_get_latest_checkpoint_sorted(tmp_path):
    (tmp_path / "checkpoint_epoch_5.pt").write_text("dummy5")
    (tmp_path / "checkpoint_epoch_10.pt").write_text("dummy10")
    (tmp_path / "checkpoint_epoch_15.pt").write_text("dummy15")
    latest = get_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint_epoch_15.pt"

def test_load_training_config():
    root = Path(__file__).parent.parent
    cfg = load_training_config(str(root / "configs" / "train_config.yaml"))
    assert "training" in cfg
    assert cfg["training"]["epochs"] == 20
    assert cfg["training"]["batch_size"] == 16
