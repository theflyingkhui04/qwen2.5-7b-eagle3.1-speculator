from pathlib import Path
from unittest.mock import patch
import pytest

from training.train import (
    SPECFORGE_SHA, BEST_METRIC, build_specforge_command, load_training_config,
    resolved_config, feature_counts, feature_paths, feature_selection,
    run_specforge_training, write_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_unified_entry_and_plan():
    assert build_specforge_command("resolved.yaml") == ["specforge", "train", "-c", "resolved.yaml"]
    assert build_specforge_command("resolved.yaml", plan=True)[-1] == "--plan"
    assert BEST_METRIC == "eval/simulated_acc_len"


def test_dry_run_never_executes():
    with patch("training.train.subprocess.run") as run:
        assert run_specforge_training("x.yaml", dry_run=True)[0] == "specforge"
        run.assert_not_called()


def test_windows_cannot_launch_training():
    with patch("training.train.sys.platform", "win32"), pytest.raises(RuntimeError, match="Modal Linux"):
        run_specforge_training("x.yaml")


def test_smoke_changes_only_scale_and_run_identity():
    cfg = load_training_config()
    smoke = resolved_config(cfg, smoke=True, run_suffix="unique")
    assert cfg["run_id"] != smoke["run_id"]
    assert smoke["training"]["max_steps"] == 2
    for field in ("ttt_length", "num_epochs", "batch_size", "accumulation_steps", "learning_rate", "strategy"):
        assert smoke["training"][field] == cfg["training"][field]
    assert smoke["model"] == cfg["model"]
    assert smoke["data"] == cfg["data"]
    assert smoke["training"]["eval_interval"] == 1
    assert smoke["tracking"]["wandb_project"] == cfg["tracking"]["wandb_project"]


def test_resume_is_full_resume_not_warm_start():
    cfg = resolved_config(load_training_config(), resume=True)
    assert cfg["training"]["resume_from"].endswith(cfg["run_id"] + "-latest")
    assert "draft_checkpoint_path" not in cfg["model"]


def test_smoke_requires_unique_identity():
    with pytest.raises(ValueError):
        resolved_config(load_training_config(), smoke=True)


def test_content_addressed_features_and_exclusive_config(tmp_path):
    cfg = load_training_config()
    draft = ROOT / "configs/eagle3.1_qwen2.5_7b.json"
    resolved, key, contract = feature_paths(cfg, draft)
    assert resolved["data"]["hidden_states_path"].endswith("/train/" + key)
    assert resolved["data"]["eval_hidden_states_path"].endswith("/eval/" + key)
    assert contract["specforge_sha"] == SPECFORGE_SHA
    assert key == feature_paths(cfg, draft)[1]
    path = write_config(resolved, tmp_path / "run.yaml")
    assert load_training_config(path) == resolved
    with pytest.raises(FileExistsError):
        write_config(resolved, path)


def test_no_duplicate_tracker_or_legacy_training_path():
    source = (ROOT / "training/train.py").read_text()
    assert "wandb.init" not in source
    assert "specforge.train" not in source
    assert "torchrun" not in source


def test_smoke_defaults_full_counts_and_separate_identity():
    cfg = load_training_config()
    draft = ROOT / "configs/eagle3.1_qwen2.5_7b.json"
    full, full_key, full_contract = feature_paths(cfg, draft)
    options = {"feature_scope": "smoke", "target_revision": "immutable-target", "capture_layers": [1, 13, 24]}
    smoke, smoke_key, smoke_contract = feature_paths(cfg, draft, **options)
    assert feature_selection("smoke") == {"train": 64, "eval": 16}
    assert feature_counts(smoke_contract) == {"train": 64, "eval": 16}
    assert feature_counts(full_contract) == {"train": 40000, "eval": 3000}
    assert "feature_scope" not in full_contract  # Existing full identity semantics.
    assert smoke_key != full_key
    assert smoke["data"]["hidden_states_path"] == f"/vol/hidden_states/m3-04/smoke/{smoke_key}/train"
    assert smoke["data"]["eval_hidden_states_path"] == f"/vol/hidden_states/m3-04/smoke/{smoke_key}/eval"
    assert full["data"]["hidden_states_path"] == f"/vol/hidden_states/m3-04/train/{full_key}"
    assert full["data"]["eval_hidden_states_path"] == f"/vol/hidden_states/m3-04/eval/{full_key}"
    assert feature_paths(cfg, draft, **options)[1:] == (smoke_key, smoke_contract)
    assert feature_paths(cfg, draft, train_samples=128, **options)[1] != smoke_key
    assert feature_paths(cfg, draft, eval_samples=32, **options)[1] != smoke_key
    assert smoke_contract["raw"]["train"]["rows"] == 40000
    assert smoke_contract["raw"]["val"]["rows"] == 3000
    assert smoke_contract["target_revision"] == "immutable-target"
    assert smoke_contract["capture_contract"]["layers"] == [1, 13, 24]
    for key in ("specforge_sha", "draft_sha256", "adapter_sha256", "loader_bridge_sha256", "max_length", "chat_template"):
        assert smoke_contract[key] == full_contract[key]


@pytest.mark.parametrize("scope,train,eval_count", [
    ("full", 64, 16), ("smoke", 0, 16), ("smoke", 64, -1),
    ("smoke", 7, 16), ("smoke", 40008, 16), ("smoke", 64, 3001),
])
def test_invalid_feature_limits_fail_closed(scope, train, eval_count):
    with pytest.raises(ValueError):
        feature_selection(scope, train, eval_count)
