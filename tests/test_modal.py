import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from training.train import sha256_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pipeline(monkeypatch):
    image = MagicMock()
    for name in ("entrypoint", "apt_install", "env", "run_commands", "pip_install", "add_local_python_source", "add_local_file", "add_local_dir"):
        getattr(image, name).return_value = image
    volume = MagicMock()
    secret = MagicMock()
    class App:
        def __init__(self, *args, **kwargs):
            pass
        def function(self, **options):
            def decorate(fn):
                fn.options = options
                fn.remote = MagicMock()
                return fn
            return decorate
        def local_entrypoint(self):
            return lambda fn: fn
    modal = SimpleNamespace(
        Image=SimpleNamespace(from_registry=MagicMock(return_value=image)),
        Volume=SimpleNamespace(from_name=MagicMock(return_value=volume)),
        Secret=SimpleNamespace(from_name=secret),
        App=App,
    )
    monkeypatch.setitem(sys.modules, "modal", modal)
    spec = importlib.util.spec_from_file_location("pipeline_under_test", ROOT / "modal_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.execution_source = lambda: {"head": "test"}
    return module, modal, image


def test_image_pin_and_no_raw_dataset_upload(pipeline):
    module, modal, image = pipeline
    commands = image.run_commands.call_args_list[0].args
    assert any(module.SPECFORGE_SHA in cmd for cmd in commands)
    image.pip_install.assert_called_once_with("/opt/SpecForge")
    modal.Volume.from_name.assert_called_once_with("qwen-eagle3-volume", create_if_missing=False)
    modal.Secret.from_name.assert_called_once_with("wandb-secret", required_keys=["WANDB_API_KEY"])
    assert not any("raw" in str(call) for call in image.add_local_file.call_args_list)
    image.add_local_python_source.assert_called_once_with("training")
    assert image.add_local_dir.call_args.args == ("configs",)


def test_cuda_development_image_and_environment(pipeline):
    _, modal, image = pipeline
    modal.Image.from_registry.assert_called_once_with(
        "nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04", add_python="3.11",
    )
    image.entrypoint.assert_called_once_with([])
    image.apt_install.assert_called_once_with("git", "build-essential", "ninja-build")
    environment = image.env.call_args.args[0]
    assert environment["CUDA_HOME"] == "/usr/local/cuda"
    assert environment["PATH"].split(":")[0] == "/usr/local/cuda/bin"
    assert "/usr/local/cuda/nvvm/bin" in environment["PATH"].split(":")


def test_plan_has_no_gpu_and_training_is_remote(pipeline):
    module, _, _ = pipeline
    assert "gpu" not in module.preflight.options
    assert module.train_specforge.options["gpu"] == "A100-80GB"
    assert module.prepare_hidden_states.options["gpu"] == "A100-80GB"
    assert module.preflight.options["volumes"].keys() == {"/vol"}
    module.preflight.remote.return_value = {"status": "CPU_PLAN_PASS", "evidence": "/vol/runs/test", "capture_layers": [1, 13, 24]}
    module.main()
    module.preflight.remote.assert_called_once_with(
        mode="plan", execution={"head": "test"}, feature_scope="full",
        train_samples=None, eval_samples=None,
    )
    module.train_specforge.remote.assert_not_called()
    module.prepare_hidden_states.remote.assert_not_called()


@pytest.mark.parametrize("step", ["smoke", "train"])
def test_gpu_training_requires_cpu_feature_preflight(pipeline, step):
    module, _, _ = pipeline
    payload = {"evidence": "test"}
    module.preflight.remote.return_value = payload
    module.main(step=step)
    module.preflight.remote.assert_called_once_with(
        mode="features", execution={"head": "test"},
        feature_scope="smoke" if step == "smoke" else "full",
        train_samples=None, eval_samples=None,
    )
    module.train_specforge.remote.assert_called_once_with(payload, smoke=step == "smoke", resume=False)


def test_invalid_step_cannot_fall_through_to_training(pipeline):
    module, _, _ = pipeline
    with pytest.raises(ValueError):
        module.main(step="all")
    module.train_specforge.remote.assert_not_called()


def test_feature_manifest_detects_corruption(pipeline, tmp_path):
    module, _, _ = pipeline
    paths = {}
    contract = {"raw": {"train": {"rows": 1}, "val": {"rows": 1}}}
    for split, field in (("train", "hidden_states_path"), ("val", "eval_hidden_states_path")):
        folder = tmp_path / split
        folder.mkdir()
        (folder / "0.ckpt").write_bytes(b"feature")
        paths[field] = str(folder)
        (folder / "artifact_manifest.json").write_text(json.dumps({
            "contract": contract, "capture_layers": [1, 13, 24],
            "files": module.feature_inventory(folder),
        }))
    payload = {"contract": contract, "capture_layers": [1, 13, 24], "config": {"data": paths}}
    module.check_features(payload)
    (tmp_path / "train/0.ckpt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        module.check_features(payload)



def test_bounded_preparation_cli_dispatch_and_no_test_capture(pipeline):
    module, _, _ = pipeline
    module.main(step="prepare_hidden_states", feature_scope="smoke", train_samples=64, eval_samples=16)
    module.preflight.remote.assert_called_once_with(
        mode="prepare", execution={"head": "test"}, feature_scope="smoke",
        train_samples=64, eval_samples=16,
    )
    module.prepare_hidden_states.remote.assert_called_once_with(module.preflight.remote.return_value)
    module.train_specforge.remote.assert_not_called()


@pytest.mark.parametrize("step,scope", [("smoke", "full"), ("train", "smoke")])
def test_training_cannot_use_wrong_feature_scope(pipeline, step, scope):
    module, _, _ = pipeline
    with pytest.raises(ValueError):
        module.main(step=step, feature_scope=scope)
    module.preflight.remote.assert_not_called()


@pytest.mark.parametrize("smoke", [True, False])
def test_extraction_limits_and_manifest_counts(pipeline, tmp_path, monkeypatch, smoke):
    module, _, _ = pipeline
    counts = {"train": 64, "eval": 16} if smoke else {"train": 40000, "eval": 3000}
    contract = {"raw": {"train": {"rows": 40000}, "val": {"rows": 3000}, "test": {"rows": 2000}}}
    if smoke:
        contract.update(feature_scope="smoke", train_samples=64, eval_samples=16)
    cfg = {"data": {"hidden_states_path": str(tmp_path / "train"),
                    "eval_hidden_states_path": str(tmp_path / "eval")}}
    payload = {"contract": contract, "config": cfg, "artifact_key": "test",
               "config_path": "resolved.yaml", "capture_layers": [1, 13, 24], "target_revision": "pinned"}
    monkeypatch.setattr(module, "verify_runtime", lambda **kwargs: None)
    commands = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(module, "feature_inventory", lambda path: [{"path": "mock-feature"}] * counts[path.name])
    module.prepare_hidden_states(payload)  # Fake Modal decorator; subprocess and runtime fully mocked.
    assert len(commands) == 2
    for label, command in zip(("train", "eval"), commands):
        assert "test.jsonl" not in " ".join(command)
        if smoke:
            assert command[command.index("--num-samples") + 1] == str(counts[label])
        else:
            assert "--num-samples" not in command
        result = json.loads((tmp_path / label / "artifact_manifest.json").read_text())
        assert len(result["files"]) == counts[label]
        if smoke:
            assert result["raw_counts"] == {"train": 40000, "val": 3000}
            assert result["selected_counts"] == counts
            assert result["produced_counts"] == {label: counts[label]}
        else:
            assert set(result) == {"contract", "capture_layers", "files", "target_revision"}


@pytest.mark.parametrize("smoke", [True, False])
def test_checker_enforces_scope_specific_exact_counts(pipeline, tmp_path, monkeypatch, smoke):
    module, _, _ = pipeline
    module.SMOKE_FEATURE_ROOT = tmp_path / "smoke"
    counts = {"train": 64, "eval": 16} if smoke else {"train": 40000, "eval": 3000}
    contract = {"raw": {"train": {"rows": 40000}, "val": {"rows": 3000}}}
    if smoke:
        contract.update(feature_scope="smoke", train_samples=64, eval_samples=16)
    root = tmp_path / ("smoke" if smoke else "full")
    config = {"data": {}}
    for label, field in (("train", "hidden_states_path"), ("eval", "eval_hidden_states_path")):
        directory = root / label
        directory.mkdir(parents=True)
        config["data"][field] = str(directory)
        manifest = {"contract": contract, "capture_layers": [1, 13, 24],
                    "files": [{"path": "mock-feature"}] * counts[label]}
        if smoke:
            manifest.update(raw_counts={"train": 40000, "val": 3000},
                            selected_counts=counts, produced_counts={label: counts[label]})
        (directory / "artifact_manifest.json").write_text(json.dumps(manifest))
    payload = {"contract": contract, "config": config, "capture_layers": [1, 13, 24]}
    monkeypatch.setattr(module, "feature_inventory",
                        lambda directory: json.loads((directory / "artifact_manifest.json").read_text())["files"])
    module.check_features(payload)
    path = root / "train/artifact_manifest.json"
    altered = json.loads(path.read_text())
    altered["files"] = altered["files"][:-1]
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="count mismatch"):
        module.check_features(payload)
    # Paths from the other scope cannot satisfy this contract.
    config["data"]["hidden_states_path"] = str(tmp_path / ("full" if smoke else "smoke") / "train")
    with pytest.raises(ValueError, match="namespace"):
        module.check_features(payload)


def test_extraction_rejects_incomplete_smoke_features(pipeline, tmp_path, monkeypatch):
    module, _, _ = pipeline
    payload = {
        "contract": {"feature_scope": "smoke", "train_samples": 64, "eval_samples": 16},
        "config": {"data": {"hidden_states_path": str(tmp_path / "train"),
                            "eval_hidden_states_path": str(tmp_path / "eval")}},
        "config_path": "resolved.yaml", "artifact_key": "test",
    }
    monkeypatch.setattr(module, "verify_runtime", lambda **kwargs: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "feature_inventory", lambda path: [])
    with pytest.raises(ValueError, match="selected count"):
        module.prepare_hidden_states(payload)
    assert not (tmp_path / "train/artifact_manifest.json").exists()
