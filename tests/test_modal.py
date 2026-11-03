import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from training.train import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_optimizer_step_ab_is_explicit_and_does_not_dispatch_train(pipeline):
    module, _, _ = pipeline
    module.optimizer_step_ab.remote.return_value = {"classification": "BENCHMARK INVALID"}
    module.main(step="optimizer_step_ab")
    module.optimizer_step_ab.remote.assert_called_once_with({"head": "test"})
    module.train_specforge.remote.assert_not_called()
    module.smoke_specforge.remote.assert_not_called()
    module.preflight.remote.assert_not_called()
    assert "secrets" not in module.optimizer_step_ab.options


@pytest.mark.parametrize("options", [
    {"feature_scope": "smoke"}, {"resume": True}, {"resume_from": "step3000"},
    {"compile_mode_ab": True}, {"train_samples": 8},
])
def test_optimizer_step_ab_rejects_canonical_resume_selection_and_other_compile_modes(pipeline, options):
    module, _, _ = pipeline
    with pytest.raises(ValueError):
        module.main(step="optimizer_step_ab", **options)
    module.optimizer_step_ab.remote.assert_not_called()


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
                fn.local = fn
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
    if step == "smoke":
        module.smoke_specforge.remote.assert_called_once_with(payload, resume=False)
        module.train_specforge.remote.assert_not_called()
    else:
        module.train_specforge.remote.assert_called_once_with(payload, smoke=False, resume=False)


def test_canonical_timeout_and_unchanged_smoke_resources(pipeline):
    module, _, _ = pipeline
    assert module.train_specforge.options["timeout"] == 79200
    assert module.smoke_specforge.options["timeout"] == 21600
    assert module.prepare_hidden_states.options["timeout"] == 21600
    assert {k: v for k, v in module.train_specforge.options.items() if k != "timeout"} == {
        k: v for k, v in module.smoke_specforge.options.items() if k != "timeout"}


def test_explicit_resume_dispatch_is_cpu_gated(pipeline):
    module, _, _ = pipeline
    path = "/vol/checkpoints/qwen2.5-7b-eagle3.1/m3-04-qwen2.5-7b-eagle3.1-step3000"
    module.main(step="train", resume=True, resume_from=path)
    assert module.preflight.remote.call_args.kwargs["resume_from"] == path
    assert module.preflight.remote.call_args.kwargs["resume"] is True
    module.train_specforge.remote.assert_called_once_with(
        module.preflight.remote.return_value, smoke=False, resume=True)


def test_failed_resume_preflight_cannot_dispatch_gpu(pipeline):
    module, _, _ = pipeline
    module.preflight.remote.side_effect = ValueError("Resume checkpoint identity mismatch")
    with pytest.raises(ValueError, match="identity"):
        module.main(step="train", resume=True)
    module.train_specforge.remote.assert_not_called()
    module.smoke_specforge.remote.assert_not_called()


def test_smoke_wrapper_calls_same_training_body_locally(pipeline, monkeypatch):
    module, _, _ = pipeline
    body = MagicMock()
    monkeypatch.setattr(module.train_specforge, "local", body)
    payload = {"contract": {"feature_scope": "smoke"}}
    module.smoke_specforge(payload)
    body.assert_called_once_with(payload, smoke=True, resume=False)
    module.train_specforge.remote.assert_not_called()


def test_resume_execution_yaml_retains_concrete_anchor(pipeline, tmp_path, monkeypatch):
    import torch
    from training.train import load_training_config, resolved_config
    module, modal, _ = pipeline
    cfg = load_training_config()
    cfg["output_dir"] = str(tmp_path / "checkpoints")
    path = str(Path(cfg["output_dir"]) / (cfg["run_id"] + "-step3000"))
    cfg = resolved_config(cfg, resume=True, resume_from=path)
    payload = {"config": cfg, "contract": {}, "evidence": str(tmp_path)}
    monkeypatch.setattr(module, "verify_runtime", lambda **kw: None)
    monkeypatch.setattr(module, "validate_native_vocab_mapping", lambda value: None)
    guard = MagicMock()
    monkeypatch.setattr(module, "validate_resume_checkpoint", guard)
    monkeypatch.setattr(modal, "current_function_call_id", lambda: "CPU mock", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU mock")
    launch = MagicMock()
    monkeypatch.setattr("training.train.run_specforge_training", launch)
    module.train_specforge(payload, resume=True)
    guard.assert_called_once()
    execution = load_training_config(launch.call_args.args[0])
    assert execution["training"]["resume_from"] == path
    expected = resolved_config(cfg, resume=True, resume_from=path)
    expected["tracking"]["wandb_dir"] = str(tmp_path / "wandb")
    assert execution == expected
    assert "draft_checkpoint_path" not in execution["model"]


@pytest.mark.parametrize("step,resume,path", [("smoke", True, ""), ("plan", True, ""),
                                               ("train", False, "/wrong")])
def test_invalid_resume_dispatch_never_calls_remote(pipeline, step, resume, path):
    module, _, _ = pipeline
    with pytest.raises(ValueError, match="Resume"):
        module.main(step=step, resume=resume, resume_from=path)
    module.preflight.remote.assert_not_called()
    module.train_specforge.remote.assert_not_called()


def test_invalid_step_cannot_fall_through_to_training(pipeline):
    module, _, _ = pipeline
    with pytest.raises(ValueError):
        module.main(step="all")
    module.train_specforge.remote.assert_not_called()


def test_flex_diagnostic_is_explicit_and_has_no_training_dispatch(pipeline):
    module, _, _ = pipeline
    module.flex_attention_ab.remote.return_value = {"status": "EXECUTED"}
    module.main(step="flex_attention_ab")
    module.flex_attention_ab.remote.assert_called_once_with({"head": "test"})
    module.preflight.remote.assert_not_called()
    module.train_specforge.remote.assert_not_called()
    module.smoke_specforge.remote.assert_not_called()
    assert module.flex_attention_ab.options == {
        "gpu": "A100-80GB", "volumes": module.train_specforge.options["volumes"], "timeout": 1800}


@pytest.mark.parametrize("kwargs", [{"resume": True}, {"resume_from": "/step3000"},
                                    {"feature_scope": "smoke"}, {"train_samples": 2}])
def test_flex_diagnostic_rejects_protocol_overrides(pipeline, kwargs):
    module, _, _ = pipeline
    with pytest.raises(ValueError):
        module.main(step="flex_attention_ab", **kwargs)
    module.flex_attention_ab.remote.assert_not_called()


def test_compile_mode_ab_is_explicit_diagnostic_only(pipeline):
    module, _, _ = pipeline
    module.flex_attention_ab.remote.return_value = {"status": "EXECUTED"}
    module.main(step="flex_attention_ab", compile_mode_ab=True)
    module.flex_attention_ab.remote.assert_called_once_with({"head": "test"}, compile_mode_ab=True)
    for step in ("plan", "smoke", "train"):
        with pytest.raises(ValueError, match="explicit flex_attention_ab"):
            module.main(step=step, compile_mode_ab=True)
    module.train_specforge.remote.assert_not_called()


def test_diagnostic_persists_failure_in_exclusive_namespace(pipeline, monkeypatch, tmp_path):
    module, modal, _ = pipeline
    import training.flex_attention_ab as diagnostic
    # A CPU failure-path test must not query/initialize the host's real CUDA.
    monkeypatch.setattr(diagnostic, "record_diagnostic_hardware", lambda report: None)
    monkeypatch.setattr(module, "Path", lambda value: tmp_path if str(value).startswith("/vol/") else Path(value))
    monkeypatch.setattr(module, "verify_runtime", lambda **kw: {})
    monkeypatch.setattr(modal, "current_function_call_id", lambda: "CPU mock", raising=False)
    def fail(report, checker):
        raise RuntimeError("compiler failed")
    monkeypatch.setattr(diagnostic, "run_diagnostic", fail)
    with pytest.raises(RuntimeError, match="compiler failed"):
        module.flex_attention_ab({"head": "test"})
    artifacts = list(tmp_path.glob("*/diagnostic.json"))
    assert len(artifacts) == 1
    report = json.loads(artifacts[0].read_text())
    assert report["semantic_equivalence"] == "UNDECIDED"
    assert report["exception"]["type"] == "RuntimeError"
    module.volume.commit.assert_called_once()


@pytest.mark.parametrize("phase", ["software_verification", "diagnostic", "success"])
def test_diagnostic_persists_hardware_identity_even_on_early_failure(pipeline, monkeypatch, tmp_path, phase):
    module, modal, _ = pipeline
    import training.flex_attention_ab as diagnostic
    events = []
    identity = {
        "device_name": "NVIDIA A100 80GB PCIe", "total_memory_bytes": 80 * 1024 ** 3,
        "compute_capability": [8, 0], "torch_cuda": "13.0", "device_count": 1,
        "hardware_policy": "noncanonical_a100_80gb_diagnostic",
        "performance_comparable_to_canonical_sxm4": False,
        "performance_comparability_scope": "hardware_identity_only",
    }
    def hardware(report):
        events.append("hardware")
        report.update(identity)
    def verify(**kwargs):
        events.append("software_verification")
        if phase == "software_verification":
            raise RuntimeError("software failed")
        return {}
    def replay(report, checker):
        events.append("diagnostic")
        if phase == "diagnostic":
            raise RuntimeError("diagnostic failed")
    monkeypatch.setattr(module, "Path", lambda value: tmp_path if str(value).startswith("/vol/") else Path(value))
    monkeypatch.setattr(modal, "current_function_call_id", lambda: "CPU mock", raising=False)
    monkeypatch.setattr(diagnostic, "record_diagnostic_hardware", hardware)
    monkeypatch.setattr(diagnostic, "run_diagnostic", replay)
    monkeypatch.setattr(module, "verify_runtime", verify)
    if phase == "success":
        module.flex_attention_ab({"head": "test"})
    else:
        with pytest.raises(RuntimeError, match="failed"):
            module.flex_attention_ab({"head": "test"})
    report = json.loads(next(tmp_path.glob("*/diagnostic.json")).read_text())
    assert {key: report[key] for key in identity} == identity
    assert events[0] == "hardware" and events[1] == "software_verification"
    assert report["status"] == ("EXECUTED" if phase == "success" else "ERROR")
    module.volume.commit.assert_called_once()


@pytest.fixture
def sealed_features(pipeline, tmp_path, monkeypatch):
    from training.train import feature_paths, load_training_config
    module, _, _ = pipeline
    module.FEATURE_ROOT = tmp_path / "features"
    module.SMOKE_FEATURE_ROOT = module.FEATURE_ROOT / "smoke"

    def make(smoke=False):
        cfg, key, contract = feature_paths(
            load_training_config(), ROOT / "configs/eagle3.1_qwen2.5_7b.json",
            feature_scope="smoke" if smoke else "full",
            target_revision="pinned", capture_layers=[1, 13, 24],
        )
        counts = module.feature_counts(contract)
        observed = {}
        for label, field in (("train", "hidden_states_path"), ("eval", "eval_hidden_states_path")):
            directory = (module.SMOKE_FEATURE_ROOT / key / label if smoke
                         else module.FEATURE_ROOT / label / key)
            directory.mkdir(parents=True)
            cfg["data"][field] = str(directory)
            files = [{"path": f"{i:05d}.ckpt", "bytes": 7, "sha256": "a" * 64}
                     for i in range(counts[label])]
            observed[directory] = [{"path": f["path"], "bytes": f["bytes"]} for f in files]
            manifest = {"contract": contract, "capture_layers": [1, 13, 24],
                        "target_revision": "pinned", "files": files}
            if smoke:
                manifest.update(raw_counts={"train": 40000, "val": 3000},
                                selected_counts=counts, produced_counts={label: counts[label]})
            (directory / "artifact_manifest.json").write_text(json.dumps(manifest))
        monkeypatch.setattr(module, "feature_metadata", lambda directory: observed[directory])
        payload = {"contract": contract, "config": cfg, "capture_layers": [1, 13, 24],
                   "artifact_key": key, "target_revision": "pinned"}
        return payload, observed
    return make


def test_full_feature_preflight_never_hashes_contents(pipeline, sealed_features, monkeypatch):
    module, _, _ = pipeline
    payload, observed = sealed_features()
    def forbidden(*args):
        pytest.fail("Training preflight must not rehash checkpoint contents")
    monkeypatch.setattr(module, "sha256_file", forbidden)
    monkeypatch.setattr(module, "feature_inventory", forbidden)
    module.check_features(payload)
    # Keep the file size check even though full hashes are no longer recomputed.
    observed[Path(payload["config"]["data"]["hidden_states_path"])][0]["bytes"] += 1
    with pytest.raises(ValueError, match="changed"):
        module.check_features(payload)


def test_feature_metadata_reads_no_contents(pipeline, tmp_path, monkeypatch):
    module, _, _ = pipeline
    (tmp_path / "0.ckpt").write_bytes(b"feature")
    def forbidden(*args, **kwargs):
        pytest.fail("Metadata validation must not open checkpoint contents")
    monkeypatch.setattr(Path, "open", forbidden)
    assert module.feature_metadata(tmp_path) == [{"path": "0.ckpt", "bytes": 7}]


@pytest.mark.parametrize("mutation", ["contract", "capture_layers", "target_revision", "hash"])
def test_feature_manifest_metadata_mismatch(pipeline, sealed_features, mutation):
    module, _, _ = pipeline
    payload, _ = sealed_features()
    path = Path(payload["config"]["data"]["hidden_states_path"]) / "artifact_manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "contract":
        manifest["contract"]["specforge_sha"] = "wrong"
    elif mutation == "hash":
        manifest["files"][0]["sha256"] = "invalid"
    else:
        manifest[mutation] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        module.check_features(payload)


@pytest.mark.parametrize("mutation", ["key", "path", "test_split", "symlink"])
def test_feature_identity_and_path_mismatch(pipeline, sealed_features, mutation):
    module, _, _ = pipeline
    payload, _ = sealed_features()
    if mutation == "key":
        payload["artifact_key"] = "wrong"
    elif mutation == "symlink":
        # Simulate a path resolving outside its selected artifact, without OS symlink privileges.
        original = Path.resolve
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path, "resolve", lambda self: original(self).parent / "other")
            with pytest.raises(ValueError, match="path mismatch"):
                module.check_features(payload)
        return
    else:
        field = "hidden_states_path"
        path = Path(payload["config"]["data"][field])
        payload["config"]["data"][field] = str(path.parent / "wrong" if mutation == "path"
                                               else module.FEATURE_ROOT / "test" / payload["artifact_key"])
    with pytest.raises(ValueError):
        module.check_features(payload)


@pytest.mark.parametrize("missing", ["manifest", "artifact"])
def test_missing_feature_artifact_fails(pipeline, sealed_features, missing):
    module, _, _ = pipeline
    payload, _ = sealed_features()
    directory = Path(payload["config"]["data"]["hidden_states_path"])
    (directory / "artifact_manifest.json").unlink()
    if missing == "artifact":
        directory.rmdir()
    with pytest.raises(FileNotFoundError):
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
@pytest.mark.parametrize("label,field", [("train", "hidden_states_path"), ("eval", "eval_hidden_states_path")])
def test_checker_enforces_scope_specific_exact_counts(pipeline, sealed_features, smoke, label, field):
    module, _, _ = pipeline
    payload, _ = sealed_features(smoke)
    module.check_features(payload)
    path = Path(payload["config"]["data"][field]) / "artifact_manifest.json"
    altered = json.loads(path.read_text())
    altered["files"] = altered["files"][:-1]
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="count mismatch"):
        module.check_features(payload)
    # Paths from the other scope cannot satisfy this contract.
    payload["config"]["data"]["hidden_states_path"] = str(
        module.FEATURE_ROOT / "full" if smoke else module.SMOKE_FEATURE_ROOT / "train")
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


def test_full_config_to_real_features_on_aliased_volume(pipeline, tmp_path, monkeypatch):
    """No current entries: resolve canonical placeholders to sealed keyed data.

    Modal exposes /vol through its physical /__modal/volumes/<id> mount.
    Only the mount location is relocated here; resolve and inventory are real.
    """
    from training.train import feature_paths, load_training_config

    module, _, _ = pipeline
    physical = tmp_path / "physical-volume"
    physical.mkdir()
    mount = tmp_path / "vol"

    def link(source, destination):
        if sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(str(source), str(destination))
        else:
            destination.symlink_to(source, target_is_directory=True)

    link(physical, mount)

    def mounted_path(value):
        text = str(value)
        return mount / text[len("/vol/"):] if text.startswith("/vol/") else Path(value)

    monkeypatch.setattr(module, "Path", mounted_path)
    module.FEATURE_ROOT = mount / "hidden_states/m3-04"
    module.SMOKE_FEATURE_ROOT = module.FEATURE_ROOT / "smoke"
    original = load_training_config()
    assert original["data"]["hidden_states_path"].endswith("/train/current")
    assert original["data"]["eval_hidden_states_path"].endswith("/eval/current")
    revision = "a09a35458c702b33eeacc393d103063234e8bc28"
    original["model"]["target_model_path"] = (
        "/__modal/volumes/vo-OV83DsWyBYwZAYq9SQ3y2E/hf-cache/hub/"
        "models--Qwen--Qwen2.5-7B-Instruct/snapshots/" + revision
    )
    cfg, key, contract = feature_paths(original, ROOT / "configs/eagle3.1_qwen2.5_7b.json")
    assert key == "a113d35ad76e959e31dd"  # Persisted canonical preparation identity.
    for label, field, count in (("train", "hidden_states_path", 40000),
                                ("eval", "eval_hidden_states_path", 3000)):
        assert cfg["data"][field] == f"/vol/hidden_states/m3-04/{label}/{key}"
        directory = mounted_path(cfg["data"][field])
        directory.mkdir(parents=True)
        assert not (directory.parent / "current").exists()
        files = []
        for i in range(count):
            name = f"{i:05d}.ckpt"
            (directory / name).write_bytes(b"feature")
            files.append({"path": name, "bytes": 7, "sha256": "a" * 64})
        (directory / "artifact_manifest.json").write_text(json.dumps({
            "contract": contract, "capture_layers": [1, 13, 24],
            "target_revision": revision, "files": files,
        }))
    payload = {"config": cfg, "contract": contract, "artifact_key": key,
               "capture_layers": [1, 13, 24], "target_revision": revision}
    module.check_features(payload)

    # Keep the real inventory and resolver for every rejection as well.
    field = "hidden_states_path"
    selected = cfg["data"][field]
    for invalid in (
        f"/vol/hidden_states/m3-04/train/wrong-key",
        f"/vol/elsewhere/train/{key}",
        f"/vol/hidden_states/m3-04/smoke/{key}/train",
        f"/vol/hidden_states/m3-04/test/{key}",
        "/vol/hidden_states/m3-04/train/current",
    ):
        cfg["data"][field] = invalid
        with pytest.raises(ValueError):
            module.check_features(payload)
    cfg["data"][field] = selected
    directory = mounted_path(selected)
    wrong = directory.with_name("wrong-key")
    directory.rename(wrong)
    link(wrong.resolve(), directory)
    with pytest.raises(ValueError, match="path mismatch"):
        module.check_features(payload)
