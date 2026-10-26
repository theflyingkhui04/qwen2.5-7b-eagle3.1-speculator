"""Production integration regressions; no Modal calls or accelerator runtime."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pytest
import torch

from training import native_vocab_mapping as native
from training.train import SPECFORGE_SHA, feature_paths, load_training_config, resolved_config, write_config
from tests.test_vocab_contract import upstream
from tests.test_offline_startup_audit import offline
from tests.test_modal import pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sealed_cache(tmp_path, monkeypatch, upstream, offline):
    """A small independently serialized cache/feature pair and trusted profile."""
    monkeypatch.setattr(native, "VOLUME_ROOT", tmp_path / "vol")
    cfg = load_training_config()
    cfg["model"]["target_model_path"] = "target/pinned-revision"
    cfg["output_dir"] = str(tmp_path / "checkpoints")
    cfg, _, contract = feature_paths(cfg, ROOT / "configs/eagle3.1_qwen2.5_7b.json")
    contract["raw"]["train"]["rows"] = 12
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20]
    train = native.VOLUME_ROOT / "hidden_states/m3-04/train" / key
    cache = native.VOLUME_ROOT / "datasets/m3-04/processed" / key / "cache"
    cfg["data"].update(hidden_states_path=str(train),
                       eval_hidden_states_path=str(native.VOLUME_ROOT / "hidden_states/m3-04/eval" / key),
                       cache_dir=str(cache))
    rows = [{"input_ids": [[0]], "loss_mask": [[0]]} for _ in range(12)]
    rows[2] = {"input_ids": [[7]], "loss_mask": [[1]]}
    rows[10] = {"input_ids": [[2, 9]], "loss_mask": [[1, 2]]}
    rows[11] = {"input_ids": [[11, 12, 13, 14]], "loss_mask": [[0, 0, 0, 0]]}
    group = train / "rows_0-2000"
    group.mkdir(parents=True)
    for start in range(0, 12, 2):
        width = max(len(rows[i]["input_ids"][0]) for i in (start, start + 1))
        for i in (start, start + 1):
            raw = {field: torch.tensor(rows[i][field][0] + [0] * (width - len(rows[i][field][0])))
                   for field in ("input_ids", "loss_mask")}
            torch.save(raw, group / f"data_{i}.ckpt")
    manifest = {"contract": contract, "target_revision": "pinned-revision", "capture_layers": [1, 13, 24],
                "files": [{"path": str(p.relative_to(train)).replace("\\", "/"),
                           "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in sorted(group.glob("*.ckpt"))]}
    manifest_path = train / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    recipe = f"{cache.parent / 'train.jsonl'}-2048-qwen-{contract['target']}-None-True"
    cache_key = hashlib.md5(recipe.replace("\\", "/").encode()).hexdigest()
    directory = cache / "processed_dataset"
    directory.mkdir(parents=True)
    shards = []
    for i in range(2):
        table = pa.Table.from_pylist(rows[i*6:(i+1)*6])
        path = directory / f"{cache_key}_{i:05d}_of_00002.pkl"
        with pa.OSFile(str(path), "wb") as stream:
            with pa.ipc.new_stream(stream, table.schema) as writer:
                writer.write_table(table)
        shards.append({"name": path.name, "rows": 6, "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    t2d = torch.zeros(16, dtype=torch.bool)
    t2d[[2, 9]] = True
    profile = {"artifact_key": key, "specforge_sha": SPECFORGE_SHA, "target_revision": "pinned-revision",
               "train_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "cache_identity": cache_key, "train_samples": 12, "max_length": 2048,
               "target_vocab_size": 16, "draft_vocab_size": 2, "cache_shards": shards,
               "tensor_sha256": native.tensor_hashes({"t2d": t2d, "d2t": torch.tensor([2, 8])})}
    profile_path = tmp_path / "proof.json"
    profile_path.write_text(json.dumps(profile))
    monkeypatch.setattr(native, "PROFILE_PATH", profile_path)
    upstream["module"]("specforge.runtime.data_plane.offline_reader", list_feature_files=offline["list_feature_files"])
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    payload = {"config": cfg, "contract": contract, "artifact_key": key,
               "target_revision": "pinned-revision", "capture_layers": [1, 13, 24],
               "resolved_draft": {"vocab_size": 16, "draft_vocab_size": 2}, "evidence": str(evidence)}
    return payload, profile


def forbid_feature_contents(monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.suffix == ".ckpt":
            pytest.fail("Production vocabulary derivation must not read feature contents")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)


def test_derive_persist_reuse_bind_and_resume_without_feature_reads(sealed_cache, monkeypatch, tmp_path):
    payload, profile = sealed_cache
    forbid_feature_contents(monkeypatch)
    native.prepare_native_vocab_mapping(payload)
    path = Path(payload["config"]["model"]["vocab_mapping_path"])
    assert path == Path(payload["config"]["data"]["cache_dir"]) / "vocab_mapping/native_canonical.pt"
    native.validate_native_vocab_mapping(payload)
    assert native.tensor_hashes(torch.load(path, weights_only=True)) == profile["tensor_sha256"]
    original = path.read_bytes()
    monkeypatch.setattr(native, "count_cache_tokens", lambda *a, **kw: pytest.fail("A sealed hit must not recount"))
    native.prepare_native_vocab_mapping(payload)
    assert path.read_bytes() == original
    cfg = resolved_config(payload["config"], resume=True)
    write_config(cfg, tmp_path / "execution.yaml")
    assert load_training_config(tmp_path / "execution.yaml")["model"]["vocab_mapping_path"] == str(path)
    assert "test.jsonl" not in path.with_suffix(".json").read_text()


@pytest.mark.parametrize("mutation", ["key", "revision", "vocab", "max_length", "target", "cache_path", "train_path", "eval_path", "contract", "prepared_A"])
def test_wrong_identity_fails_closed(sealed_cache, mutation):
    payload, _ = sealed_cache
    cfg = payload["config"]
    if mutation == "key": payload["artifact_key"] = "wrong"
    elif mutation == "revision": payload["target_revision"] = "wrong"
    elif mutation == "vocab": payload["resolved_draft"]["draft_vocab_size"] = 3
    elif mutation == "max_length": cfg["data"]["max_length"] = 10
    elif mutation == "target": cfg["model"]["target_model_path"] = "wrong"
    elif mutation == "contract": payload["contract"]["raw"]["test"]["sha256"] = "wrong"
    elif mutation == "prepared_A": cfg["model"]["vocab_mapping_path"] = str(Path(cfg["data"]["hidden_states_path"]) / "vocab_mapping/vocab_mapping.pt")
    else:
        field = {"cache_path": "cache_dir", "train_path": "hidden_states_path", "eval_path": "eval_hidden_states_path"}[mutation]
        cfg["data"][field] = cfg["data"][field].replace("/train/", "/test/") + "-wrong"
    with pytest.raises(ValueError):
        native.prepare_native_vocab_mapping(payload)


@pytest.mark.parametrize("mutation", ["missing", "bytes", "metadata", "shape", "dtype", "tensor", "unset_config"])
def test_persisted_mapping_cannot_be_silently_replaced(sealed_cache, mutation):
    payload, _ = sealed_cache
    native.prepare_native_vocab_mapping(payload)
    path = Path(payload["config"]["model"]["vocab_mapping_path"])
    if mutation == "missing": path.unlink()
    elif mutation == "bytes": path.write_bytes(b"broken")
    elif mutation == "metadata": path.with_suffix(".json").write_text("{}")
    elif mutation == "unset_config": payload["config"]["model"].pop("vocab_mapping_path")
    else:
        state = torch.load(path, weights_only=True)
        if mutation == "shape": state["t2d"] = state["t2d"][:3]
        elif mutation == "dtype": state["t2d"] = state["t2d"].long()
        else: state["d2t"][0] += 1
        torch.save(state, path)
    with pytest.raises(Exception):
        native.validate_native_vocab_mapping(payload)
    if mutation != "unset_config":
        with pytest.raises(Exception):
            native.prepare_native_vocab_mapping(payload)


def test_cache_changes_are_rejected_even_on_mapping_hit(sealed_cache):
    payload, profile = sealed_cache
    native.prepare_native_vocab_mapping(payload)
    shard = Path(payload["config"]["data"]["cache_dir"]) / "processed_dataset" / profile["cache_shards"][0]["name"]
    shard.write_bytes(shard.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="content changed"):
        native.prepare_native_vocab_mapping(payload)


def test_smoke_cannot_import_full_mapping_or_test_namespace(sealed_cache):
    payload, _ = sealed_cache
    native.prepare_native_vocab_mapping(payload)
    payload["contract"]["feature_scope"] = "smoke"
    with pytest.raises(ValueError, match="Smoke"):
        native.validate_native_vocab_mapping(payload)
    payload["config"]["model"].pop("vocab_mapping_path")
    with pytest.raises(ValueError):
        native.prepare_native_vocab_mapping(payload)


def test_execution_gate_consumes_explicit_mapping_before_upstream_scan(sealed_cache, pipeline, upstream, monkeypatch):
    payload, _ = sealed_cache
    native.prepare_native_vocab_mapping(payload)
    module, modal, _ = pipeline
    monkeypatch.setattr(module, "verify_runtime", lambda **kwargs: None)
    monkeypatch.setattr(modal, "current_function_call_id", lambda: "cpu-test", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU mock")
    commands = []
    def consume(path):
        cfg = load_training_config(path)
        mapping = cfg["model"]["vocab_mapping_path"]
        assert mapping == payload["config"]["model"]["vocab_mapping_path"]
        # Pinned model provider loads this mapping before the early-return guard.
        assert native.tensor_hashes(torch.load(mapping, weights_only=True))
        def forbidden(*args, **kwargs): pytest.fail("Explicit mapping must skip the native feature scan")
        upstream["module"]("specforge.runtime.data_plane.offline_reader", list_feature_files=forbidden)
        upstream["module"]("specforge.training.vocab_mapping", count_effective_feature_tokens=forbidden)
        from types import SimpleNamespace as NS
        upstream["_ensure_offline_vocab_mapping"](
            NS(model=NS(vocab_mapping_path=mapping)), NS(draft_vocab_size=2,target_vocab_size=16),
            NS(providers=NS(vocab_mapping_modes={"offline"})))
        commands.append(path)
    monkeypatch.setattr("training.train.run_specforge_training", consume)
    module.train_specforge(payload)  # Fake Modal decorator and trainer; CPU only.
    assert len(commands) == 1 and Path(commands[0]).name == "execution.yaml"


def test_accepted_full_counter_produces_exact_native_tensors_and_persists(upstream, tmp_path):
    fixture = json.loads((ROOT / "tests/fixtures/native_train_counts.json").read_text())
    profile = json.loads((ROOT / "configs/native_vocab_mapping_proof.json").read_text())
    assert fixture["artifact_key"] == profile["artifact_key"]
    counts = Counter(dict(fixture["ordered_counts"]))
    assert len(counts) == 80790 and sum(counts.values()) == 16749122
    selected = torch.tensor(fixture["native_selected_target_ids"], dtype=torch.int64)
    expected = {"t2d": torch.zeros(152064, dtype=torch.bool), "d2t": selected - torch.arange(16000)}
    expected["t2d"][selected] = True
    mapping = native.select_mapping(counts, profile)
    assert all(torch.equal(mapping[key], expected[key]) for key in expected)
    output = tmp_path / "native_canonical.pt"
    native.persist_mapping(mapping, output, output.with_suffix(".json"), {"proof": profile}, profile)
    loaded = torch.load(output, weights_only=True)
    assert all(torch.equal(loaded[key], expected[key]) for key in expected)
    assert native.tensor_hashes(loaded) == profile["tensor_sha256"]


def test_real_canonical_execution_config_preserves_explicit_path_and_frozen_fields(tmp_path):
    cfg = load_training_config()
    revision = "a09a35458c702b33eeacc393d103063234e8bc28"
    cfg["model"]["target_model_path"] = "/__modal/volumes/vo-OV83DsWyBYwZAYq9SQ3y2E/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/" + revision
    cfg, key, _ = feature_paths(cfg, ROOT / "configs/eagle3.1_qwen2.5_7b.json")
    assert key == "a113d35ad76e959e31dd"
    frozen = copy.deepcopy(cfg)
    mapping = str(PurePosixPath(cfg["data"]["cache_dir"]) / "vocab_mapping/native_canonical.pt")
    cfg["model"]["vocab_mapping_path"] = mapping
    execution = resolved_config(cfg)
    write_config(execution, tmp_path / "execution.yaml")
    loaded = load_training_config(tmp_path / "execution.yaml")
    assert loaded["model"].pop("vocab_mapping_path") == "/vol/datasets/m3-04/processed/a113d35ad76e959e31dd/cache/vocab_mapping/native_canonical.pt"
    assert loaded == frozen


def test_cpu_preflight_seals_mapping_before_writing_resolved_config(sealed_cache, pipeline, upstream, monkeypatch, tmp_path):
    """Execute the real orchestration seam with local CPU inputs and fake I/O."""
    from types import SimpleNamespace as NS
    import data.prepare_specforge as adapter
    import huggingface_hub
    import transformers
    payload, _ = sealed_cache
    module, modal, _ = pipeline
    cfg = copy.deepcopy(payload["config"])
    draft = {**payload["resolved_draft"], "fc_norm": True, "norm_output": True, "num_hidden_layers": 1}
    monkeypatch.setattr(module, "verify_runtime", lambda **kwargs: {})
    monkeypatch.setattr(module, "load_training_config", lambda path: cfg)
    monkeypatch.setattr(module, "feature_paths", lambda *a, **kw: (cfg, payload["artifact_key"], payload["contract"]))
    monkeypatch.setattr(module, "check_features", lambda value: None)  # Its independent guards have existing tests.
    monkeypatch.setattr(modal, "current_function_call_id", lambda: "cpu-preflight", raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: cfg["model"]["target_model_path"])
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: NS(_commit_hash=payload["target_revision"]))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(adapter, "validate_raw", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "audit_prepared", lambda *a, **kw: {})
    upstream["module"]("specforge.config", Config=NS(model_validate=lambda value: value))
    upstream["module"]("specforge.application", resolve_offline_capture=lambda *a, **kw:
                       NS(capture_layers=payload["capture_layers"], draft_config=NS(to_dict=lambda: draft)))
    freeze = tmp_path / "runtime.freeze.txt"
    freeze.write_text("CPU test")
    def local_path(value):
        return {"/root/configs/m3_04_dataset.json": ROOT / "configs/m3_04_dataset.json",
                "/opt/specforge-runtime.freeze.txt": freeze,
                "/vol/runs/m3-04": tmp_path / "runs"}.get(str(value), Path(value))
    monkeypatch.setattr(module, "Path", local_path)
    def plan(command, **kwargs):
        resolved = load_training_config(command[command.index("-c") + 1])
        assert Path(resolved["model"]["vocab_mapping_path"]).is_file()
        assert resolved["model"]["vocab_mapping_path"] == cfg["model"]["vocab_mapping_path"]
        return NS(stdout="CPU plan", check_returncode=lambda: None)
    monkeypatch.setattr(module.subprocess, "run", plan)
    result = module.preflight(mode="features")
    assert result["config_sha256"] == hashlib.sha256(Path(result["config_path"]).read_bytes()).hexdigest()
    native.validate_native_vocab_mapping(result)
    assert result["status"] == "CPU_PLAN_PASS"
