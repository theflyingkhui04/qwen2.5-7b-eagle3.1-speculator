"""Modal-only M3-04 runtime. Default entrypoint is the CPU plan gate."""
import hashlib
import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from stat import S_ISREG

import modal

from training.train import (
    BEST_METRIC, REPO_ROOT, SPECFORGE_SHA, feature_counts, feature_paths,
    feature_selection, load_training_config,
    resolved_config, sha256_file, verify_runtime, write_config,
)
from training.native_vocab_mapping import prepare_native_vocab_mapping, validate_native_vocab_mapping

volume = modal.Volume.from_name("qwen-eagle3-volume", create_if_missing=False)
app_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "build-essential", "ninja-build")
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HOME": "/vol/hf-cache",
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    })
    .run_commands(
        "git init /opt/SpecForge",
        f"git -C /opt/SpecForge fetch --depth=1 https://github.com/sgl-project/SpecForge.git {SPECFORGE_SHA}",
        f"git -C /opt/SpecForge checkout --detach {SPECFORGE_SHA}",
    )
    .pip_install("/opt/SpecForge")
    .run_commands("python -m pip freeze > /opt/specforge-runtime.freeze.txt")
    .add_local_python_source("training")
    # Explicit files: never upload data/raw or local processed datasets.
    .add_local_file("data/__init__.py", "/root/data/__init__.py")
    .add_local_file("data/prepare_specforge.py", "/root/data/prepare_specforge.py")
    .add_local_dir("configs", remote_path="/root/configs")
)
app = modal.App("qwen2-5-eagle3.1-speculator", image=app_image)
wandb_secret = modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])
FEATURE_ROOT = Path("/vol/hidden_states/m3-04")
SMOKE_FEATURE_ROOT = FEATURE_ROOT / "smoke"


def execution_source():
    root = str(REPO_ROOT)
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "-C", root, *args], text=True
        ).strip()
    files = ["modal_pipeline.py", "requirements.txt"] + [
        str(p.relative_to(REPO_ROOT)).replace("\\", "/")
        for folder in ("training", "configs")
        for p in (REPO_ROOT / folder).glob("*") if p.is_file()
    ] + ["data/prepare_specforge.py"]
    return {
        "head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
        "working_tree": git("status", "--short"),
        "source_sha256": {p: sha256_file(REPO_ROOT / p) for p in sorted(files)},
    }


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)


def feature_metadata(directory):
    """Check names and sizes without reading the large checkpoint contents."""
    files = sorted(Path(directory).rglob("*.ckpt"))
    metadata = []
    for path in files:
        info = path.lstat()
        if not S_ISREG(info.st_mode):
            raise ValueError("Feature must be a regular file")
        metadata.append({"path": str(path.relative_to(directory)), "bytes": info.st_size})
    return metadata


def feature_inventory(directory):
    # Full-content evidence is produced once, before the manifest is written.
    return [{**item, "sha256": sha256_file(Path(directory) / item["path"])}
            for item in feature_metadata(directory)]


def check_features(payload):
    counts = feature_counts(payload["contract"])
    smoke = payload["contract"].get("feature_scope") == "smoke"
    key = hashlib.sha256(json.dumps(payload["contract"], sort_keys=True).encode()).hexdigest()[:20]
    if payload["artifact_key"] != key:
        raise ValueError("Feature artifact identity mismatch")
    if not smoke and counts != feature_selection("full"):
        raise ValueError("Feature sample count mismatch")
    for split, field in (("train", "hidden_states_path"), ("val", "eval_hidden_states_path")):
        label = "eval" if split == "val" else split
        directory = Path(payload["config"]["data"][field])
        if smoke != directory.is_relative_to(SMOKE_FEATURE_ROOT):
            raise ValueError("Feature namespace does not match scope")
        expected = SMOKE_FEATURE_ROOT / key / label if smoke else FEATURE_ROOT / label / key
        # /vol may itself alias Modal's physical volume mount. Resolve that
        # shared root, but forbid artifact links escaping the selected key.
        resolved_expected = FEATURE_ROOT.resolve() / expected.relative_to(FEATURE_ROOT)
        if directory != expected or directory.resolve() != resolved_expected:
            raise ValueError("Feature artifact path mismatch")
        manifest = json.loads((directory / "artifact_manifest.json").read_text())
        if manifest["contract"] != payload["contract"] or manifest["capture_layers"] != payload["capture_layers"]:
            raise ValueError("Feature contract mismatch")
        if manifest["target_revision"] != payload["target_revision"]:
            raise ValueError("Feature target revision mismatch")
        if len(manifest["files"]) != counts[label]:
            raise ValueError("Feature sample count mismatch")
        metadata = []
        for item in manifest["files"]:
            path = Path(item["path"])
            if (path.is_absolute() or ".." in path.parts or path.suffix != ".ckpt"
                    or type(item["bytes"]) is not int or item["bytes"] <= 0
                    or not isinstance(item["sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
                raise ValueError("Invalid feature inventory metadata")
            metadata.append({"path": item["path"], "bytes": item["bytes"]})
        if smoke and (
            manifest.get("raw_counts") != {s: payload["contract"]["raw"][s]["rows"] for s in ("train", "val")}
            or manifest.get("selected_counts") != counts
            or manifest.get("produced_counts") != {label: counts[label]}
        ):
            raise ValueError("Smoke manifest count metadata mismatch")
        # Trust preparation's stored hashes; retain O(file count) metadata checks.
        # Same-size content mutation requires a separate full integrity audit.
        if feature_metadata(directory) != metadata:
            raise ValueError("Feature contents changed")


@app.function(volumes={"/vol": volume}, secrets=[wandb_secret], timeout=7200)
def preflight(mode="plan", execution=None, feature_scope="full", train_samples=None, eval_samples=None):
    """CPU only: resolve real upstream config/capture and inspect the immutable Volume."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    from specforge.application import resolve_offline_capture
    from specforge.config import Config
    from data.prepare_specforge import audit_prepared, prepare_file, validate_raw

    volume.reload()
    versions = verify_runtime(check_toolchain=True)
    cfg = load_training_config("/root/configs/train_config.yaml")
    target_id = cfg["model"]["target_model_path"]
    target = AutoConfig.from_pretrained(target_id)
    revision = target._commit_hash
    if not revision:
        raise ValueError("Cannot resolve immutable target revision")
    snapshot = snapshot_download(
        target_id, revision=revision,
        allow_patterns=["config.json", "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json", "generation_config.json"],
    )
    cfg["model"]["target_model_path"] = snapshot
    resolved = resolve_offline_capture(Config.model_validate(cfg), target_config=target)
    cfg, key, contract = feature_paths(
        cfg, cfg["model"]["draft_model_config"], feature_scope=feature_scope,
        train_samples=train_samples, eval_samples=eval_samples,
        target_revision=revision, capture_layers=resolved.capture_layers,
    )
    draft = resolved.draft_config.to_dict()
    if not draft.get("fc_norm") or not draft.get("norm_output") or draft["num_hidden_layers"] != 1:
        raise ValueError("EAGLE-3.1 draft invariant failed")
    if len(resolved.capture_layers) != 3:
        raise ValueError("Expected canonical three-layer capture")
    manifest = json.loads(Path("/root/configs/m3_04_dataset.json").read_text())
    raw_root = Path(manifest["raw_root"])
    raw = {split: validate_raw(raw_root / (split + ".jsonl"), expected)
           for split, expected in manifest["splits"].items()}
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    prepared_audit = {
        split: audit_prepared(raw_root / (split + ".jsonl"), tokenizer, expected)
        for split, expected in manifest["splits"].items()
    }
    evidence = Path("/vol/runs/m3-04") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    evidence.mkdir(parents=True, exist_ok=False)
    config_path = evidence / "resolved.yaml"
    payload = {
        "config": cfg, "config_path": str(config_path),
        "contract": contract, "artifact_key": key, "capture_layers": list(resolved.capture_layers),
        "resolved_draft": draft, "versions": versions, "execution": execution,
        "target_revision": revision, "target_id": target_id, "raw": raw,
        "prepared_audit": prepared_audit,
        "best_metric": BEST_METRIC, "evidence": str(evidence),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "modal_function_call_id": modal.current_function_call_id(),
    }
    try:
        (evidence / "runtime.freeze.txt").write_text(Path("/opt/specforge-runtime.freeze.txt").read_text(), encoding="utf-8")
        if mode == "prepare":
            processed = Path("/vol/datasets/m3-04/processed") / key
            processed.mkdir(parents=True, exist_ok=True)
            prepared_manifest = processed / "prepared_manifest.json"
            if prepared_manifest.exists():
                prepared = json.loads(prepared_manifest.read_text())
                if prepared["contract"] != contract:
                    raise ValueError("Prepared contract mismatch")
                for split, item in prepared["splits"].items():
                    if item != prepared_audit[split] or sha256_file(processed / (split + ".jsonl")) != item["prepared_sha256"]:
                        raise ValueError("Prepared data changed")
            else:
                prepared = {"contract": contract, "splits": {}}
                for split, expected in manifest["splits"].items():
                    prepared["splits"][split] = prepare_file(
                        raw_root / (split + ".jsonl"), processed / (split + ".jsonl"),
                        tokenizer, expected, raw_root=raw_root,
                    )
                    if prepared["splits"][split] != prepared_audit[split]:
                        raise ValueError("Prepared serialization differs from CPU audit")
                save_json(prepared_manifest, prepared)
            # Download exact weights on remote CPU before paying for capture GPU.
            snapshot_download(target_id, revision=revision)
            payload["prepared"] = prepared
        elif mode == "features":
            check_features(payload)
            prepare_native_vocab_mapping(payload)
        elif mode != "plan":
            raise ValueError("Unknown preflight mode")
        write_config(cfg, config_path)
        payload["config_sha256"] = sha256_file(config_path)
        result = subprocess.run(
            ["specforge", "train", "-c", str(config_path), "--plan"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (evidence / "plan.log").write_text(result.stdout, encoding="utf-8")
        print(result.stdout)
        result.check_returncode()
        payload["status"] = "CPU_PLAN_PASS"
        save_json(evidence / "evidence.json", payload)
        return payload
    finally:
        volume.commit()


@app.function(gpu="A100-80GB", volumes={"/vol": volume}, timeout=21600)
def prepare_hidden_states(payload):
    """Future GPU operation; canonical extraction, train and validation only."""
    verify_runtime(check_toolchain=True)
    volume.reload()
    cfg = payload["config"]
    counts = feature_counts(payload["contract"])
    smoke = payload["contract"].get("feature_scope") == "smoke"
    processed = Path("/vol/datasets/m3-04/processed") / payload["artifact_key"]
    try:
        for split, field in (("train", "hidden_states_path"), ("val", "eval_hidden_states_path")):
            output = Path(cfg["data"][field])
            output.mkdir(parents=True, exist_ok=False)
            command = [
                "torchrun", "--standalone", "--nproc_per_node=1", "-m", "training.prepare",
                "--config", payload["config_path"],
                "--source", str(processed / (split + ".jsonl")), "--output", str(output),
            ]
            label = "eval" if split == "val" else split
            if smoke:
                command.extend(["--num-samples", str(counts[label])])
            subprocess.run(command, check=True)
            inventory = feature_inventory(output)
            if len(inventory) != counts[label]:
                raise ValueError("Canonical extraction count differs from selected count")
            manifest = {
                "contract": payload["contract"], "capture_layers": payload["capture_layers"],
                "files": inventory, "target_revision": payload["target_revision"],
            }
            if smoke:
                manifest.update(
                    raw_counts={s: payload["contract"]["raw"][s]["rows"] for s in ("train", "val")},
                    selected_counts=counts, produced_counts={label: len(inventory)},
                )
            save_json(output / "artifact_manifest.json", manifest)
    finally:
        volume.commit()


@app.function(gpu="A100-80GB", volumes={"/vol": volume}, secrets=[wandb_secret], timeout=21600)
def train_specforge(payload, smoke=False, resume=False):
    import torch
    from training.train import run_specforge_training
    verify_runtime(check_toolchain=True)
    volume.reload()
    if smoke != (payload["contract"].get("feature_scope") == "smoke"):
        raise ValueError("Training mode and feature scope differ")
    validate_native_vocab_mapping(payload)
    cfg = resolved_config(payload["config"], smoke=smoke, run_suffix=uuid.uuid4().hex[:10], resume=resume)
    output = Path(cfg["output_dir"])
    latest = output / (cfg["run_id"] + "-latest")
    existing = latest.exists() or latest.is_symlink() or any(output.glob(cfg["run_id"] + "-step*")) or (output / (cfg["run_id"] + ".best_meta.json")).exists()
    if existing and not resume:
        raise FileExistsError("Run exists; use explicit --resume or a new run_id")
    if resume and not latest.exists():
        raise FileNotFoundError("Requested resume checkpoint is absent")
    evidence = Path(payload["evidence"])
    cfg["tracking"]["wandb_dir"] = str(evidence / "wandb")
    Path(cfg["tracking"]["wandb_dir"]).mkdir(exist_ok=False)
    path = write_config(cfg, evidence / "execution.yaml")
    save_json(evidence / "execution.json", {
        **payload, "config": cfg, "config_sha256": sha256_file(path),
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "modal_function_call_id": modal.current_function_call_id(),
        "checkpoint_latest": str(latest),
        "checkpoint_best": str(output / (cfg["run_id"] + "-best")),
    })
    try:
        run_specforge_training(path)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(step: str = "plan", resume: bool = False, feature_scope: str = "",
         train_samples: int | None = None, eval_samples: int | None = None):
    if step not in {"plan", "dataset_check", "prepare_hidden_states", "smoke", "train"}:
        raise ValueError("Choose plan, dataset_check, prepare_hidden_states, smoke, train")
    scope = feature_scope or ("smoke" if step == "smoke" else "full")
    if (step == "smoke" and scope != "smoke") or (step == "train" and scope != "full"):
        raise ValueError("Smoke training requires smoke scope; full training requires full scope")
    feature_selection(scope, train_samples, eval_samples)
    options = {"feature_scope": scope, "train_samples": train_samples, "eval_samples": eval_samples}
    execution = execution_source()
    if step in {"plan", "dataset_check"}:
        result = preflight.remote(mode="plan", execution=execution, **options)
        print(json.dumps({"status": result["status"], "evidence": result["evidence"], "capture_layers": result["capture_layers"]}))
    elif step == "prepare_hidden_states":
        result = preflight.remote(mode="prepare", execution=execution, **options)
        prepare_hidden_states.remote(result)
    else:
        result = preflight.remote(mode="features", execution=execution, **options)
        train_specforge.remote(result, smoke=step == "smoke", resume=resume)
