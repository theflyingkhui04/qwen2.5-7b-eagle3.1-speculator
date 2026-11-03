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
    resolved_config, sha256_file, validate_resume_checkpoint, verify_runtime, write_config,
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


def execution_identity(source):
    """Execution gate: exact Git revision and complete source-byte inventory.

    Keep working_tree in reports as provenance, never as a source-integrity gate.
    Missing fields fail closed; source_sha256 compares paths as well as digests.
    """
    return {key: source[key] for key in ("branch", "head", "source_sha256")}


def execution_source(*, artifact_dir=None):
    """Inventory the uploaded training package, configs and explicit data bridge.

    Match add_local_python_source's recursive .py scope; exclude bytecode.
    Configs are uploaded as a directory, so inventory all their files recursively.
    Bounded profiler callers also authenticate their command/bootstrap/config.
    The request holding the expected identity is the comparison baseline, not a
    self-hashed input. Never upgrade an older request's inventory implicitly.
    """
    root = str(REPO_ROOT)
    def git(*args, text=True):
        output = subprocess.check_output(
            ["git", "-c", f"safe.directory={Path(root).as_posix()}", "-C", root, *args],
            **({"text": True, "encoding": "utf-8"} if text else {}),
        )
        return output.strip() if text else output
    files = ["modal_pipeline.py", "requirements.txt", "data/__init__.py", "data/prepare_specforge.py"]
    files += [
        p.relative_to(REPO_ROOT).as_posix()
        for p in (REPO_ROOT / "training").rglob("*.py")
        if p.is_file()
    ]
    files += [p.relative_to(REPO_ROOT).as_posix()
              for p in (REPO_ROOT / "configs").rglob("*") if p.is_file()]
    if artifact_dir is not None:
        directory = Path(artifact_dir).resolve()
        files += [(directory / name).relative_to(REPO_ROOT).as_posix() for name in (
            "future-gpu-command.txt", "remote-command.txt", "diagnostic.yaml",
        )]
    return {
        "head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
        # Lossless JSON-safe bytes; no locale decoding, strip or newline conversion.
        "working_tree": "porcelain-v1-z:" + git("status", "--porcelain=v1", "-z", text=False).hex(),
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
def preflight(mode="plan", execution=None, feature_scope="full", train_samples=None, eval_samples=None,
              resume=False, resume_from=None):
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
        allow_patterns=["config.json", "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json", "generation_config.json", "model*safetensors*"],
    )
    legacy_vol = Path("/__modal/volumes/vo-OV83DsWyBYwZAYq9SQ3y2E")
    if not legacy_vol.exists():
        try:
            legacy_vol.parent.mkdir(parents=True, exist_ok=True)
            legacy_vol.symlink_to(Path("/vol").resolve())
        except Exception:
            pass

    canonical_target = "/__modal/volumes/vo-OV83DsWyBYwZAYq9SQ3y2E/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    if feature_scope == "full":
        cfg["model"]["target_model_path"] = canonical_target
    else:
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
            if resume:
                if feature_scope != "full":
                    raise ValueError("Canonical resume requires full feature scope")
                cfg = resolved_config(cfg, resume=True, resume_from=resume_from)
                validate_resume_checkpoint(cfg)
                payload["config"] = cfg
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
    legacy_vol = Path("/__modal/volumes/vo-OV83DsWyBYwZAYq9SQ3y2E")
    if not legacy_vol.exists():
        try:
            legacy_vol.parent.mkdir(parents=True, exist_ok=True)
            legacy_vol.symlink_to(Path("/vol").resolve())
        except Exception:
            pass
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


# 12,000 remaining steps / ~675 steps/hour = 17.8h; allow eval/save/startup headroom.
@app.function(gpu="A100-80GB", volumes={"/vol": volume}, secrets=[wandb_secret], timeout=22 * 3600)
def train_specforge(payload, smoke=False, resume=False, target_step=15000, max_runtime_seconds=28800):
    import torch
    import os
    from training.train import run_specforge_training
    verify_runtime(check_toolchain=True)
    volume.reload()
    legacy_vol = Path("/__modal/volumes/vo-OV83DsWyBYwZAYq9SQ3y2E")
    if not legacy_vol.exists():
        try:
            legacy_vol.parent.mkdir(parents=True, exist_ok=True)
            legacy_vol.symlink_to(Path("/vol").resolve())
        except Exception:
            pass

    # Ensure exact provenance timestamps for resumed runs
    base_hf = Path("/vol/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
    expected_mtimes = {
        base_hf / "model-00001-of-00004.safetensors": 1792929760060396605,
        base_hf / "model-00002-of-00004.safetensors": 1792929746583459492,
        base_hf / "model-00003-of-00004.safetensors": 1792929736758313492,
        base_hf / "model-00004-of-00004.safetensors": 1792929763296686281,
        Path("/vol/datasets/m3-04/processed/a113d35ad76e959e31dd/cache/vocab_mapping/native_canonical.pt"): 1792983477294186449,
    }
    for p, ns in expected_mtimes.items():
        if p.exists():
            try:
                os.utime(p, ns=(ns, ns), follow_symlinks=True)
                os.utime(p, ns=(ns, ns), follow_symlinks=False)
            except Exception:
                pass
    if smoke != (payload["contract"].get("feature_scope") == "smoke"):
        raise ValueError("Training mode and feature scope differ")
    validate_native_vocab_mapping(payload)
    cfg = resolved_config(payload["config"], smoke=smoke, run_suffix=uuid.uuid4().hex[:10], resume=resume,
                          resume_from=payload["config"]["training"].get("resume_from") if resume else None)
    if resume:
        validate_resume_checkpoint(cfg)
    output = Path(cfg["output_dir"])
    latest = output / (cfg["run_id"] + "-latest")
    existing = latest.exists() or latest.is_symlink() or any(output.glob(cfg["run_id"] + "-step*")) or (output / (cfg["run_id"] + ".best_meta.json")).exists()
    if existing and not resume:
        raise FileExistsError("Run exists; use explicit --resume or a new run_id")
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
    step_target = None if smoke else target_step
    try:
        run_specforge_training(path, target_step=step_target, max_runtime_seconds=max_runtime_seconds)
    finally:
        volume.commit()


@app.function(gpu="A100-80GB", volumes={"/vol": volume}, secrets=[wandb_secret], timeout=21600)
def smoke_specforge(payload, resume=False):
    return train_specforge.local(payload, smoke=True, resume=resume)


@app.function(volumes={"/vol": volume}, timeout=1800)
def export_checkpoint(checkpoint_step: int = 15000, output_name: str = "hf_export_step15000"):
    import subprocess
    import shutil
    volume.reload()
    chk_path = f"/vol/checkpoints/qwen2.5-7b-eagle3.1/m3-04-qwen2.5-7b-eagle3.1-step{checkpoint_step}"
    out_dir = Path(f"/vol/checkpoints/qwen2.5-7b-eagle3.1/{output_name}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = Path("/vol/hidden_states/m3-04/train/a113d35ad76e959e31dd/vocab_mapping/vocab_mapping.pt")

    cmd = [
        "specforge", "export",
        "--to", "hf",
        "--checkpoint", str(chk_path),
        "--draft-config", "/root/configs/eagle3.1_qwen2.5_7b.json",
        "--output-dir", str(out_dir),
        "--embedding-source", "Qwen/Qwen2.5-7B-Instruct",
    ]
    if mapping_path.exists():
        cmd.extend(["--vocab-mapping", str(mapping_path)])

    print(f"Running export: {' '.join(cmd)}")
    res = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(res.stdout)
    res.check_returncode()

    files = [f.name for f in out_dir.iterdir()]
    print("Exported files:", files)
    volume.commit()
    return {"status": "SUCCESS", "output_dir": str(out_dir), "files": files}


@app.function(volumes={"/vol": volume}, timeout=1800)
def upload_exported_hf(
    repo_id: str,
    hf_token: str,
    export_name: str = "hf_export_step15000",
    tag: str = "v0.3.0-epoch3",
    tag_message: str = "Epoch 3 Checkpoint (Step 15,000 - Final)",
    commit_message: str = "feat: release EAGLE-3.1 speculator head (epoch 3 - step 15,000, simulated_acc_len=1.579)",
    readme_content: str = "",
):
    from huggingface_hub import HfApi
    from pathlib import Path

    volume.reload()
    export_dir = Path(f"/vol/checkpoints/qwen2.5-7b-eagle3.1/{export_name}")
    if not (export_dir / "model.safetensors").exists() or not (export_dir / "config.json").exists():
        raise FileNotFoundError(f"Export directory {export_dir} is incomplete")

    if readme_content:
        (export_dir / "README.md").write_text(readme_content, encoding="utf-8")
        print(f"✓ Embedded updated README.md into {export_dir}")

    api = HfApi(token=hf_token)
    user = api.whoami()
    username = user.get("name", user.get("fullname", "duckhoi"))
    print(f"✓ Authenticated as Hugging Face user: {username}")

    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    print(f"Uploading {export_dir} to https://huggingface.co/{repo_id}...")

    commit = api.upload_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"Commit completed: {commit}")

    if tag:
        try:
            api.create_tag(repo_id=repo_id, tag=tag, tag_message=tag_message, repo_type="model")
            print(f"✓ Created release tag '{tag}' on Hugging Face!")
        except Exception as e:
            print(f"Note on tag creation: {e}")

    volume.commit()
    return {
        "status": "SUCCESS",
        "repo_url": f"https://huggingface.co/{repo_id}",
        "commit": str(commit),
        "tag": tag,
    }


@app.local_entrypoint()
def main(step: str = "plan", resume: bool = False, feature_scope: str = "",
         train_samples: int | None = None, eval_samples: int | None = None, resume_from: str = "",
         compile_mode_ab: bool = False, hf_token: str = "", repo_id: str = "duckhoi/qwen2.5-7b-eagle3.1-speculator",
         target_step: int = 15000, max_runtime_seconds: int = 28800,
         checkpoint_step: int = 15000, tag: str = "",
         commit_message: str = "docs: update model card with clean model specifications and benchmarks"):
    if compile_mode_ab and step != "flex_attention_ab":
        raise ValueError("Compile-mode A/B requires explicit flex_attention_ab diagnostic")
    if step == "export":
        res = export_checkpoint.remote(checkpoint_step=checkpoint_step, output_name=f"hf_export_step{checkpoint_step}")
        print(json.dumps(res, indent=2))
        return
    if step == "upload_hf":
        readme_path = REPO_ROOT / "MODEL_CARD.md"
        readme_content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
        res = upload_exported_hf.remote(
            repo_id=repo_id,
            hf_token=hf_token,
            export_name=f"hf_export_step{checkpoint_step}",
            tag=tag,
            commit_message=commit_message,
            readme_content=readme_content,
        )
        print(json.dumps(res, indent=2))
        return
    if step not in {"plan", "dataset_check", "prepare_hidden_states", "smoke", "train", "flex_attention_ab", "optimizer_step_ab", "export", "upload_hf"}:
        raise ValueError("Choose plan, dataset_check, prepare_hidden_states, smoke, train, flex_attention_ab, optimizer_step_ab, export, upload_hf")
    if (resume or resume_from) and (step != "train" or not resume):
        raise ValueError("Resume is supported only for canonical train with explicit --resume")
    scope = feature_scope or ("smoke" if step == "smoke" else "full")
    if (step == "smoke" and scope != "smoke") or (step == "train" and scope != "full"):
        raise ValueError("Smoke training requires smoke scope; full training requires full scope")
    feature_selection(scope, train_samples, eval_samples)
    options = {"feature_scope": scope, "train_samples": train_samples, "eval_samples": eval_samples}
    execution = execution_source()
    if step == "optimizer_step_ab":
        if scope != "full":
            raise ValueError("Optimizer-step benchmark requires existing full TRAIN features")
        print(json.dumps(optimizer_step_ab.remote(execution)))
        return
    if step == "flex_attention_ab":
        if scope != "full":
            raise ValueError("Diagnostic requires existing full TRAIN features")
        result = (flex_attention_ab.remote(execution, compile_mode_ab=True) if compile_mode_ab
                  else flex_attention_ab.remote(execution))
        print(json.dumps(result))
        return
    if step in {"plan", "dataset_check"}:
        result = preflight.remote(mode="plan", execution=execution, **options)
        print(json.dumps({"status": result["status"], "evidence": result["evidence"], "capture_layers": result["capture_layers"]}))
    elif step == "prepare_hidden_states":
        result = preflight.remote(mode="prepare", execution=execution, **options)
        prepare_hidden_states.remote(result)
    else:
        if resume:
            options.update(resume=True, resume_from=resume_from or None)
        result = preflight.remote(mode="features", execution=execution, **options)
        if step == "smoke":
            smoke_specforge.remote(result, resume=False)
        else:
            print("==================================================================")
            print("✓ DANG KHOI DONG HUAN LUYEN TREN A100-80GB (MODAL CLOUD)...")
            print(f"Diem dung tu dong (Auto-stop): step{target_step} (Xac thuc ghi file 100% tren dia)")
            print(f"Gioi han toi da: {max_runtime_seconds/3600:.1f} gio (Toi da ~$26.9, bao ve ngan sach $30)")
            print("==================================================================")
            train_specforge.remote(
                result, smoke=False, resume=resume, target_step=target_step, max_runtime_seconds=max_runtime_seconds
            )


@app.function(gpu="A100-80GB", volumes={"/vol": volume}, timeout=1800)
def flex_attention_ab(execution, compile_mode_ab=False):
    """Explicit non-canonical replay; no training entrypoint or W&B secret."""
    from training.flex_attention_ab import record_diagnostic_hardware, run_diagnostic

    volume.reload()
    evidence = Path("/vol/runs/m3-04/non-canonical/flex_attention_ab") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    evidence.mkdir(parents=True, exist_ok=False)
    report = {"status": "ERROR", "semantic_equivalence": "UNDECIDED", "execution": execution,
              "evidence": str(evidence), "modal_function_call_id": modal.current_function_call_id()}
    try:
        record_diagnostic_hardware(report)  # Preserve identity even if software verification fails.
        report["versions"] = verify_runtime(check_toolchain=True)
        if compile_mode_ab:
            run_diagnostic(report, check_features, compile_mode_ab=True)
        else:
            run_diagnostic(report, check_features)
        report["status"] = "EXECUTED"
    except Exception as exc:
        import traceback
        report["exception"] = {"type": type(exc).__name__, "message": str(exc),
                               "traceback": traceback.format_exc()}
        raise
    finally:
        save_json(evidence / "diagnostic.json", report)
        volume.commit()
    return {key: report[key] for key in ("status", "semantic_equivalence", "evidence")}


@app.function(gpu="A100-80GB", volumes={"/vol": volume}, timeout=3600)
def optimizer_step_ab(execution):
    """Non-canonical native training A/B; no W&B secret or canonical launcher."""
    from training.optimizer_step_ab import NAMESPACE, run_benchmark, validated_config
    volume.reload()
    evidence = NAMESPACE / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    report = {"classification": "BENCHMARK INVALID", "execution": execution, "evidence": str(evidence),
              "modal_function_call_id": modal.current_function_call_id()}
    try:
        report["versions"] = verify_runtime(check_toolchain=True)
        run_benchmark(validated_config(check_features), evidence, report)
    except Exception as exc:
        report["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        save_json(evidence / "benchmark.json", report)
        volume.commit()
    return {key: report[key] for key in ("classification", "evidence", "canonical_state_unchanged")}
