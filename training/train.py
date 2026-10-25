"""Lightweight helpers; the actual SpecForge runtime runs only in Modal Linux."""
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

SPECFORGE_SHA = "fddcb4ad37733a821708170df28cac2f7bb6ea05"
SPECFORGE_ROOT = Path("/opt/SpecForge")
BEST_METRIC = "eval/simulated_acc_len"
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_training_config(config_path="configs/train_config.yaml"):
    with open(config_path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_specforge_command(config_path, *, plan=False):
    cmd = ["specforge", "train", "-c", str(config_path)]
    if plan:
        cmd.append("--plan")
    return cmd


def resolved_config(cfg, *, smoke=False, run_suffix=None, resume=False):
    cfg = copy.deepcopy(cfg)
    if smoke:
        if not run_suffix:
            raise ValueError("Smoke requires a unique run suffix")
        cfg["run_id"] += "-smoke-" + run_suffix
        cfg["tracking"]["wandb_name"] = cfg["run_id"]
        cfg["training"].update(max_steps=2, eval_interval=1, save_interval=1)
    if resume:
        cfg["training"]["resume_from"] = str(Path(cfg["output_dir"]) / (cfg["run_id"] + "-latest"))
    return cfg


def artifact_contract(cfg, draft_path):
    manifest = json.loads((REPO_ROOT / "configs/m3_04_dataset.json").read_text(encoding="utf-8"))
    return {
        "specforge_sha": SPECFORGE_SHA,
        "draft_sha256": sha256_file(draft_path),
        "adapter_sha256": sha256_file(REPO_ROOT / "data/prepare_specforge.py"),
        "loader_bridge_sha256": sha256_file(REPO_ROOT / "training/prepare.py"),
        "target": cfg["model"]["target_model_path"],
        "dtype": cfg["model"]["torch_dtype"],
        "max_length": cfg["data"]["max_length"],
        "chat_template": cfg["data"]["chat_template"],
        "raw": manifest["splits"],
    }


def feature_selection(feature_scope="full", train_samples=None, eval_samples=None):
    if feature_scope not in {"full", "smoke"}:
        raise ValueError("feature_scope must be full or smoke")
    if feature_scope == "full":
        if train_samples is not None or eval_samples is not None:
            raise ValueError("Sample limits apply only to smoke features")
        return {"train": 40000, "eval": 3000}
    counts = {"train": 64 if train_samples is None else train_samples,
              "eval": 16 if eval_samples is None else eval_samples}
    for split, maximum in (("train", 40000), ("eval", 3000)):
        value = counts[split]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
            raise ValueError(f"Invalid smoke {split} sample count")
    if counts["train"] % 8:
        raise ValueError("Smoke train samples must contain complete accumulation windows of 8")
    return counts


def feature_counts(contract):
    if contract.get("feature_scope") == "smoke":
        return feature_selection("smoke", contract["train_samples"], contract["eval_samples"])
    return {"train": contract["raw"]["train"]["rows"], "eval": contract["raw"]["val"]["rows"]}


def feature_paths(cfg, draft_path, *, feature_scope="full", train_samples=None,
                  eval_samples=None, target_revision=None, capture_layers=None):
    counts = feature_selection(feature_scope, train_samples, eval_samples)
    contract = artifact_contract(cfg, draft_path)
    if feature_scope == "smoke":
        if not target_revision or capture_layers is None or len(capture_layers) != 3:
            raise ValueError("Smoke identity requires the resolved target revision and capture layers")
        contract.update(
            feature_scope="smoke", train_samples=counts["train"], eval_samples=counts["eval"],
            selection_method="prepared-prefix-before-upstream-shuffle",
            target_revision=target_revision,
            capture_contract={"strategy": cfg["training"]["strategy"], "layers": list(capture_layers)},
        )
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20]
    cfg = copy.deepcopy(cfg)
    if feature_scope == "smoke":
        cfg["data"]["hidden_states_path"] = f"/vol/hidden_states/m3-04/smoke/{key}/train"
        cfg["data"]["eval_hidden_states_path"] = f"/vol/hidden_states/m3-04/smoke/{key}/eval"
    else:
        cfg["data"]["hidden_states_path"] = f"/vol/hidden_states/m3-04/train/{key}"
        cfg["data"]["eval_hidden_states_path"] = f"/vol/hidden_states/m3-04/eval/{key}"
    cfg["data"]["cache_dir"] = f"/vol/datasets/m3-04/processed/{key}/cache"
    return cfg, key, contract


def write_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    return path


def run_specforge_training(config_path, *, plan=False, dry_run=False):
    cmd = build_specforge_command(config_path, plan=plan)
    if not dry_run:
        if sys.platform != "linux" or not Path("/vol").is_dir():
            raise RuntimeError("Use modal run modal_pipeline.py; runtime requires Modal Linux")
        subprocess.run(cmd, check=True)
    return cmd


def verify_cuda_toolchain(torch_cuda):
    """Compiler/headers/host-tool checks only; no GPU allocation or JIT compilation."""
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = shutil.which("nvcc")
    evidence = {
        "CUDA_HOME": cuda_home, "which_nvcc": nvcc, "nvcc_version": None,
        "torch.version.cuda": torch_cuda,
        "host_tools": {name: shutil.which(name) for name in ("gcc", "g++", "make", "ninja")},
    }
    if nvcc:
        try:
            evidence["nvcc_version"] = subprocess.check_output(
                [nvcc, "--version"], text=True, stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            evidence["nvcc_error"] = str(exc)
    print(json.dumps({"cuda_toolchain": evidence}, sort_keys=True), flush=True)
    if cuda_home != "/usr/local/cuda" or not Path(cuda_home).is_dir():
        raise RuntimeError("CUDA_HOME must point to the CUDA development toolkit at /usr/local/cuda")
    expected_nvcc = Path(cuda_home) / "bin/nvcc"
    if not nvcc or Path(nvcc).resolve() != expected_nvcc.resolve() or not expected_nvcc.is_file():
        raise RuntimeError("CUDA nvcc is missing or PATH does not select CUDA_HOME/bin/nvcc")
    if not (Path(cuda_home) / "include/cuda.h").is_file():
        raise RuntimeError("CUDA development headers are missing")
    release = re.search(r"release (\d+\.\d+),", evidence["nvcc_version"] or "")
    if not release or release.group(1) != "13.0" or torch_cuda != "13.0":
        raise RuntimeError("CUDA toolchain and pinned Torch must both use CUDA 13.0")
    if not all(evidence["host_tools"].values()):
        raise RuntimeError("FlashInfer JIT requires gcc/g++/make/ninja host build tools")
    return evidence


def verify_runtime(*, check_toolchain=False):
    from importlib.metadata import version
    actual = subprocess.check_output(["git", "-C", str(SPECFORGE_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if actual != SPECFORGE_SHA:
        raise RuntimeError("SpecForge checkout revision mismatch")
    versions = {name: version(name) for name in ("specforge", "torch", "transformers", "sglang", "flashinfer-python", "wandb")}
    versions["specforge_sha"] = actual
    print(json.dumps({"runtime_versions": versions}, sort_keys=True), flush=True)
    expected = {"specforge": "0.2.0", "torch": "2.13.0", "transformers": "5.12.1", "sglang": "0.5.18", "flashinfer-python": "0.6.17"}
    for name, wanted in expected.items():
        if versions[name].split("+")[0] != wanted:
            raise RuntimeError(f"Unexpected {name} version")
    if check_toolchain:
        import torch
        versions["cuda_toolchain"] = verify_cuda_toolchain(torch.version.cuda)
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(" ".join(run_specforge_training(args.config, plan=args.plan, dry_run=args.dry_run)))


if __name__ == "__main__":
    main()
