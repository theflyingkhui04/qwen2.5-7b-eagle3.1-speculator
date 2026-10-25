"""CPU-only regression checks; never import real torch or invoke CUDA/Modal."""
import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import training.train as training
from training.prepare import preparation_args


@pytest.fixture
def toolchain(monkeypatch, tmp_path):
    root = tmp_path / "usr/local/cuda"
    (root / "bin").mkdir(parents=True)
    (root / "include").mkdir()
    (root / "bin/nvcc").write_text("mock compiler")
    (root / "include/cuda.h").write_text("mock header")
    monkeypatch.setattr(training, "Path", lambda p: Path(tmp_path / str(p).lstrip("/")))
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    monkeypatch.setattr(training.shutil, "which", lambda name:
                        "/usr/local/cuda/bin/nvcc" if name == "nvcc" else "/usr/bin/" + name)
    output = {"nvcc": "Cuda compilation tools, release 13.0, V13.0.88"}
    monkeypatch.setattr(training.subprocess, "check_output",
                        lambda command, **kwargs: training.SPECFORGE_SHA if command[0] == "git" else output["nvcc"])
    return root, output


def test_cuda_compiler_evidence_without_gpu(toolchain, capsys):
    evidence = training.verify_cuda_toolchain("13.0")
    assert evidence["which_nvcc"] == "/usr/local/cuda/bin/nvcc"
    assert evidence["CUDA_HOME"] == "/usr/local/cuda"
    assert evidence["torch.version.cuda"] == "13.0"
    assert "release 13.0" in evidence["nvcc_version"]
    assert all(evidence["host_tools"].values())
    assert json.loads(capsys.readouterr().out)["cuda_toolchain"] == evidence


@pytest.mark.parametrize("fault", ["nvcc", "home", "headers", "compiler_version", "torch_version", "host_tools"])
def test_invalid_cuda_environment_fails_before_model_loading(toolchain, monkeypatch, fault):
    root, output = toolchain
    torch_cuda = "13.0"
    if fault == "nvcc":
        monkeypatch.setattr(training.shutil, "which", lambda name: None if name == "nvcc" else "/usr/bin/" + name)
    elif fault == "home":
        monkeypatch.setenv("CUDA_HOME", "/wrong/cuda")
    elif fault == "headers":
        (root / "include/cuda.h").unlink()
    elif fault == "compiler_version":
        output["nvcc"] = "Cuda compilation tools, release 13.1, V13.1.0"
    elif fault == "torch_version":
        torch_cuda = "12.8"
    elif fault == "host_tools":
        monkeypatch.setattr(training.shutil, "which", lambda name:
                            "/usr/local/cuda/bin/nvcc" if name == "nvcc" else None)
    with pytest.raises(RuntimeError):
        training.verify_cuda_toolchain(torch_cuda)


def test_runtime_records_frozen_versions_and_toolchain_without_cuda_calls(toolchain, monkeypatch, capsys):
    versions = {"specforge": "0.2.0", "torch": "2.13.0+cu130", "transformers": "5.12.1",
                "sglang": "0.5.18", "flashinfer-python": "0.6.17", "wandb": "test"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    # This fake torch has no cuda device API: evidence must not allocate/query a GPU.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda="13.0")))
    result = training.verify_runtime(check_toolchain=True)
    assert result["sglang"] == "0.5.18"
    assert result["flashinfer-python"] == "0.6.17"
    assert result["specforge_sha"] == training.SPECFORGE_SHA
    assert result["cuda_toolchain"]["torch.version.cuda"] == "13.0"
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert "runtime_versions" in lines[0]
    assert "cuda_toolchain" in lines[1]


def test_runtime_rejects_flashinfer_version_change(toolchain, monkeypatch):
    versions = {"specforge": "0.2.0", "torch": "2.13.0", "transformers": "5.12.1",
                "sglang": "0.5.18", "flashinfer-python": "different", "wandb": "test"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    with pytest.raises(RuntimeError, match="flashinfer-python"):
        training.verify_runtime(check_toolchain=True)


def test_protocol_and_backend_unchanged():
    cfg = training.load_training_config()
    assert training.SPECFORGE_SHA == "fddcb4ad37733a821708170df28cac2f7bb6ea05"
    assert cfg["model"]["target_model_path"] == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg["model"]["torch_dtype"] == "bfloat16"
    assert cfg["training"]["attention_backend"] == "flex_attention"
    assert cfg["training"]["strategy"] == "eagle3"
    assert cfg["training"]["num_epochs"] == 3
    assert cfg["training"]["batch_size"] == 2
    assert cfg["training"]["accumulation_steps"] == 4
    assert cfg["training"]["learning_rate"] == 1e-4
    assert cfg["training"]["ttt_length"] == 5
    assert cfg["data"]["max_length"] == 2048
    assert training.feature_selection("smoke") == {"train": 64, "eval": 16}
    args = preparation_args(cfg, "prepared.jsonl", "smoke", 64)
    # Keep the pinned canonical target capture default: FlashInfer.
    assert "--sglang-attention-backend" not in args
    assert training.BEST_METRIC == "eval/simulated_acc_len"
