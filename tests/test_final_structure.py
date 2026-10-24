import json
import pytest
from pathlib import Path


def test_full_repository_layout():
    root = Path(__file__).parent.parent
    required = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "configs/eagle3.1_qwen2.5_7b.json",
        "configs/train_config.yaml",
        "data/prepare_hybrid_dataset.py",
        "training/train.py",
        "export/package_and_upload.py",
        "notebooks/train_on_vastai.ipynb",
        "run_all.sh",
    ]
    for rel_path in required:
        file_path = root / rel_path
        assert file_path.exists(), f"Thiếu file bắt buộc theo spec: {rel_path}"


def test_notebook_is_valid_json():
    root = Path(__file__).parent.parent
    nb_path = root / "notebooks" / "train_on_vastai.ipynb"
    assert nb_path.exists()
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    assert "cells" in nb_data
    assert "metadata" in nb_data
    assert len(nb_data["cells"]) > 0


def test_readme_contains_instructions():
    root = Path(__file__).parent.parent
    readme_path = root / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")
    assert "EAGLE-3.1" in content
    assert "Qwen2.5-7B" in content
    assert "Vast.ai" in content
    assert "vLLM" in content
