from pathlib import Path

def test_essential_files_exist():
    root = Path(__file__).parent.parent
    assert (root / "requirements.txt").exists(), "requirements.txt must exist"
    assert (root / ".gitignore").exists(), ".gitignore must exist"

def test_requirements_contains_pinned_packages():
    root = Path(__file__).parent.parent
    req_content = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "torch" in req_content
    assert "transformers" in req_content
    assert "safetensors" in req_content
    assert "accelerate" in req_content
