from pathlib import Path


def test_essential_files_exist():
    root = Path(__file__).parent.parent
    assert (root / "requirements.txt").exists(), "requirements.txt must exist"
    assert (root / ".gitignore").exists(), ".gitignore must exist"


def test_requirements_contains_pinned_packages():
    root = Path(__file__).parent.parent
    req_content = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "SpecForge.git@fddcb4ad37733a821708170df28cac2f7bb6ea05" in req_content
    assert "modal==1.6.2" in req_content
    assert "pytest" in req_content
