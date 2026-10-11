import pytest
from pathlib import Path

def test_run_all_script_safeguards():
    root = Path(__file__).parent.parent
    script_path = root / "run_all.sh"
    assert script_path.exists(), "run_all.sh must exist"

    content = script_path.read_text(encoding="utf-8")
    assert "pipefail" in content
    assert "disk" in content.lower()
    assert "vastai stop instance" in content
    assert "50" in content  # 50GB disk check
    assert "prepare_hybrid_dataset.py" in content
    assert "extract_hidden_states.py" in content
    assert "train.py" in content
    assert "package_and_upload.py" in content
