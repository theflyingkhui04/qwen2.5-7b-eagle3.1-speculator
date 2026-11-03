from pathlib import Path


def test_shell_entry_delegates_to_modal_cpu_plan_by_default():
    text = (Path(__file__).resolve().parents[1] / "run_all.sh").read_text()
    assert "set -euo pipefail" in text
    assert '${1:-plan}' in text
    assert 'exec modal run modal_pipeline.py --step "$step"' in text
    assert "pip install" not in text
    assert "prepare_hybrid_dataset.py" not in text
    assert "--upload" not in text
