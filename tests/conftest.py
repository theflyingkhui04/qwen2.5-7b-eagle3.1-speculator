"""Historical execution guards, independent of the reviewed identity helpers."""
import ast
from pathlib import Path

import pytest


@pytest.fixture
def historical_modal_source():
    """Reconstruct only the two identity helpers for historical hash assertions.

    The original digest remains unchanged, authenticating every other byte of
    orchestration, image setup and entrypoints. Current identity helpers are
    covered by test_execution_identity.py, not silently blessed by a new hash.
    This reconstruction is test-only and is never used by the execution gate.
    """
    root = Path(__file__).resolve().parents[1]
    source = (root / "modal_pipeline.py").read_bytes().replace(b"\r\n", b"\n")
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    first, last = functions["execution_identity"], functions["execution_source"]
    between = [node for node in ast.parse(source).body if first.lineno <= node.lineno <= last.end_lineno]
    assert [node.name for node in between] == ["execution_identity", "execution_source"]
    lines = source.splitlines(keepends=True)
    legacy = (root / "tests/fixtures/legacy_execution_source.py.txt").read_bytes().replace(b"\r\n", b"\n")
    return b"".join(lines[:first.lineno - 1]) + legacy + b"".join(lines[last.end_lineno:])
