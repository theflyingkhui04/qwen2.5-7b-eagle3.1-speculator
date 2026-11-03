"""CPU-only source gate tests: extract actual functions without importing Modal."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "runs/profiler-bounded-auth-20261028T092754Z-ddfa476dc32f"


def load_functions(root, check_output):
    body = []
    for path, names in (
        (ROOT / "training/train.py", {"sha256_file"}),
        (ROOT / "modal_pipeline.py", {"execution_identity", "execution_source"}),
    ):
        body.extend(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name in names)
    namespace = dict(REPO_ROOT=root, Path=Path, hashlib=hashlib,
                     subprocess=SimpleNamespace(check_output=check_output))
    exec(compile(ast.Module(body=body, type_ignores=[]), "<actual-source-gate>", "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def gate(tmp_path):
    # No commits, Modal imports, image construction, GPU, or network calls.
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for name in ("modal_pipeline.py", "requirements.txt", "data/__init__.py",
                 "data/prepare_specforge.py", "training/__init__.py",
                 "training/profiler_stages.py", "configs/train_config.yaml"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original\n")
    revision = {"head": "a" * 40, "branch": "test-branch"}
    calls = []

    def git_only(argv, **kwargs):
        calls.append((argv, kwargs))
        assert argv[0] == "git"
        if argv[-2:] == ["rev-parse", "HEAD"]:
            return revision["head"] + "\n"
        if argv[-2:] == ["branch", "--show-current"]:
            return revision["branch"] + "\n"
        assert argv[-3:] == ["status", "--porcelain=v1", "-z"]
        assert not kwargs  # raw bytes, no decoding or newline conversion
        return subprocess.check_output(argv, **kwargs)

    module = load_functions(tmp_path, git_only)
    return module, tmp_path, revision, calls


def identity(module, **kwargs):
    return module.execution_identity(module.execution_source(**kwargs))


def test_identical_identity_and_required_fields(gate):
    module, _, _, _ = gate
    source = module.execution_source()
    assert identity(module) == module.execution_identity(source)
    assert set(identity(module)) == {"head", "branch", "source_sha256"}
    for key in identity(module):
        with pytest.raises(KeyError):
            module.execution_identity({k: v for k, v in source.items() if k != key})


@pytest.mark.parametrize("name", ["evidence.diff", "screenshot.png", "docs/report.md",
                                  "runs/profiler/training.log", "unrelated/source.py"])
def test_unrelated_evidence_changes_only_provenance(gate, name):
    module, root, _, _ = gate
    before = module.execution_source()
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"evidence")
    after = module.execution_source()
    assert after["working_tree"] != before["working_tree"]
    assert module.execution_identity(after) == module.execution_identity(before)
    assert before["working_tree"].startswith("porcelain-v1-z:")
    records = bytes.fromhex(after["working_tree"].split(":", 1)[1])
    assert records.endswith(b"\0") and b"?? " in records


@pytest.mark.parametrize("name", ["modal_pipeline.py", "requirements.txt",
                                  "configs/train_config.yaml", "data/__init__.py",
                                  "data/prepare_specforge.py", "training/profiler_stages.py"])
def test_content_changes_including_used_untracked_module(gate, name):
    module, root, _, _ = gate
    before = module.execution_source()
    (root / name).write_bytes(b"changed\n")
    after = module.execution_source()
    assert module.execution_identity(after) != module.execution_identity(before)
    assert after["source_sha256"][name] == hashlib.sha256(b"changed\n").hexdigest()
    assert after["working_tree"] == before["working_tree"]  # both still untracked


@pytest.mark.parametrize("name", ["training/nested/new_source.py", "training/.hidden/new_source.py",
                                  "training/__pycache__/source.py", "configs/nested/new.json"])
def test_relevant_add_remove_changes_inventory(gate, name):
    module, root, _, _ = gate
    before = identity(module)
    path = root / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"new")
    added = identity(module)
    assert added != before and name in added["source_sha256"]
    path.unlink()
    assert identity(module) == before


@pytest.mark.parametrize("name", ["data/__init__.py", "data/prepare_specforge.py", "modal_pipeline.py"])
def test_missing_explicit_source_fails_closed(gate, name):
    module, root, _, _ = gate
    (root / name).unlink()
    with pytest.raises(FileNotFoundError):
        identity(module)


@pytest.mark.parametrize("key", ["branch", "head"])
def test_revision_change_invalidates_identity(gate, key):
    module, _, revision, _ = gate
    before = identity(module)
    revision[key] = "changed"
    assert identity(module) != before


def test_provenance_format_and_line_endings_do_not_gate_source(gate):
    module, root, _, _ = gate
    before = module.execution_source()
    for text in (" M file\n", " M file\r\n", "different order/quoting", ""):
        assert module.execution_identity(dict(before, working_tree=text)) == identity(module)
    # Source bytes, unlike status representation, must never be normalized.
    path = root / "training/profiler_stages.py"
    path.write_bytes(b"original\r\n")
    after = module.execution_source()
    assert module.execution_identity(after) != module.execution_identity(before)
    assert after["source_sha256"]["training/profiler_stages.py"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_cache_and_nonuploaded_training_evidence_are_excluded(gate):
    module, root, _, _ = gate
    before = identity(module)
    for name in ("training/__pycache__/ignored.pyc", "training/.hidden/ignored.pyc",
                 "training/result.log", "training/result.diff"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not uploaded Python source")
    assert identity(module) == before


@pytest.mark.parametrize("name", ["future-gpu-command.txt", "remote-command.txt", "diagnostic.yaml"])
def test_profiler_orchestration_and_config_are_hashed(gate, name):
    module, root, _, _ = gate
    plan = root / "runs/review-plan"
    plan.mkdir(parents=True)
    for filename in ("future-gpu-command.txt", "remote-command.txt", "diagnostic.yaml"):
        (plan / filename).write_bytes(b"validated\n")
    before = identity(module, artifact_dir=plan)
    (plan / name).write_bytes(b"changed\n")
    assert identity(module, artifact_dir=plan) != before
    (plan / name).unlink()
    with pytest.raises(FileNotFoundError):
        identity(module, artifact_dir=plan)


def test_current_upload_and_transitive_local_import_closure(gate):
    _, _, _, calls = gate
    module = load_functions(ROOT, lambda *args, **kwargs: b"" if not kwargs else "revision")
    hashes = module.execution_source()["source_sha256"]
    assert {"data/__init__.py", "data/prepare_specforge.py", "modal_pipeline.py",
            "requirements.txt"} <= hashes.keys()
    assert all(p.relative_to(ROOT).as_posix() in hashes for p in (ROOT / "training").rglob("*.py"))
    assert all(p.relative_to(ROOT).as_posix() in hashes for p in (ROOT / "configs").rglob("*") if p.is_file())
    image = ast.parse((ROOT / "modal_pipeline.py").read_text(encoding="utf-8"))
    uploads = {}
    for node in ast.walk(image):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"add_local_python_source", "add_local_file", "add_local_dir"}:
                uploads.setdefault(node.func.attr, []).append(ast.literal_eval(node.args[0]))
    assert uploads == {
        "add_local_python_source": ["training"],
        "add_local_file": ["data/prepare_specforge.py", "data/__init__.py"],
        "add_local_dir": ["configs"],
    }
    # Audit every local absolute import, including imports nested in functions.
    for name in hashes:
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse((ROOT / name).read_text(encoding="utf-8"))):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                       else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for imported in modules:
                if imported.split(".")[0] not in {"training", "data"}:
                    continue
                path = imported.replace(".", "/")
                assert path + ".py" in hashes or path + "/__init__.py" in hashes
    # Only the fake fixture's explicitly restricted Git adapter can be invoked.
    assert not calls


@pytest.mark.skipif(not PLAN.is_dir(), reason="Historical profiler artifact is local ignored evidence")
def test_review_command_uses_identity_gate_and_old_baseline_stays_rejected():
    text = (PLAN / "future-gpu-command.txt").read_text(encoding="utf-8")
    python = text.split("@'", 1)[1].rsplit("'@", 1)[0]
    tree = ast.parse(python)
    gate = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert)
                and isinstance(n.msg, ast.Constant) and n.msg.value == "Source changed since CPU validation")
    assert ast.unparse(gate.test) == (
        "execution_identity(execution_source(artifact_dir=plan)) == execution_identity(request['execution'])"
    )
    module = load_functions(ROOT, lambda *args, **kwargs: b"" if not kwargs else "revision")
    expected = json.loads((PLAN / "request.json").read_text(encoding="utf-8"))["execution"]
    current = module.execution_source(artifact_dir=PLAN)
    current.update(branch=expected["branch"], head=expected["head"])
    assert module.execution_identity(current) != module.execution_identity(expected)
    assert "data/__init__.py" not in expected["source_sha256"]


def test_legacy_partial_inventory_is_not_upgraded(gate):
    module, _, _, _ = gate
    current = module.execution_source()
    old = dict(current, source_sha256={key: value for key, value in current["source_sha256"].items()
                                      if key != "data/__init__.py"})
    assert module.execution_identity(old) != module.execution_identity(current)
