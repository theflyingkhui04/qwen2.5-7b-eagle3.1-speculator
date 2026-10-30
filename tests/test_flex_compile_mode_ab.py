import ast
import inspect
import os
from types import SimpleNamespace

import pytest
import torch

from training import flex_compile_mode_ab as ab
from training.flex_attention_ab import reference_comparison


def test_independent_code_objects_and_exact_compile_options():
    calls = []
    def flex(query, *, flag=True):
        return query
    def compile_fn(fn, **options):
        calls.append((fn, options))
        return lambda *args, **kwargs: fn(*args, **kwargs)
    candidates = ab.compile_candidates(flex, compile_fn)
    assert candidates["default"] is not candidates["max_autotune"]
    assert calls[0][1] == {}
    assert calls[1][1] == {"dynamic": True, "mode": "max-autotune"}
    assert len({id(flex.__code__), *(id(fn.__code__) for fn, _ in calls)}) == 3
    assert all(fn.__globals__ is flex.__globals__ and fn.__kwdefaults__ == flex.__kwdefaults__
               for fn, _ in calls)


@pytest.mark.parametrize("a,b,expected", [
    (.1, .01, "MAX-AUTOTUNE BETTER OR EQUIVALENT"),
    (.01, .1, "MAX-AUTOTUNE NUMERICALLY WORSE"),
    (.01, .01, "MAX-AUTOTUNE BETTER OR EQUIVALENT"),
])
def test_numerical_rule_uses_frozen_reference_ranking_never_speed(a, b, expected):
    r = torch.ones(1, 1, 2, 2)
    result = reference_comparison(r + a, r + b, r, r.double())
    assert ab.numerical_classification(result, inputs_identical=True) == expected
    assert ab.numerical_classification(result, inputs_identical=False) == "NUMERICALLY INCONCLUSIVE"


def test_conflicting_or_invalid_reference_is_inconclusive():
    assert ab.numerical_classification({"classification": "REFERENCE INVALID / SEMANTICS MISMATCH"},
                                     inputs_identical=True) == "NUMERICALLY INCONCLUSIVE"
    assert ab.numerical_classification({"classification": "BOTH COMPARABLY CLOSE",
        "direct_a_vs_b": {"finite": True, "max_abs": 1e-9}},
        inputs_identical=True) == "NUMERICALLY INCONCLUSIVE"


def test_aggregate_improvement_cannot_hide_max_error_regression():
    r = torch.ones(1, 1, 10, 10)
    a, b = r + .1, r.clone()
    a[..., ::2] = r[..., ::2] - .1
    b[0, 0, 0, 0] += .2
    result = reference_comparison(a, b, r, r.double())
    assert result["classification"] == "COMPILED CLOSER TO REFERENCE"
    assert ab.numerical_classification(result, inputs_identical=True) == "NUMERICALLY INCONCLUSIVE"


@pytest.mark.parametrize("fault", [None, "mutate", "fail", "dtype"])
def test_cpu_boundary_same_objects_cache_scope_correctness_then_warm_timing(monkeypatch, fault):
    events, received, cache_dirs = [], [], []
    inputs = {n: torch.ones(1, 1, 2, 2, dtype=torch.bfloat16, requires_grad=True)
              for n in ("query", "key", "value")}
    mask = SimpleNamespace(mask_mod=lambda *indices: True, seq_lengths=(2, 2),
                           BLOCK_SIZE=(128, 128), dq_kv_order_spt=None)
    for n in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
              "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices",
              "dq_write_order", "dq_write_order_full", "dq_kv_order"):
        setattr(mask, n, None)
    def create_mask(mod, b, h, q, kv, device):
        return torch.ones(b, h, q, kv, dtype=torch.bool)
    def compile_fn(fn, **options):
        name = "B" if options else "A"
        def run(**kw):
            events.append(name)
            received.append(kw)
            cache_dirs.append(os.environ["TORCHINDUCTOR_CACHE_DIR"])
            if fault == "fail" and name == "B":
                raise RuntimeError("autotune failed")
            if fault == "mutate" and name == "A":
                with torch.no_grad():
                    kw["query"].add_(1)
            if fault == "dtype" and name == "B":
                return torch.ones_like(kw["query"]).float()
            return torch.ones_like(kw["query"])
        return run
    monkeypatch.setattr(torch, "compile", compile_fn)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    reference = ab.ttt0_reference
    def collect(*args):
        assert events == ["A", "B"]
        events.append("reference")
        return reference(*args)
    monkeypatch.setattr(ab, "ttt0_reference", collect)
    report = {"device_name": "NVIDIA A100 80GB PCIe",
              "performance_comparable_to_canonical_sxm4": False}
    previous = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            ab.run_compile_mode_ab(inputs, mask, create_mask, report)
        assert "steady_state" not in report["compile_mode_ab"]["default"]
    else:
        ab.run_compile_mode_ab(inputs, mask, create_mask, report)
        result = report["compile_mode_ab"]
        assert events == ["A", "B", "reference"] + ["A"] * 7 + ["B"] * 7
        assert len(set(cache_dirs)) == 2
        assert result["hardware"]["performance_comparable_to_canonical_sxm4"] is False
        assert "no historical or optimizer-step comparison" in result["performance_scope"]
        for name in ("default", "max_autotune"):
            assert len(result[name]["steady_state"]["samples"]) == 5
            assert "cold_first_call" in result[name]
            assert result[name]["reference_metrics"]["fp32_screening_only"]["rtol"] == 1.3e-6
        assert result["numerical_classification"] == "MAX-AUTOTUNE BETTER OR EQUIVALENT"
    assert all(call["block_mask"] is mask and call["enable_gqa"] is True
               and all(call[n] is t for n, t in inputs.items()) for call in received)
    assert os.environ.get("TORCHINDUCTOR_CACHE_DIR") == previous


def test_no_training_or_checkpoint_mutations():
    calls = {n.func.attr for n in ast.walk(ast.parse(inspect.getsource(ab)))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"step", "backward", "save", "save_checkpoint", "write_text", "write_bytes"}
