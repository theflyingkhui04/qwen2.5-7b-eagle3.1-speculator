"""CPU tests of diagnostic isolation/orchestration; never exercise a GPU kernel."""
import ast
import copy
import inspect
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from training import flex_attention_ab as diag


def test_selection_keeps_sampler_pairs_and_no_shortening():
    assert diag.short_batch_indices([3, 0, 1, 2, 4, 5], [129, 64, 128, 32, 1, 2]) == (1, [1, 2])
    # Short samples exist, but no original pair is short: regrouping would change data semantics.
    with pytest.raises(ValueError, match="No existing"):
        diag.short_batch_indices([0, 1, 2, 3], [64, 129, 127, 2048])
    with pytest.raises(ValueError, match="microbatch"):
        diag.short_batch_indices([0], [1], batch_size=1)


def test_forward_rebinding_is_per_instance_and_keeps_branch_body():
    namespace = {"create_block_mask": lambda: "raw-mask", "flex_attention": lambda: "raw-attn"}
    exec("class Attention:\n def forward(self, q_len):\n  if q_len <= 128:\n   return create_block_mask(), flex_attention()\n  return 'long'\n", namespace)
    cls = namespace["Attention"]
    a, b = cls(), cls()
    b.forward = diag.isolated_forward(b, lambda: "compiled-mask", lambda: "compiled-attn")
    assert a.forward(128) == ("raw-mask", "raw-attn")
    assert b.forward(128) == ("compiled-mask", "compiled-attn")
    assert b.forward(129) == "long"
    assert b.forward.__func__.__code__ is cls.forward.__code__
    assert namespace["flex_attention"]() == "raw-attn"


def test_raw_deltas_have_no_arbitrary_tolerance_and_expose_zero_reference():
    d = diag.delta(torch.tensor([0., 2.]), torch.tensor([1., 3.]))
    screening = d.pop("proposed_assert_close")
    assert d == {"finite": True, "max_abs": 1., "max_rel_nonzero_reference": .5,
                 "zero_reference_max_abs": 1.}
    assert screening["mismatched_elements"] == 2
    assert diag.delta(torch.tensor([float("nan")]), torch.tensor([1.]))["finite"] is False
    with pytest.raises(ValueError, match="shape"):
        diag.delta(torch.zeros(2), torch.zeros(3))


@pytest.mark.parametrize("dtype,rtol", [(torch.bfloat16, .016), (torch.float32, 1.3e-6)])
def test_proposed_screen_uses_original_dtype_and_matches_explicit_assert_close(dtype, rtol):
    # Near zero, relative-only checking is unstable; one local defect must count
    # even when most elements agree. Compare the CPU FP32 values explicitly.
    a = torch.tensor([0., 1e-8, 1., 2.], dtype=dtype)
    b = torch.tensor([1e-6, 2e-8, 1.01, 3.], dtype=dtype)
    screen = diag.delta(a, b)["proposed_assert_close"]
    assert screen["rtol"] == rtol
    assert screen["atol"] == 1e-5
    assert screen["dtype_a"] == screen["dtype_b"] == str(dtype)
    assert screen["dtype_equal"] is True
    assert screen["shape"] == [4] and screen["numel"] == 4
    assert screen["reference"] == "A"
    assert screen["mismatched_elements"] == (1 if dtype == torch.bfloat16 else 2)
    matches = torch.isclose(b.float(), a.float(), rtol=rtol, atol=1e-5)
    assert screen["mismatched_elements"] == (~matches).sum().item()
    with pytest.raises(AssertionError):
        torch.testing.assert_close(b.float(), a.float(), rtol=rtol, atol=1e-5)
    identical = diag.delta(a, a)["proposed_assert_close"]
    assert identical["mismatched_elements"] == 0
    torch.testing.assert_close(a.float(), a.float(), rtol=rtol, atol=1e-5)


def test_proposed_screen_cannot_pass_dtype_mismatch_unsupported_dtype_or_nonfinite():
    for a, b in [(torch.ones(2), torch.ones(2, dtype=torch.bfloat16)),
                 (torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)),
                 (torch.tensor([float("inf")]), torch.tensor([float("inf")]))]:
        screen = diag.delta(a, b)["proposed_assert_close"]
        assert screen["mismatched_elements"] is None
        assert screen["status"] == "FROZEN"
    assert diag.delta(torch.tensor(0.), torch.tensor(0.))["proposed_assert_close"]["shape"] == []


def test_forward_metrics_have_known_norms_and_explicit_zero_norm_policy():
    metrics = diag.forward_metrics(torch.tensor([3., 4.]), torch.tensor([0., 4.]))
    assert metrics["mean_abs"] == 1.5
    assert metrics["rms_diff"] == pytest.approx(3 / 2 ** .5)
    assert (metrics["l2_a"], metrics["l2_b"], metrics["diff_l2"]) == (5., 4., 3.)
    assert metrics["relative_l2"] == .6
    assert metrics["cosine_similarity"] == pytest.approx(.8)
    assert metrics["proposed_assert_close"]["mismatched_elements"] == 1
    assert metrics["passes_gate"] is False
    zeros = diag.forward_metrics(torch.zeros(2), torch.zeros(2))
    assert zeros["passes_gate"] is True
    assert zeros["relative_l2"] == 0 and zeros["cosine_similarity"] is None
    zero_ref = diag.forward_metrics(torch.zeros(2), torch.ones(2))
    assert zero_ref["relative_l2"] is None and zero_ref["cosine_similarity"] is None
    nonfinite = diag.forward_metrics(torch.ones(2), torch.full((2,), float("nan")))
    assert nonfinite["finite"] is False and nonfinite["passes_gate"] is False
    assert nonfinite["rms_diff"] is None
    json.dumps(nonfinite, allow_nan=False)


def test_input_identity_checks_values_shape_dtype_stride_and_hashes():
    a = torch.ones(2, 2, dtype=torch.bfloat16)
    assert diag.tensor_identity(a, a.clone())["identical"] is True
    changed = a.clone()
    changed[0, 1] = 2
    identity = diag.tensor_identity(a, changed)
    assert identity["identical"] is False
    assert identity["A"]["sha256"] != identity["B"]["sha256"]
    for b in (a.float(), a.reshape(4), a.T, a.detach().requires_grad_(True)):
        assert diag.tensor_identity(a, b)["identical"] is False


@pytest.fixture
def forward_isolation_fixture(monkeypatch):
    """Exercise the real boundary wrapper with CPU stand-ins, never a GPU kernel."""
    import importlib
    events, rng_restores = [], []
    settings = {"fail_iteration": None, "fault": None}
    report = {"selection": {"sample_ids": ["a"], "q_len": 2}}
    calls = {"A": [], "B": []}
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw:
                        self if a == ("cuda",) else original_to(self, *a, **kw))

    def mask(compiled=False, **kwargs):
        fields = ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
                  "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices",
                  "dq_write_order", "dq_write_order_full", "dq_kv_order")
        result = SimpleNamespace(**dict.fromkeys(fields))
        result.kv_num_blocks = torch.ones(1, 1, 1, dtype=torch.int32)
        result.kv_indices = torch.zeros(1, 1, 1, 1, dtype=torch.int32)
        result.seq_lengths = (kwargs["Q_LEN"], kwargs["KV_LEN"])
        result.BLOCK_SIZE = (128, 128)
        result.dq_kv_order_spt = None
        result.mask_mod = lambda *indices: not (compiled and settings["fault"] == "logical_mask")
        if compiled and settings["fault"] == "block_metadata":
            result.kv_indices += 1
        if compiled and settings["fault"] == "dq_metadata":
            result.dq_write_order = torch.ones(1, dtype=torch.int32)
        return result

    def kernel(label, query, key, value, **kwargs):
        assert torch.is_grad_enabled()
        assert query.requires_grad
        # The immediate-output report must not exist before the two kernels run.
        assert "output" not in report["forward_isolation"]["iterations"][-1]
        assert report["forward_isolation"]["iterations"][-1]["arguments"]["identical"]
        calls[label].append(tuple(t.clone() for t in (query, key, value)))
        events.append("kernel_" + label)
        output = query + 1
        if label == "B" and len(calls[label]) - 1 == settings["fail_iteration"]:
            output += .1
        if settings["fault"] == "mutate_" + label:
            with torch.no_grad():
                query.add_(1)
        if settings["fault"] == "exception_" + label:
            raise RuntimeError("kernel failure")
        return output

    class Attention(torch.nn.Module):
        def forward(self, x, iteration):
            q_len = x.shape[-2]
            kv = torch.cat([x] * (iteration + 1), dim=-2)
            block_mask = create_block_mask(mask_mod=lambda: True, B=1, H=1,
                                           Q_LEN=q_len, KV_LEN=kv.shape[-2], device=x.device)
            out = flex_attention(query=x, key=kv, value=kv.clone(), block_mask=block_mask,
                                 enable_gqa=settings["fault"] != "arguments")
            # Both kernel-output capture and comparison must precede projection.
            assert "output" in report["forward_isolation"]["iterations"][-1]
            events.append("projection")
            return out * 2

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.midlayer = torch.nn.Module()
            self.midlayer.self_attn = Attention()
            self.weight = torch.nn.Parameter(torch.tensor(1.))

    model = Model()
    original_forward = model.midlayer.self_attn.forward

    class Strategy:
        def forward_loss(self, batch):
            assert torch.is_grad_enabled()
            x = batch.tensors["hidden_state"]
            for iteration in range(5):
                x = model.midlayer.self_attn(x, iteration)
                events.append("loss")
            return SimpleNamespace(loss=x.sum())

    raw = lambda **kw: kernel("A", **kw)
    compiled = lambda **kw: kernel("B", **kw)
    monkeypatch.setitem(Attention.forward.__globals__, "flex_attention", raw)
    monkeypatch.setitem(Attention.forward.__globals__, "create_block_mask", mask)
    modules = {
        "specforge.runtime.contracts": SimpleNamespace(TrainBatch=lambda **kw: SimpleNamespace(**kw)),
        "specforge.training.backend": SimpleNamespace(FSDPTrainingBackend=SimpleNamespace(
            _set_rng_state=lambda rng: rng_restores.append(rng))),
        "specforge.modeling.draft": SimpleNamespace(llama3_eagle=SimpleNamespace(
            create_block_mask=mask, flex_attention=raw)),
        "specforge.modeling.draft.flex_attention": SimpleNamespace(
            compile_friendly_create_block_mask=lambda **kw: mask(compiled=True, **kw),
            compile_friendly_flex_attention=compiled),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(importlib.import_module("torch.nn.attention.flex_attention"), "create_mask",
                        lambda mod, b, h, q, kv, device: torch.full((b, h, q, kv), mod(), dtype=torch.bool))
    return SimpleNamespace(
        args=(SimpleNamespace(model=model, draft_model=model), Strategy(),
              {"hidden_state": torch.ones(1, 1, 2, 2)}, {"saved": "rng"}, report),
        model=model, original_forward=original_forward, settings=settings,
        report=report, calls=calls, events=events, rng_restores=rng_restores)


@pytest.mark.parametrize("iteration,verdict", [
    (0, "FORWARD DIVERGENCE AT TTT0"), (3, "FORWARD DIVERGENCE EMERGES LATER"),
    (None, "ATTENTION FORWARD PASSES GATE"),
])
def test_forward_isolation_uses_identical_inputs_and_captures_before_projection_loss(
        forward_isolation_fixture, iteration, verdict):
    f = forward_isolation_fixture
    f.settings["fail_iteration"] = iteration
    diag.run_forward_isolation(*f.args)
    evidence = f.report["forward_isolation"]
    assert evidence["verdict"] == verdict
    assert f.events == ["kernel_A", "kernel_B", "projection", "loss"] * 5
    assert [row["kv_len"] for row in evidence["iterations"]] == [2, 4, 6, 8, 10]
    # First pre-projection output is [2,2,2,2], L2=4; projection doubles it.
    assert evidence["iterations"][0]["output"]["l2_a"] == 4
    assert torch.equal(f.calls["A"][1][0], torch.full((1, 1, 2, 2), 4.))
    for raw, compiled, row in zip(f.calls["A"], f.calls["B"], evidence["iterations"]):
        assert all(torch.equal(a, b) for a, b in zip(raw, compiled))
        assert all(item["identical"] for item in row["inputs"].values())
        assert all(item["identical"] for item in row["mask"].values())
        assert row["inputs_unchanged_A"] and row["inputs_unchanged_B"]
        assert row["arguments"]["A"] == row["arguments"]["B"]
        assert row["arguments"]["A"]["scale_argument"] is None
        assert row["arguments"]["A"]["grad_enabled"] is True
    assert evidence["backward_executed"] is False
    assert "timing" not in f.report and "comparison" not in f.report
    assert f.model.midlayer.self_attn.forward == f.original_forward
    assert f.model.weight.grad is None
    assert f.rng_restores == [{"saved": "rng"}] * 2
    json.dumps(f.report, allow_nan=False)


@pytest.mark.parametrize("fault", ["logical_mask", "block_metadata", "dq_metadata", "arguments",
                                  "mutate_A", "mutate_B"])
def test_forward_identity_failure_stops_before_output_attribution(forward_isolation_fixture, fault):
    f = forward_isolation_fixture
    f.settings["fault"] = fault
    with pytest.raises(ValueError, match="INPUTS NOT IDENTICAL"):
        diag.run_forward_isolation(*f.args)
    evidence = f.report["forward_isolation"]
    assert evidence["verdict"] == "INPUTS NOT IDENTICAL"
    assert "output" not in evidence["iterations"][0]
    assert "projection" not in f.events and "loss" not in f.events
    if fault not in {"mutate_A", "mutate_B"}:
        assert not f.calls["A"] and not f.calls["B"]
    assert f.model.midlayer.self_attn.forward == f.original_forward
    assert f.rng_restores == [{"saved": "rng"}] * 2
    assert f.model.weight.grad is None


def test_qkv_identity_failure_prevents_either_kernel(monkeypatch, forward_isolation_fixture):
    f = forward_isolation_fixture
    original = diag.tensor_identity
    def identity(a, b):
        # Inject a genuine changed B tensor into the pre-kernel identity check.
        return original(a, b + 1) if a.dtype == torch.float32 else original(a, b)
    monkeypatch.setattr(diag, "tensor_identity", identity)
    with pytest.raises(ValueError, match="Q/K/V"):
        diag.run_forward_isolation(*f.args)
    assert f.report["forward_isolation"]["verdict"] == "INPUTS NOT IDENTICAL"
    assert not f.events and not f.calls["A"] and not f.calls["B"]


def test_forward_kernel_exception_does_not_claim_divergence_and_restores_state(forward_isolation_fixture):
    f = forward_isolation_fixture
    f.settings["fault"] = "exception_B"
    with pytest.raises(RuntimeError, match="kernel failure"):
        diag.run_forward_isolation(*f.args)
    assert f.report["forward_isolation"]["verdict"] == "NOT EVALUABLE"
    assert f.model.midlayer.self_attn.forward == f.original_forward
    assert f.model.weight.grad is None
    assert f.rng_restores == [{"saved": "rng"}] * 2


def test_gpu_entrypoint_selects_only_forward_isolation_and_keeps_checkpoint_finally():
    tree = ast.parse(inspect.getsource(diag.run_diagnostic))
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "run_forward_isolation" in calls and "run_replays" not in calls
    selected = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "run_forward_isolation")
    assert any(k.arg == "reference_ttt0" and isinstance(k.value, ast.Constant)
               and k.value.value is True for k in selected.keywords)
    assert any(isinstance(n, ast.Try) and n.finalbody and any(
        isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "checkpoint_inventory"
        for statement in n.finalbody for c in ast.walk(statement)) for n in ast.walk(tree))
    forward = ast.parse(inspect.getsource(diag.run_forward_isolation))
    attributes = {n.func.attr for n in ast.walk(forward) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)}
    assert not attributes & {"backward", "step", "compile", "save", "save_checkpoint",
                             "use_deterministic_algorithms", "synchronize", "perf_counter"}
    assert "enable_grad" in attributes and "no_grad" not in attributes


# Exact generator from SpecForge fddcb4ad37733a821708170df28cac2f7bb6ea05,
# specforge/modeling/draft/flex_attention.py (blob f4b2c36d302521d61d5a53fd99ea843c698a77bb).
# Keep the upstream function separate from the implementation under test.
PINNED_EAGLE3_MASK = '''def generate_eagle3_mask(
    seq_lengths: torch.Tensor, Q_LEN: int, KV_LEN: int, lck: int = 0
):

    def causal_mask(b, h, q_idx, kv_idx):
        # Causal will keep shrinking by 1 diagnol due to appended suffix
        # Shirnk the causal by diagnol
        causal_mask = q_idx >= kv_idx
        padding_mask = (kv_idx < seq_lengths[b]) & (q_idx < seq_lengths[b])
        return causal_mask & padding_mask

    def suffix_mask(b, h, q_idx, kv_idx):
        suffix_mask = kv_idx >= Q_LEN
        padding_mask = kv_idx % Q_LEN < seq_lengths[b]
        diagnol_mask = (kv_idx - q_idx) % Q_LEN == 0
        return suffix_mask & padding_mask & diagnol_mask

    mask_mod = or_masks(causal_mask, suffix_mask)
    mask_mod.__name__ = f"eagle3_mask_Q_{Q_LEN}_KV_{KV_LEN}_lck_{lck}"
    return mask_mod
'''


@pytest.mark.parametrize("kv,lck", [(3, 0), (9, 2)])
def test_reference_dense_mask_matches_pinned_eagle3_generator_and_scalar_semantics(kv, lck):
    from torch.nn.attention.flex_attention import create_mask, or_masks
    namespace = {"torch": torch, "or_masks": or_masks}
    exec(PINNED_EAGLE3_MASK, namespace)
    lengths = torch.tensor([3, 1])  # Already adjusted lengths supplied by pinned caller.
    mod = namespace["generate_eagle3_mask"](lengths, Q_LEN=3, KV_LEN=kv, lck=lck)
    dense = diag.dense_reference_mask(mod, (2, 28, 3, kv), "cpu")
    assert torch.equal(dense, create_mask(mod, 2, 28, 3, kv, "cpu"))
    for batch in range(2):
        for q in range(3):
            for k in range(kv):
                causal = q >= k and k < lengths[batch] and q < lengths[batch]
                suffix = k >= 3 and k % 3 < lengths[batch] and (k - q) % 3 == 0
                assert bool(dense[batch, 0, q, k]) == bool(causal or suffix)
    assert torch.equal(dense[:, 0], dense[:, 27])


def test_explicit_reference_gqa_maps_each_of_four_kv_heads_to_seven_query_heads():
    q = torch.zeros(1, 28, 1, 1, dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(1, 4, 1, 1, dtype=torch.bfloat16)
    v = torch.tensor([10., 20., 30., 40.], dtype=torch.bfloat16).reshape(1, 4, 1, 1)
    mask = torch.ones(1, 28, 1, 1, dtype=torch.bool)
    result = diag.explicit_ttt0_attention(q, k, v, mask)
    assert result.flatten().tolist() == [10.] * 7 + [20.] * 7 + [30.] * 7 + [40.] * 7
    assert result.dtype == torch.float32 and result.device.type == "cpu"
    assert not result.requires_grad and torch.is_grad_enabled()


def test_explicit_reference_matches_known_weighted_attention_and_fully_masked_rows():
    import math
    q = torch.tensor([math.log(3), 0.]).reshape(1, 1, 2, 1)
    k = torch.tensor([1., 0.]).reshape(1, 1, 2, 1)
    v = torch.tensor([2., 6.]).reshape(1, 1, 2, 1)
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    result = diag.explicit_ttt0_attention(q, k, v, mask)
    # First row weights are 3/4 and 1/4; second row is uniform.
    torch.testing.assert_close(result.flatten(), torch.tensor([3., 4.]))
    mask[..., 1, :] = False
    result = diag.explicit_ttt0_attention(q, k, v, mask)
    torch.testing.assert_close(result.flatten(), torch.tensor([3., 0.]))
    assert torch.isfinite(result).all()


def test_reference_rejects_non_ttt0_large_shapes_and_nonboolean_masks():
    for length, kv in [(121, 121), (2, 4)]:
        q = torch.zeros(1, 1, length, 1)
        k = torch.zeros(1, 1, kv, 1)
        with pytest.raises(ValueError, match="bounded to TTT0"):
            diag.explicit_ttt0_attention(q, k, k, torch.ones(1, 1, length, kv, dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean"):
        diag.dense_reference_mask(lambda *args: torch.tensor(1.), (1, 1, 1, 1), "cpu")


@pytest.mark.parametrize("raw,compiled,classification", [
    ([1., 2.], [3., 2.], "RAW CLOSER TO REFERENCE"),
    ([3., 2.], [1., 2.], "COMPILED CLOSER TO REFERENCE"),
    ([1., 2.], [1., 2.], "BOTH COMPARABLY CLOSE"),
    ([2., 2.], [2., 2.], "BOTH COMPARABLY CLOSE"),
])
def test_reference_classification_is_symmetric_and_uses_precision_margin(raw, compiled, classification):
    reference = torch.tensor([1., 2.])
    result = diag.reference_comparison(torch.tensor(raw), torch.tensor(compiled), reference, reference.double())
    assert result["classification"] == classification
    assert result["reference_precision_envelope"]["certified_mathematical_error_bound"] is False
    assert "passes_gate" not in result["raw_vs_reference"]
    assert result["raw_vs_reference"]["fp32_screening_only"]["rtol"] == 1.3e-6
    json.dumps(result, allow_nan=False)


def test_reference_classification_does_not_promote_tiny_or_conflicting_metric_differences():
    reference = torch.tensor([1., 0.])
    # Clearer absolute error but worse direction must not become a winner.
    conflicting = diag.reference_comparison(torch.tensor([.9, .1]), torch.tensor([2., 0.]),
                                           reference, reference.double())
    assert conflicting["classification"] == "BOTH COMPARABLY CLOSE"
    a = torch.tensor([1.01, .02])
    b = torch.nextafter(a, torch.full_like(a, float("inf")))
    tiny = diag.reference_comparison(a, b, reference, reference.double())
    assert tiny["classification"] == "BOTH COMPARABLY CLOSE"
    assert tiny["ratios"]["relative_l2_B_over_A"] > 1


def test_reference_invalid_finiteness_is_not_a_correctness_claim():
    ref = torch.tensor([1., float("nan")])
    result = diag.reference_comparison(torch.ones(2), torch.ones(2), ref, ref.double())
    assert result["classification"] == "REFERENCE INVALID / SEMANTICS MISMATCH"


def test_reference_retains_fp32_values_and_native_candidate_dtype_without_using_gate_as_accuracy():
    r = torch.tensor([1.001, 2.001])
    raw = torch.tensor([1., 2.], dtype=torch.bfloat16)
    compiled = torch.tensor([3., 2.], dtype=torch.bfloat16)
    saved = raw.clone()
    result = diag.reference_comparison(raw, compiled, r, r.double())
    assert result["classification"] == "RAW CLOSER TO REFERENCE"
    for key in ("raw_vs_reference", "compiled_vs_reference"):
        m = result[key]
        assert m["candidate_original_dtype"] == "torch.bfloat16"
        assert m["reference_dtype"] == "torch.float32"
        assert m["fp32_screening_only"]["mismatched_elements"] == 2
    assert result["raw_vs_reference"]["mean_abs"] == pytest.approx(.001, abs=2e-7)
    assert torch.equal(raw, saved) and raw.dtype == torch.bfloat16


def test_reference_precision_audit_can_withhold_a_winner():
    r = torch.tensor([1., 2.])
    # A nominal FP32 winner is not stable if reference refinement can reverse it.
    result = diag.reference_comparison(r, r + .05, r, r.double() + .1)
    assert result["classification"] == "BOTH COMPARABLY CLOSE"


def test_ttt0_reference_records_fully_masked_counts_and_checks_candidate_zero_behavior():
    inputs = {name: torch.ones(1, 1, 2, 1) for name in ("query", "key", "value")}
    mask = SimpleNamespace(mask_mod=lambda *indices: False)
    def create_mask(mod, b, h, q, kv, device):
        return torch.zeros(b, h, q, kv, dtype=torch.bool)
    zeros = torch.zeros(1, 1, 2, 1)
    result = diag.ttt0_reference(inputs, mask, mask, zeros, zeros, "cpu", create_mask)
    assert result["classification"] == "BOTH COMPARABLY CLOSE"
    assert result["mask"]["allowed_elements"] == 0
    assert result["mask"]["masked_elements"] == 4
    assert result["mask"]["fully_masked_rows"] == 2
    assert result["mask"]["per_row_allowed_min"] == result["mask"]["per_row_allowed_max"] == 0
    assert result["mask"]["any_fully_masked_row"] is True
    with pytest.raises(ValueError, match="fully masked rows"):
        diag.ttt0_reference(inputs, mask, mask, zeros, torch.ones_like(zeros), "cpu", create_mask)


def test_ttt0_reference_runs_once_before_projection_and_does_not_change_a_b(forward_isolation_fixture, monkeypatch):
    f = forward_isolation_fixture
    original = diag.explicit_ttt0_attention
    calls = []
    def reference(*args, **kwargs):
        assert f.events == ["kernel_A", "kernel_B"]
        assert args == ()
        calls.append(kwargs.get("dtype", torch.float32))
        return original(*args, **kwargs)
    monkeypatch.setattr(diag, "explicit_ttt0_attention", reference)
    diag.run_forward_isolation(*f.args, reference_ttt0=True)
    result = f.report["ttt0_reference"]
    assert calls == [torch.float32, torch.float64]
    assert result["classification"] == "BOTH COMPARABLY CLOSE"
    assert result["mask"]["allowed_elements"] == 4
    assert result["mask"]["masked_elements"] == 0
    assert result["mask"]["per_row_allowed_min"] == result["mask"]["per_row_allowed_max"] == 2
    assert result["mask"]["any_fully_masked_row"] is False
    assert result["kv_head_for_query_head"] == [0]
    assert f.events == ["kernel_A", "kernel_B"]
    assert f.report["forward_isolation"]["completed_attention_calls"] == 1
    assert len(f.calls["A"]) == len(f.calls["B"]) == 1
    assert torch.equal(f.calls["A"][0][0], f.calls["B"][0][0])
    assert f.model.midlayer.self_attn.forward == f.original_forward
    assert f.rng_restores == [{"saved": "rng"}] * 2
    json.dumps(f.report, allow_nan=False)


def test_reference_mask_or_fully_masked_row_mismatch_stops_and_restores(
        forward_isolation_fixture, monkeypatch):
    f = forward_isolation_fixture
    monkeypatch.setattr(diag, "dense_reference_mask", lambda mod, shape, device:
                        torch.zeros(shape, dtype=torch.bool))
    with pytest.raises(ValueError, match="REFERENCE INVALID / SEMANTICS MISMATCH"):
        diag.run_forward_isolation(*f.args, reference_ttt0=True)
    assert f.report["ttt0_reference"]["classification"] == "REFERENCE INVALID / SEMANTICS MISMATCH"
    assert "projection" not in f.events
    assert f.model.midlayer.self_attn.forward == f.original_forward
    assert f.rng_restores == [{"saved": "rng"}] * 2


def test_compile_mode_hook_reuses_authenticated_ttt0_and_stops_before_projection(
        forward_isolation_fixture, monkeypatch):
    from training import flex_compile_mode_ab as ab
    f = forward_isolation_fixture
    calls = []
    def compare(inputs, mask, create_mask, report):
        assert f.events == ["kernel_A", "kernel_B"]
        assert "ttt0_reference" in report
        assert all(torch.equal(inputs[n].detach(), f.calls["A"][0][i])
                   for i, n in enumerate(("query", "key", "value")))
        calls.append(mask)
    monkeypatch.setattr(ab, "run_compile_mode_ab", compare)
    diag.run_forward_isolation(*f.args, reference_ttt0=True, compile_mode_ab=True)
    assert len(calls) == 1
    assert f.events == ["kernel_A", "kernel_B"]
    assert f.model.midlayer.self_attn.forward == f.original_forward


@pytest.fixture
def runtime_guard_fixture(monkeypatch):
    # Stop before features/checkpoint/model assembly; all CUDA queries are mocks.
    for name, module in {
        "specforge.algorithms.eagle3.providers": SimpleNamespace(create_registration=lambda: None),
        "specforge.training.assembly": SimpleNamespace(build_model_bundle=lambda: None),
        "specforge.training.checkpoint": SimpleNamespace(CheckpointManager=object),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    observed = {"available": True, "count": 1, "name": "NVIDIA A100-SXM4-80GB",
                "memory": 80 * 1024 ** 3, "capability": (8, 0), "cuda": "13.0"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: observed["available"])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: observed["count"])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: observed["name"])
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index:
                        SimpleNamespace(total_memory=observed["memory"]))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: observed["capability"])
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    def stop_before_features(checker):
        raise RuntimeError("past runtime guard")
    monkeypatch.setattr(diag, "load_inputs", stop_before_features)
    return observed


@pytest.mark.parametrize("count,cuda,name,failed", [
    (0, "13.0", None, {"single_device", "known_a100_80gb_name", "compute_capability_8_0", "memory_80gb_class"}),
    (2, "13.0", "NVIDIA A100-SXM4-80GB", {"single_device"}),
    (1, "12.4", "NVIDIA A100-SXM4-80GB", {"cuda_13_0"}),
    (1, "13.0", "NVIDIA A10", {"known_a100_80gb_name"}),
    (1, "13.0", "NVIDIA L4", {"known_a100_80gb_name"}),
    (1, "13.0", "NVIDIA H100 80GB HBM3", {"known_a100_80gb_name"}),
    (1, "13.0", "NVIDIA A100-SXM4-40GB", {"known_a100_80gb_name"}),
    (1, "13.0", "NVIDIA A100-SXM4-80GB extra", {"known_a100_80gb_name"}),
])
def test_runtime_guard_failure_reports_actual_values_before_raise(
        monkeypatch, capsys, runtime_guard_fixture, count, cuda, name, failed):
    observed = runtime_guard_fixture
    observed.update(count=count, name=name)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    if not count:
        def forbidden(index):
            pytest.fail("Device zero must not be queried when no GPU is visible")
        for query in ("get_device_name", "get_device_properties", "get_device_capability"):
            monkeypatch.setattr(torch.cuda, query, forbidden)
    report = {}
    with pytest.raises(ValueError, match="Requires a single known A100 80GB") as error:
        diag.run_diagnostic(report, lambda p: None)
    evidence = json.loads(str(error.value).split("; runtime_guard=", 1)[1])
    assert evidence == report["runtime_guard"]
    assert json.loads(capsys.readouterr().out) == {"runtime_guard": evidence}
    assert evidence["cuda_available"] is True
    assert evidence["device_count"] == count and evidence["torch_cuda"] == cuda
    assert evidence["device_name"] == name
    assert evidence["total_memory_bytes"] == (observed["memory"] if count else None)
    assert evidence["compute_capability"] == (list(observed["capability"]) if count else None)
    assert {key for key, passed in evidence["checks"].items() if not passed} == failed
    assert evidence["query_errors"] == {}
    assert report["hardware_policy"] == "noncanonical_a100_80gb_diagnostic"
    for key in ("device_name", "device_count", "total_memory_bytes", "compute_capability", "torch_cuda"):
        assert report[key] == evidence[key]
    assert "selection" not in report and "checkpoint" not in report


@pytest.mark.parametrize("name,comparable", [("NVIDIA A100-SXM4-80GB", True), ("NVIDIA A100 80GB PCIe", False)])
def test_runtime_guard_accepts_only_known_a100_80gb_variants_with_hardware_scope(
        runtime_guard_fixture, name, comparable):
    observed = runtime_guard_fixture
    observed["name"] = name
    report = {}
    with pytest.raises(RuntimeError, match="past runtime guard"):
        diag.run_diagnostic(report, lambda p: None)
    assert all(report["runtime_guard"]["checks"].values())
    assert report["device_name"] == name
    assert report["hardware_policy"] == "noncanonical_a100_80gb_diagnostic"
    assert report["performance_comparable_to_canonical_sxm4"] is comparable
    assert report["performance_comparability_scope"] == "hardware_identity_only"
    assert report["runtime_guard"]["canonical_policy_checks"]["a100_sxm4_80gb_name"] is comparable
    if not comparable:
        assert "no speedup versus historical SXM4" in report["performance_comparability_note"]
        assert "canonical optimizer-step speedup" in report["performance_comparability_note"]
    else:
        assert "comparability is not established" in report["performance_comparability_note"]


@pytest.mark.parametrize("field,value,failed", [
    ("capability", (7, 5), "compute_capability_8_0"),
    ("capability", (9, 0), "compute_capability_8_0"),
    ("memory", 40 * 1024 ** 3, "memory_80gb_class"),
    ("memory", 74_999_999_999, "memory_80gb_class"),
    ("memory", 86_000_000_001, "memory_80gb_class"),
    ("memory", 96_000_000_000, "memory_80gb_class"),
])
def test_runtime_guard_rejects_wrong_capability_and_non_80gb_memory(runtime_guard_fixture, field, value, failed):
    runtime_guard_fixture[field] = value
    report = {}
    with pytest.raises(ValueError, match="Requires a single known A100 80GB"):
        diag.run_diagnostic(report, lambda p: None)
    assert {key for key, passed in report["runtime_guard"]["checks"].items() if not passed} == {failed}
    assert report["performance_comparable_to_canonical_sxm4"] is False
    assert "selection" not in report and "checkpoint" not in report


@pytest.mark.parametrize("memory", [75_000_000_000, 80_000_000_000, 80 * 1024 ** 3, 86_000_000_000])
def test_runtime_guard_memory_class_has_explicit_inclusive_bounds(runtime_guard_fixture, memory):
    runtime_guard_fixture["memory"] = memory
    report = {}
    with pytest.raises(RuntimeError, match="past runtime guard"):
        diag.run_diagnostic(report, lambda p: None)
    assert report["total_memory_bytes"] == memory
    assert report["runtime_guard"]["diagnostic_policy"]["memory_range_bytes_inclusive"] == [75_000_000_000, 86_000_000_000]


@pytest.mark.parametrize("query,field", [("get_device_properties", "total_memory_bytes"),
                                       ("get_device_capability", "compute_capability")])
def test_runtime_guard_requires_memory_and_capability_evidence(monkeypatch, runtime_guard_fixture, query, field):
    def unavailable(index):
        raise RuntimeError("query unavailable")
    monkeypatch.setattr(torch.cuda, query, unavailable)
    report = {}
    with pytest.raises(ValueError, match="query unavailable"):
        diag.run_diagnostic(report, lambda p: None)
    assert report[field] is None
    assert field in report["runtime_guard"]["query_errors"]


def test_runtime_guard_preserves_evidence_when_device_name_query_fails(
        monkeypatch, capsys, runtime_guard_fixture):
    def unavailable(index):
        raise RuntimeError("device name unavailable")
    monkeypatch.setattr(torch.cuda, "get_device_name", unavailable)
    report = {}
    with pytest.raises(ValueError, match="device name unavailable"):
        diag.run_diagnostic(report, lambda p: None)
    assert report["runtime_guard"]["device_name"] is None
    assert report["runtime_guard"]["query_errors"] == {
        "device_name": "RuntimeError: device name unavailable"}
    assert report["runtime_guard"]["total_memory_bytes"] == 80 * 1024 ** 3
    assert json.loads(capsys.readouterr().out) == {"runtime_guard": report["runtime_guard"]}


def result_fixture():
    return {"masks": [{"q_len": 64, "kv_len": (i + 1) * 64,
                       "block_size": (128, 128),
                       "blocks": {"indices": torch.tensor([i]), "optional": None},
                       "logical": torch.ones(2, 1, 64, (i + 1) * 64, dtype=torch.bool)} for i in range(5)],
            "outputs": [torch.ones(2, 64, 4) for _ in range(5)], "loss": torch.tensor(2.),
            "plosses": torch.ones(5), "input_grad": torch.ones(2, 64, 12),
            "parameter_grads": {"q_proj.weight": torch.ones(4, 4)}}


def test_comparison_detects_fine_mask_metadata_output_and_gradient_changes():
    a = result_fixture()
    b = copy.deepcopy(a)
    b["masks"][0]["logical"][0, 0, 0, 0] = False
    b["masks"][1]["blocks"]["indices"][0] = 9
    b["parameter_grads"]["q_proj.weight"][0, 0] = float("inf")
    b["outputs"][0][0, 0, 0] += 1
    report = diag.compare_results(a, b)
    assert report["masks"][0]["logical_equal"] is False
    assert report["masks"][1]["block_metadata_equal"] is False
    assert report["outputs"][0]["max_abs"] == 1
    assert report["all_finite"] is False
    assert report["loss_values"] == {"A": 2., "B": 2.}
    b["masks"][2]["kv_len"] = 1
    with pytest.raises(ValueError, match="progression"):
        diag.compare_results(a, b)


def test_checkpoint_fingerprint_is_read_only_and_detects_mutation(tmp_path):
    checkpoint = tmp_path / "run-step3000"
    checkpoint.mkdir()
    state = checkpoint / "training_state.pt"
    state.write_bytes(b"durable")
    best = tmp_path / "run.best_meta.json"
    best.write_text('{"step":3000}')
    before = diag.checkpoint_inventory(checkpoint)
    assert diag.checkpoint_inventory(checkpoint) == before
    assert state.read_bytes() == b"durable"
    best.write_text('{"step":4000}')
    assert diag.checkpoint_inventory(checkpoint) != before


@pytest.mark.parametrize("nested", [False, True])
def test_real_feature_selection_accepts_capture_pair_padding_and_authenticates_content(tmp_path, monkeypatch, nested):
    from training import native_vocab_mapping as native
    cfg = diag.load_training_config()
    draft = tmp_path / "draft.json"
    draft.write_text('{}')
    cfg["model"]["draft_model_config"] = str(draft)
    cfg["data"]["hidden_states_path"] = str(tmp_path)
    lengths = [129, 64, 100, 127]
    paths, files = [], []
    for i, length in enumerate(lengths):
        # Independent capture fixture: each saved sample retains its original pair's width.
        width = max(lengths[i // 2 * 2:i // 2 * 2 + 2])
        path = tmp_path / "rows_0-2000" / f"data_{i}.ckpt"
        path.parent.mkdir(exist_ok=True)
        torch.save({"input_ids": torch.nn.functional.pad(torch.arange(length), (0, width - length)),
                    "aux_hidden_state": torch.ones(1, width, 12, dtype=torch.bfloat16),
                    "hidden_state": torch.ones(1, width, 4, dtype=torch.bfloat16),
                    "loss_mask": torch.nn.functional.pad(torch.ones(length), (0, width - length))}, path)
        paths.append(str(path))
        files.append({"path": str(path.relative_to(tmp_path)), "sha256": diag.sha256_file(path)})
    (tmp_path / "artifact_manifest.json").write_text(json.dumps({"files": files, "capture_layers": [1, 2, 3]}))
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"target_revision": "frozen", "train_samples": 4}))
    monkeypatch.setattr(native, "PROFILE_PATH", profile_path)
    monkeypatch.setattr(native, "_context", lambda payload: (None, None, paths, tmp_path / "existing-map", None, None))
    class Row:
        def __init__(self, n):
            self.n = n
        def as_py(self):
            values = list(range(self.n))
            return [values] if nested else values
    table = {"input_ids": [Row(n) for n in lengths], "loss_mask": [Row(n) for n in lengths]}
    monkeypatch.setattr(native, "_load_cache", lambda *args: table)
    reconstructed = [native.cache_record_for_feature(
        path, lambda row: {key: table[key][row].as_py() for key in ("input_ids", "loss_mask")}, 4
    )["input_ids"].numel() for path in paths]
    assert reconstructed == [129, 129, 127, 127]
    assert lengths[2] != lengths[3] and reconstructed[2] == reconstructed[3]
    calls = []
    monkeypatch.setattr(native, "validate_native_vocab_mapping", lambda payload: calls.append("mapping_read"))
    monkeypatch.setattr(diag, "load_training_config", lambda path: copy.deepcopy(cfg))
    monkeypatch.setattr(diag, "feature_paths", lambda config, draft: (config, "key", {}))
    typed = SimpleNamespace(training=SimpleNamespace(batch_size=2, accumulation_steps=4, ttt_length=5,
                            attention_backend="flex_attention", seed=42),
                            model=SimpleNamespace(torch_dtype="bfloat16", draft_checkpoint_path=""),
                            data=SimpleNamespace(max_length=2048), mode="offline", validate_world_size=lambda n: None)
    def normalize(raw, max_len):
        calls.append(("normalize", max_len))
        return {"input_ids": raw["input_ids"].unsqueeze(0),
                "hidden_state": raw["aux_hidden_state"], "target": raw["hidden_state"],
                "loss_mask": raw["loss_mask"].unsqueeze(0)}
    def collate(samples):
        calls.append(("collate", len(samples)))
        return {"input_ids": torch.zeros(2, max(s["input_ids"].shape[1] for s in samples), dtype=torch.long)}
    def sampler(size, **kwargs):
        assert kwargs == dict(dp_rank=0, dp_size=1, seed=42, epoch=0, shuffle=True)
        return [0, 1, 2, 3]
    modules = {
        "huggingface_hub": SimpleNamespace(snapshot_download=lambda *a, **kw:
                                          "existing-frozen-snapshot" if kw == dict(revision="frozen", local_files_only=True) else None),
        "specforge.config": SimpleNamespace(Config=SimpleNamespace(model_validate=lambda cfg: typed)),
        "specforge.launch": SimpleNamespace(_distributed_sampler_indices=sampler),
        "specforge.algorithms.eagle3.data": SimpleNamespace(normalize_offline_sample=normalize,
                                                           build_offline_collator=lambda: collate),
        "specforge.runtime.data_plane.feature_store": SimpleNamespace(load_feature_file=lambda path:
                                                                      torch.load(path, weights_only=True)),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    _, tensors, identity = diag.load_inputs(lambda p: calls.append("features_checked"))
    assert tensors["input_ids"].shape == (2, 127)
    assert identity["batch_index"] == 1
    assert [s["capture_row"] for s in identity["samples"]] == [2, 3]
    assert calls == ["features_checked", "mapping_read", ("normalize", 2048),
                     ("normalize", 2048), ("collate", 2)]
    Path(paths[2]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="content changed"):
        diag.load_inputs(lambda p: None)


@pytest.mark.parametrize("failure", [False, True])
def test_replay_correctness_precedes_timing_and_failure_skips_timing(monkeypatch, failure):
    # Use the real replay code with a tiny differentiable CPU strategy and fake CUDA bookkeeping.
    events = []
    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw:
                        self if a == ("cuda",) else original_to(self, *a, **kw))

    class Attention(torch.nn.Module):
        def forward(self, x):
            return flex_attention(x)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.midlayer = SimpleNamespace(self_attn=Attention())
            self.weight = torch.nn.Parameter(torch.tensor(1.))
    model = Model()
    def raw(x):
        events.append("A")
        return x
    def compiled(x):
        events.append("B")
        if failure:
            raise RuntimeError("compiler failed")
        return x
    upstream = SimpleNamespace(create_block_mask=lambda **kw: None, flex_attention=raw)
    # This local test class has its own globals, matching the mechanism of the pinned forward.
    monkeypatch.setitem(Attention.forward.__globals__, "flex_attention", raw)
    modules = {
        "specforge.runtime.contracts": SimpleNamespace(TrainBatch=lambda **kw: SimpleNamespace(**kw)),
        "specforge.training.backend": SimpleNamespace(FSDPTrainingBackend=SimpleNamespace(
            _set_rng_state=lambda rng: None)),
        "specforge.modeling.draft": SimpleNamespace(llama3_eagle=upstream),
        "specforge.modeling.draft.flex_attention": SimpleNamespace(
            compile_friendly_create_block_mask=lambda **kw: None,
            compile_friendly_flex_attention=compiled),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    class Strategy:
        def forward_loss(self, batch):
            x = batch.tensors["hidden_state"]
            for _ in range(5):
                x = model.midlayer.self_attn(x)
            return SimpleNamespace(loss=(x * model.weight).sum(), metrics={"plosses": [x.sum()] * 5})
    def comparison(a, b):
        assert events == ["A"] * 5 + ["B"] * 5
        assert a["input_grad"].sum() == b["input_grad"].sum()
        events.append("correctness_collected")
        return {"all_finite": True}
    monkeypatch.setattr(diag, "compare_results", comparison)
    report = {"selection": {"sample_ids": ["a", "b"]}}
    original = model.midlayer.self_attn.forward
    args = (SimpleNamespace(model=model, draft_model=model), Strategy(),
            {"hidden_state": torch.ones(2, 4, 3)}, {}, report)
    if failure:
        with pytest.raises(RuntimeError, match="timing skipped"):
            diag.run_replays(*args)
        assert "timing" not in report
        assert report["arms"]["A"]["executed"] is True
        assert report["arms"]["B"]["exception"]["type"] == "RuntimeError"
    else:
        diag.run_replays(*args)
        assert events[10] == "correctness_collected"
        assert len(report["timing"]["A"]["warmup"]) == 2
        assert len(report["timing"]["B"]["steady_state"]) == 5
    assert model.midlayer.self_attn.forward == original
    assert model.weight.grad is None


def test_diagnostic_has_no_mutating_training_or_checkpoint_calls():
    tree = ast.parse(inspect.getsource(diag))
    calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
             for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, (ast.Attribute, ast.Name))}
    forbidden = {"step", "fit", "save", "save_checkpoint", "update_best", "symlink_to", "write_text",
                 "write_bytes", "prepare_native_vocab_mapping", "build_training_run", "Trainer",
                 "CheckpointManager", "run_specforge_training", "init", "manual_seed"}
    assert not calls & forbidden
    assert "read_resume_state" in calls
    assert "forward_loss" in calls
    assert "backward" in calls


def test_canonical_function_bodies_match_authenticated_base():
    # AST fingerprints taken from base 2d79c14; includes decorators/resource options.
    expected = {
        "preflight": "ac307a1cd6a5ebcb4aac3f3ba6274c6e16051bbe24565805f5d935f61db9a6a8",
        "prepare_hidden_states": "2e32551410e314c309a72cf0201dfdf958459e3b1368033e9e867ce0dec3d7fa",
        "train_specforge": "81fa62115bd9405fce1c5069910a260dc0bde1dd00422ec0fdf6540e3654b8c2",
        "smoke_specforge": "c593350b959a70847d7b4d48cfc2ece4ac1980ca03a412cd6b6ac780616adf89",
    }
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "modal_pipeline.py").read_text())
    actual = {n.name: hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in expected}
    assert actual == expected
    canonical = (root / "configs/train_config.yaml").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(canonical).hexdigest() == "d135759b45aa1a4207d6f38ef91ae1bd8d8ca75ae1d4b21b92e83777bb992361"
    # Authenticated against 2d79c14 before introducing the diagnostic-only policy.
    runtime_expected = {
        "verify_cuda_toolchain": "39df2b7e4bbe6e327e6637765feba29f29eb473a0a4b7cc331f9180d501a126f",
        "verify_runtime": "8fe7aa30fc1bd188bd47454f9089a5dc9989f2fa8f263031af24c2595174f8da",
    }
    runtime_tree = ast.parse((root / "training/train.py").read_text())
    assert {n.name: hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in runtime_tree.body
            if isinstance(n, ast.FunctionDef) and n.name in runtime_expected} == runtime_expected


def test_hardware_policy_does_not_change_numerical_reference_data_or_checkpoint_contracts():
    expected = {
        "delta": "4ed7e67601b765c22deebb634f322b41958348867ad91e268850270b54fb7967",
        "dense_reference_mask": "bbaa31093912b90337187f7f6d077d07999edf8c31a75ed21bbef76ec7d023e0",
        "explicit_ttt0_attention": "aadd6d0f1c4737ec6c97e16677650825c026bcd7f95ec53cfe0b6ddc5aef0381",
        "reference_comparison": "8aac8c4a2353ec88570b4d371da22a1e472cc0b7635591188c69d7a5abd17831",
        "ttt0_reference": "02b04f728a65f2cb2a9de5ea45ccaff7bca99628b68334984024fd31770aebc1",
        # Only the opt-in compile-mode hook was added; reference functions above stay frozen.
        "run_forward_isolation": "6030e97f4c6145d6a2483988aab50def4fbcff5d620818efc44c2d4df0e982aa",
        "checkpoint_inventory": "b131c2be0518bed18889c9ed36f45974b4809ac3c65aae38add1b73663811ed7",
        "load_inputs": "8d354c618b6f3f23c0ba87e23d2062acd6481dd701f7d6b10d37ed753582c14a",
    }
    tree = ast.parse(inspect.getsource(diag))
    assert {n.name: hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in expected} == expected
