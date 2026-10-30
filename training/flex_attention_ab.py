"""One read-only, non-canonical EAGLE3 replay. Imported only by the explicit Modal step.

No trainer, optimizer, scheduler, checkpoint writer or tracking client is constructed.
Heavy/pinned imports stay lazy so selection, isolation and reporting can be tested on CPU.
"""
import copy
import time
from pathlib import Path
from types import FunctionType, MethodType

from training.train import (
    REPO_ROOT, feature_paths, load_training_config, sha256_file, validate_resume_checkpoint,
)


# Baseline identity for reporting, not a change to canonical launch/validation.
CANONICAL_HARDWARE_POLICY = {
    "device_name": "NVIDIA A100-SXM4-80GB", "device_count": 1, "torch_cuda": "13.0",
}
DIAGNOSTIC_HARDWARE_POLICY = "noncanonical_a100_80gb_diagnostic"
DIAGNOSTIC_DEVICE_NAMES = (CANONICAL_HARDWARE_POLICY["device_name"], "NVIDIA A100 80GB PCIe")
# Temporary diagnostic SKU-class window, not usable/free VRAM: accommodates
# nominal 80 GB (decimal) and 80 GiB, with reporting/reservation allowance.
# Exact names and CC are required independently; 40 GB/MIG and 96 GB fail.
DIAGNOSTIC_MEMORY_RANGE_BYTES = (75_000_000_000, 86_000_000_000)


def short_batch_indices(order, lengths, batch_size=2):
    """Select a whole existing sampler microbatch; never regroup or truncate samples."""
    if batch_size != 2:
        raise ValueError("Diagnostic requires canonical microbatch 2")
    for offset in range(0, len(order) - batch_size + 1, batch_size):
        indices = order[offset:offset + batch_size]
        if all(0 < lengths[i] <= 128 for i in indices):
            return offset // batch_size, indices
    raise ValueError("No existing TRAIN microbatch has q_len <= 128; no synthetic fallback")


def isolated_forward(attention, mask_fn, attention_fn):
    """Rebind only this instance's pinned forward globals, never the upstream module."""
    original = type(attention).forward
    namespace = dict(original.__globals__)
    namespace.update(create_block_mask=mask_fn, flex_attention=attention_fn,
                     compile_friendly_create_block_mask=mask_fn,
                     compile_friendly_flex_attention=attention_fn)
    fn = FunctionType(original.__code__, namespace, original.__name__,
                      original.__defaults__, original.__closure__)
    fn.__kwdefaults__ = original.__kwdefaults__
    return MethodType(fn, attention)


def delta(a, b):
    """Deltas and the frozen screening metric; never declare semantic equivalence.

    Explicit native-dtype tolerances from PyTorch v2.13.0 _comparison.py
    (cf30153c4c131c8164ee7798e5022d810682e2cb), frozen by the root-cause task.
    The legacy proposed_assert_close key is retained for report compatibility.
    Failure must not trigger post-hoc tolerance widening.
    Compare on CPU in FP32, as for existing deltas, with A as expected/reference.
    """
    import torch
    if a.shape != b.shape:
        raise ValueError(f"Comparison shape mismatch: {a.shape} vs {b.shape}")
    tolerances = {torch.bfloat16: (0.016, 1e-5), torch.float32: (1.3e-6, 1e-5)}
    same_dtype = a.dtype == b.dtype
    tolerance = tolerances.get(a.dtype) if same_dtype else None
    screening = {
        "status": "FROZEN",
        "shape": list(a.shape), "dtype_a": str(a.dtype), "dtype_b": str(b.dtype),
        "dtype_equal": same_dtype, "comparison_dtype": "torch.float32",
        "reference": "A", "numel": a.numel(),
        "rtol": tolerance[0] if tolerance else None,
        "atol": tolerance[1] if tolerance else None,
        "mismatched_elements": None,
    }
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"finite": False, "max_abs": None, "max_rel_nonzero_reference": None,
                "zero_reference_max_abs": None, "proposed_assert_close": screening}
    if tolerance is not None:
        # assert_close delegates value comparison to isclose. Count failures
        # without throwing, so one failed tensor cannot discard other evidence.
        matches = torch.isclose(b, a, rtol=tolerance[0], atol=tolerance[1], equal_nan=False)
        screening["mismatched_elements"] = (~matches).sum().item()
    error = (a - b).abs()
    nonzero = a != 0
    return {"finite": True, "max_abs": error.max().item() if error.numel() else 0.0,
            "max_rel_nonzero_reference": (error[nonzero] / a[nonzero].abs()).max().item()
            if nonzero.any() else 0.0,
            "zero_reference_max_abs": error[~nonzero].max().item()
            if (~nonzero).any() else 0.0, "proposed_assert_close": screening}


def tensor_identity(a, b):
    """Exact value hashes plus invocation metadata, before any output attribution."""
    import hashlib
    import torch

    def describe(t):
        raw = t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device),
                "stride": list(t.stride()), "requires_grad": t.requires_grad,
                "finite": bool(torch.isfinite(t).all()),
                "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest()}

    left, right = describe(a), describe(b)
    return {"A": left, "B": right,
            "identical": left == right and torch.equal(a, b)}


def forward_metrics(a, b):
    """FP64 CPU reductions; frozen elementwise screening remains CPU FP32."""
    import torch
    result = delta(a, b)
    names = ("mean_abs", "rms_diff", "l2_a", "l2_b", "diff_l2", "relative_l2", "cosine_similarity")
    result.update(dict.fromkeys(names))
    result["reduction_dtype"] = "torch.float64"
    result["zero_norm_policy"] = "relative L2 undefined for zero A unless both zero; cosine undefined"
    if result["finite"]:
        left, right = a.detach().cpu().double().reshape(-1), b.detach().cpu().double().reshape(-1)
        diff = left - right
        na, nb, nd = (torch.linalg.vector_norm(t).item() for t in (left, right, diff))
        result.update(mean_abs=diff.abs().mean().item() if diff.numel() else 0.,
                      rms_diff=diff.square().mean().sqrt().item() if diff.numel() else 0.,
                      l2_a=na, l2_b=nb, diff_l2=nd,
                      relative_l2=nd / na if na else (0. if nd == 0 else None),
                      cosine_similarity=torch.dot(left / na, right / nb).item() if na and nb else None)
    gate = result["proposed_assert_close"]
    result["passes_gate"] = (result["finite"] and gate["dtype_equal"]
                             and gate["mismatched_elements"] == 0)
    return result


def dense_reference_mask(mask_mod, shape, device):
    """Evaluate the original mask closure directly, independently of block metadata."""
    import torch
    b, h, q, kv = shape
    if not (0 < b <= 2 and 0 < h <= 28 and 0 < q <= 120 and 0 < kv <= 120):
        raise ValueError("Reference mask exceeds the bounded TTT0 dimensions")
    indices = [torch.arange(n, device=device).view(*[n if i == j else 1 for j in range(4)])
               for i, n in enumerate(shape)]
    allowed = torch.as_tensor(mask_mod(*indices), device=device)
    if allowed.dtype != torch.bool:
        raise ValueError("Reference mask must be boolean")
    return allowed.expand(shape).contiguous()


def explicit_ttt0_attention(query, key, value, dense_mask, *, dtype=None):
    """Independent bounded CPU attention; no FlexAttention or SDPA call.

    Primary arithmetic is FP32; FP64 is used only to audit reference precision.
    Pinned PyTorch math_attention repeats each KV head consecutively for GQA.
    Its _safe_softmax, and the Triton template, give zero for fully masked rows.
    Sources: v2.13.0/cf30153, torch/_higher_order_ops/flex_attention.py:249-280,
    aten/src/ATen/native/transformers/attention.cpp:_safe_softmax,
    torch/_inductor/kernel/flex/templates/flex_attention.py.jinja:193-198,
    torch/_inductor/kernel/flex/flex_decoding.py:439-458 (short-query reduction).
    """
    import math
    import torch
    dtype = torch.float32 if dtype is None else dtype
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("Invalid reference Q/K/V shapes")
    b, hq, q, d = query.shape
    bk, hkv, kv, dk = key.shape
    if (not (0 < b <= 2 and 0 < hq <= 28 and 0 < hkv <= 4 and 0 < d <= 128
             and 0 < q <= 120 and kv == q and b == bk and d == dk and hq % hkv == 0)
            or dense_mask.shape != (b, hq, q, kv) or dense_mask.dtype != torch.bool
            or dtype not in (torch.float32, torch.float64)):
        raise ValueError("Reference is bounded to TTT0 with matching GQA/mask semantics")
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        query, key, value = (t.detach().to(device="cpu", dtype=dtype) for t in (query, key, value))
        if not all(bool(torch.isfinite(t).all()) for t in (query, key, value)):
            raise ValueError("Nonfinite reference inputs")
        key = key.repeat_interleave(hq // hkv, dim=1)
        value = value.repeat_interleave(hq // hkv, dim=1)
        allowed = dense_mask.detach().cpu()
        scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        scores = scores.masked_fill(~allowed, -float("inf"))
        fully_masked = ~allowed.any(dim=-1, keepdim=True)
        # Avoid softmax(-inf,...,-inf) NaNs, then explicitly zero these rows.
        scores = torch.where(fully_masked, 0., scores)
        probabilities = torch.softmax(scores, dim=-1, dtype=dtype)
        probabilities = torch.where(fully_masked, 0., probabilities)
        return probabilities @ value


def reference_comparison(raw, compiled, reference, refined_reference):
    """Predeclared comparative ranking, never a reference-accuracy acceptance gate.

    Nontrivial separation means error intervals do not overlap under an empirical
    precision envelope: max(FP32/FP64 reference delta, one FP32 output ULP), per
    element. Triangle inequalities give L2/RMS/mean-absolute intervals; reference
    direction changes by at most 2*envelope_L2/reference_L2. Cosine must not
    contradict the error ordering beyond this sensitivity. This is a conservative
    precision-sensitivity rule, NOT a certified bound on the mathematical error.
    No observed A/B number or new percentage tolerance enters this rule.
    """
    import math
    import torch
    result = {"classification": "REFERENCE INVALID / SEMANTICS MISMATCH",
              "screening_role": "Existing FP32 gate after promotion; screening only, not accuracy acceptance",
              "classification_basis": "FP32/FP64 reference sensitivity plus one FP32 output ULP; no percentage cutoff"}
    tensors = (raw, compiled, reference, refined_reference)
    if (reference.dtype != torch.float32 or refined_reference.dtype != torch.float64
            or any(t.shape != reference.shape or not bool(torch.isfinite(t).all()) for t in tensors)):
        return {**result, "reason": "Reference/candidate shape, dtype or finiteness invalid"}
    r = reference.detach().cpu()
    r64 = refined_reference.detach().cpu()

    def metrics(ref, candidate):
        candidate_values = candidate.detach().cpu().to(ref.dtype)
        m = forward_metrics(ref, candidate_values)
        screen = m.pop("proposed_assert_close")
        m.pop("passes_gate")  # No automatic reference-accuracy PASS.
        m["candidate_original_dtype"] = str(candidate.dtype)
        m["reference_dtype"] = str(ref.dtype)
        m["reference_l2"] = m.pop("l2_a")
        m["candidate_l2"] = m.pop("l2_b")
        m["max_abs"] = (ref.double() - candidate_values.double()).abs().max().item()
        if ref.dtype == torch.float32:
            screen["reference"] = "R"
            m["fp32_screening_only"] = screen
        return m

    a, b = metrics(r, raw), metrics(r, compiled)
    result.update(raw_vs_reference=a, compiled_vs_reference=b,
                  fp64_audit={"raw": metrics(r64, raw), "compiled": metrics(r64, compiled)},
                  direct_a_vs_b=forward_metrics(raw, compiled),
                  ratio_zero_denominator_policy="null; inspect the corresponding absolute errors")
    result["ratios"] = {
        f"{key}_B_over_A": b[key] / a[key] if a[key] not in (None, 0.) else None
        for key in ("relative_l2", "rms_diff", "mean_abs")}
    result["cosine_B_minus_A"] = (b["cosine_similarity"] - a["cosine_similarity"]
                                  if a["cosine_similarity"] is not None and b["cosine_similarity"] is not None else None)
    spacing = torch.maximum((torch.nextafter(r, torch.full_like(r, float("inf"))) - r).double().abs(),
                            (r - torch.nextafter(r, torch.full_like(r, -float("inf")))).double().abs())
    envelope = torch.maximum((r.double() - r64).abs(), spacing)
    if not bool(torch.isfinite(envelope).all()):
        return {**result, "reason": "Nonfinite reference precision envelope"}
    uncertainty = torch.linalg.vector_norm(envelope).item()
    norm = a["reference_l2"]
    mean_uncertainty = envelope.mean().item()
    rms_uncertainty = uncertainty / math.sqrt(r.numel())
    result["reference_precision_envelope"] = {
        "l2": uncertainty, "mean_abs": mean_uncertainty, "rms": rms_uncertainty,
        "fp32_fp64_diff_l2": torch.linalg.vector_norm(r.double() - r64).item(),
        "certified_mathematical_error_bound": False}
    result["classification"] = "BOTH COMPARABLY CLOSE"
    if norm <= uncertainty or any(m["cosine_similarity"] is None for m in (a, b)):
        return {**result, "reason": "Scale/direction ordering unresolved, including zero-norm cases"}

    def intervals(m):
        return {
            "relative_l2": [max(0., m["diff_l2"] - uncertainty) / (norm + uncertainty),
                            (m["diff_l2"] + uncertainty) / (norm - uncertainty)],
            "rms_diff": [max(0., m["rms_diff"] - rms_uncertainty), m["rms_diff"] + rms_uncertainty],
            "mean_abs": [max(0., m["mean_abs"] - mean_uncertainty), m["mean_abs"] + mean_uncertainty],
            "cosine_similarity": [max(-1., m["cosine_similarity"] - 2 * uncertainty / norm),
                                  min(1., m["cosine_similarity"] + 2 * uncertainty / norm)]}

    ia, ib = intervals(a), intervals(b)
    result["sensitivity_intervals"] = {"raw": ia, "compiled": ib}
    errors = ("relative_l2", "rms_diff", "mean_abs")
    if all(ia[k][1] < ib[k][0] for k in errors) and ia["cosine_similarity"][1] >= ib["cosine_similarity"][0]:
        result["classification"] = "RAW CLOSER TO REFERENCE"
    elif all(ib[k][1] < ia[k][0] for k in errors) and ib["cosine_similarity"][1] >= ia["cosine_similarity"][0]:
        result["classification"] = "COMPILED CLOSER TO REFERENCE"
    else:
        result["reason"] = ("No consistent winner outside reference precision sensitivity; "
                            "this label does not assert absolute accuracy")
    return result


def ttt0_reference(inputs, mask_a, mask_b, raw, compiled, device, create_mask):
    """Reuse captured tensors and the live EAGLE3 mask closure; no new data path."""
    import math
    import torch
    b, h, q, _ = inputs["query"].shape
    kv = inputs["key"].shape[-2]
    dense = dense_reference_mask(mask_a.mask_mod, (b, h, q, kv), device).cpu()
    authentic = [create_mask(mask.mask_mod, b, h, q, kv, device).cpu() for mask in (mask_a, mask_b)]
    if not all(torch.equal(dense, mask) for mask in authentic):
        raise ValueError("Reference dense mask differs from A/B logical masks")
    allowed_counts = dense.sum(dim=-1)
    fully_masked = allowed_counts == 0
    if any(bool(torch.count_nonzero(out[fully_masked])) for out in (raw, compiled)):
        raise ValueError("A/B fully masked rows disagree with pinned zero-output semantics")
    reference = explicit_ttt0_attention(**inputs, dense_mask=dense)
    refined = explicit_ttt0_attention(**inputs, dense_mask=dense, dtype=torch.float64)
    return {**reference_comparison(raw, compiled, reference, refined),
            "ttt_iteration": 0, "q_len": q, "kv_len": kv, "lck": 0,
            "reference_device": "cpu", "reference_dtype": "torch.float32",
            "scale": 1.0 / math.sqrt(inputs["query"].shape[-1]), "enable_gqa": True,
            "kv_head_for_query_head": [i // (h // inputs["key"].shape[1]) for i in range(h)],
            "mask": {"mask_mod_name": getattr(mask_a.mask_mod, "__name__", None),
                     "inputs_source": "original mask_mod closure, including seq_lengths; no metadata reconstruction",
                     "authenticated_against_both_masks": True, "logical_sha256": tensor_identity(dense, dense)["A"]["sha256"],
                     "shape": list(dense.shape), "allowed_elements": int(dense.sum()),
                     "masked_elements": dense.numel() - int(dense.sum()),
                     "per_row_allowed_min": int(allowed_counts.min()), "per_row_allowed_max": int(allowed_counts.max()),
                     "any_fully_masked_row": bool(fully_masked.any()), "fully_masked_rows": int(fully_masked.sum()),
                     "fully_masked_behavior": "zero output, authenticated in pinned eager and Triton sources"}}


def run_forward_isolation(bundle, strategy, tensors, rng, report, *, reference_ttt0=False, compile_mode_ab=False):
    """Pair kernels at the call boundary on one raw-driven, forward-only TTT trace.

    Returning only A prevents divergent B recurrence from contaminating later
    Q/K/V. This measures local kernel error, NOT independent A/B TTT propagation.
    No backward, determinism experiment, warmups or performance replay occurs.
    """
    import math
    import torch
    from specforge.runtime.contracts import TrainBatch
    from specforge.training.backend import FSDPTrainingBackend
    from specforge.modeling.draft import llama3_eagle as upstream
    from specforge.modeling.draft.flex_attention import (
        compile_friendly_create_block_mask, compile_friendly_flex_attention,
    )
    from torch.nn.attention.flex_attention import create_mask

    attention = bundle.draft_model.midlayer.self_attn
    original = attention.forward
    evidence = report["forward_isolation"] = {
        "verdict": "NOT EVALUABLE", "iterations": [], "backward_executed": False,
        "trajectory": "A/raw only; B outputs never enter projection, loss or TTT recurrence",
        "scope": "same-input kernel forward, not independent A/B recurrence or full equivalence",
        "autograd": "training forward with requires_grad preserved; backward never called",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    }
    if reference_ttt0:
        evidence.update(trajectory="TTT0 only; neither output enters projection, loss or recurrence",
                        scope="TTT0 independent reference only; configured TTT length remains 5")
    pending = None

    class TTT0ReferenceComplete(Exception):
        """Unwind the diagnostic forward at the capture boundary, before o_proj."""

    def stop(reason):
        evidence.update(verdict="INPUTS NOT IDENTICAL", stop_reason=reason)
        raise ValueError(f"INPUTS NOT IDENTICAL: {reason}; no output attribution")

    def paired_mask(**kwargs):
        nonlocal pending
        if pending is not None or len(evidence["iterations"]) >= 5:
            stop("Unexpected mask/attention call order")
        a = upstream.create_block_mask(**kwargs)
        b = compile_friendly_create_block_mask(**kwargs)
        row = {"ttt_iteration": len(evidence["iterations"]), "q_len": kwargs["Q_LEN"],
               "kv_len": kwargs["KV_LEN"], "mask": {}}
        evidence["iterations"].append(row)
        # Include the deterministic-backward metadata added in pinned PyTorch 2.13.
        tensor_fields = ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
                         "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices",
                         "dq_write_order", "dq_write_order_full", "dq_kv_order")
        for name in tensor_fields:
            left, right = getattr(a, name), getattr(b, name)
            row["mask"][name] = (tensor_identity(left, right) if left is not None and right is not None
                                 else {"identical": left is None and right is None})
        for name in ("seq_lengths", "BLOCK_SIZE", "dq_kv_order_spt"):
            left, right = getattr(a, name), getattr(b, name)
            row["mask"][name] = {"A": left, "B": right, "identical": left == right}
        logical = [create_mask(mask.mask_mod, kwargs["B"], kwargs["H"], kwargs["Q_LEN"],
                               kwargs["KV_LEN"], kwargs["device"]) for mask in (a, b)]
        row["mask"]["logical"] = tensor_identity(*logical)
        if not all(item["identical"] for item in row["mask"].values()):
            stop("Logical mask or BlockMask metadata differs")
        pending = (a, b, row)
        return a

    def paired_attention(query, key, value, **kwargs):
        nonlocal pending
        if pending is None or kwargs.get("block_mask") is not pending[0]:
            stop("Attention did not receive the authenticated mask")
        mask_a, mask_b, row = pending
        pending = None
        q, kv, iteration = query.shape[-2], key.shape[-2], row["ttt_iteration"]
        if (not 0 < q <= 128 or kv != (iteration + 1) * q or value.shape[-2] != kv
                or (q, kv) != (row["q_len"], row["kv_len"])
                or q != report["selection"]["q_len"]):
            stop("Unexpected Q/K/V lengths or TTT progression")
        # The pinned caller passes only block_mask and enable_gqa=True. Refuse
        # silent new score modifiers/options rather than changing their behavior.
        if set(kwargs) != {"block_mask", "enable_gqa"} or kwargs["enable_gqa"] is not True:
            stop("Pinned FlexAttention arguments changed")
        row["arguments"] = {
            "A": {"scale_argument": None, "effective_scale": 1.0 / math.sqrt(query.shape[-1]),
                  "enable_gqa": True, "score_mod": None, "kernel_options": None,
                  "return_lse": False, "return_aux": None,
                  "grad_enabled": torch.is_grad_enabled()},
        }
        row["arguments"]["B"] = dict(row["arguments"]["A"])
        row["arguments"]["identical"] = True
        originals = dict(query=query, key=key, value=value)
        inputs_a = {name: t.detach().clone().requires_grad_(t.requires_grad) for name, t in originals.items()}
        inputs_b = {name: t.detach().clone().requires_grad_(t.requires_grad) for name, t in originals.items()}
        row["inputs"] = {name: tensor_identity(inputs_a[name], inputs_b[name]) for name in originals}
        row["clone_preserves_original"] = {
            name: tensor_identity(originals[name], inputs_a[name])["identical"] for name in originals}
        if not all(item["identical"] and item["A"]["finite"] for item in row["inputs"].values()) or not all(
                row["clone_preserves_original"].values()):
            stop("Q/K/V values, dtype, device, stride or finiteness differs before kernels")
        snapshots = {name: t.detach().cpu().clone() for name, t in inputs_a.items()}
        row["input_metrics"] = {name: forward_metrics(inputs_a[name], inputs_b[name]) for name in originals}
        out_a = upstream.flex_attention(**inputs_a, **kwargs)
        saved_a = out_a.detach().cpu().clone()  # Immediately on return, before o_proj.
        row["inputs_unchanged_A"] = all(torch.equal(t.detach().cpu(), snapshots[n]) for n, t in inputs_a.items())
        if not row["inputs_unchanged_A"]:
            stop("A mutated attention inputs")
        out_b = compile_friendly_flex_attention(**inputs_b, **{**kwargs, "block_mask": mask_b})
        saved_b = out_b.detach().cpu().clone()  # B never enters projection/loss.
        row["inputs_unchanged_B"] = all(torch.equal(t.detach().cpu(), snapshots[n]) for n, t in inputs_b.items())
        if not row["inputs_unchanged_B"]:
            stop("B mutated attention inputs")
        row["output"] = forward_metrics(saved_a, saved_b)
        if reference_ttt0 and iteration == 0:
            report["ttt0_reference"] = {"classification": "REFERENCE INVALID / SEMANTICS MISMATCH"}
            try:
                report["ttt0_reference"] = ttt0_reference(
                    snapshots, mask_a, mask_b, saved_a, saved_b, query.device, create_mask)
            except (ValueError, RuntimeError) as exc:
                report["ttt0_reference"]["reason"] = str(exc)
                raise ValueError(f"REFERENCE INVALID / SEMANTICS MISMATCH: {exc}") from exc
            if report["ttt0_reference"]["classification"] == "REFERENCE INVALID / SEMANTICS MISMATCH":
                raise ValueError("REFERENCE INVALID / SEMANTICS MISMATCH: reference precision validation failed")
            if compile_mode_ab:
                from training.flex_compile_mode_ab import run_compile_mode_ab
                run_compile_mode_ab(inputs_a, mask_a, create_mask, report)
            evidence["stop_reason"] = "TTT0 reference collected; stop before projection, loss and TTT1-TTT4"
            raise TTT0ReferenceComplete
        return out_a

    report["phase"] = "forward_isolation"
    try:
        attention.forward = isolated_forward(attention, paired_mask, paired_attention)
        FSDPTrainingBackend._set_rng_state(rng)
        bundle.model.zero_grad(set_to_none=True)
        # Preserve the grad-enabled training-forward specialization of compiled
        # FlexAttention. Forward-only means no backward, not inference mode.
        with torch.enable_grad():
            data = {name: value.detach().clone().to("cuda") for name, value in tensors.items()}
            data["hidden_state"].requires_grad_(True)
            batch = TrainBatch(sample_ids=report["selection"]["sample_ids"], strategy="eagle3",
                               tensors=data, metadata={"target_repr": "hidden_state", "ttt_length": 5})
            try:
                strategy.forward_loss(batch)
            except TTT0ReferenceComplete:
                pass
        expected_calls = 1 if reference_ttt0 else 5
        if pending is not None or len(evidence["iterations"]) != expected_calls:
            raise ValueError(f"Expected exactly {expected_calls} complete TTT attention calls")
        evidence["completed_attention_calls"] = expected_calls
        failures = [r["ttt_iteration"] for r in evidence["iterations"] if not r["output"]["passes_gate"]]
        evidence["verdict"] = ("FORWARD DIVERGENCE AT TTT0" if failures and failures[0] == 0 else
                               "FORWARD DIVERGENCE EMERGES LATER" if failures else
                               "ATTENTION FORWARD PASSES GATE")
        report["phase"] = "done"
    finally:
        attention.forward = original
        bundle.model.zero_grad(set_to_none=True)
        FSDPTrainingBackend._set_rng_state(rng)


def checkpoint_inventory(checkpoint):
    """Content fingerprints plus run pointers, read before and after the replay."""
    result = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint must contain regular files, not links")
        if path.is_file():
            result[str(path.relative_to(checkpoint))] = sha256_file(path)
    for suffix in ("-latest", "-best", ".best_meta.json"):
        path = checkpoint.parent / (checkpoint.name.removesuffix("-step3000") + suffix)
        result[suffix] = (str(path.readlink()) if path.is_symlink()
                          else sha256_file(path) if path.is_file() else None)
    return result


def load_inputs(check_features):
    """Use sealed TRAIN inventory/cache and the pinned sampler, normalizer and collator."""
    import json
    from huggingface_hub import snapshot_download
    from specforge.config import Config
    from specforge.launch import _distributed_sampler_indices
    from specforge.algorithms.eagle3.data import normalize_offline_sample, build_offline_collator
    from specforge.runtime.data_plane.feature_store import load_feature_file
    from training.native_vocab_mapping import (
        PROFILE_PATH, _context, _load_cache, validate_native_vocab_mapping,
    )

    cfg = load_training_config(REPO_ROOT / "configs/train_config.yaml")
    profile = json.loads(PROFILE_PATH.read_text())
    cfg["model"]["target_model_path"] = snapshot_download(
        cfg["model"]["target_model_path"], revision=profile["target_revision"], local_files_only=True)
    draft_path = cfg["model"]["draft_model_config"]
    cfg, key, contract = feature_paths(cfg, draft_path)
    manifest = json.loads((Path(cfg["data"]["hidden_states_path"]) / "artifact_manifest.json").read_text())
    payload = {"config": cfg, "artifact_key": key, "contract": contract,
               "resolved_draft": json.loads(Path(draft_path).read_text()),
               "target_revision": profile["target_revision"], "capture_layers": manifest["capture_layers"]}
    check_features(payload)
    context = _context(payload)
    cfg["model"]["vocab_mapping_path"] = str(context[3])
    validate_native_vocab_mapping(payload)  # Read only: never prepare/rebuild a mapping.
    table = _load_cache(context[1], profile)
    typed = Config.model_validate(cfg)
    typed.validate_world_size(1)
    if (typed.training.batch_size != 2 or typed.training.accumulation_steps != 4
            or typed.training.ttt_length != 5 or typed.training.attention_backend != "flex_attention"
            or typed.model.torch_dtype != "bfloat16" or typed.mode != "offline"
            or typed.model.draft_checkpoint_path):
        raise ValueError("Canonical diagnostic contract mismatch")
    # This is a selection from epoch 0, not a resume or a new training order.
    order = _distributed_sampler_indices(len(context[2]), dp_rank=0, dp_size=1,
                                         seed=typed.training.seed, epoch=0, shuffle=True)
    # Cache rows are indexed by original capture row, not lexicographic feature order.
    from training.native_vocab_mapping import cache_record_for_feature, cache_sample_index
    rows = [cache_sample_index(p, profile["train_samples"]) for p in context[2]]
    def get_row(index):
        return {key: table[key][index].as_py() for key in ("input_ids", "loss_mask")}
    # Saved features retain capture-pair padding; individual cache rows do not.
    lengths = [min(cache_record_for_feature(path, get_row, profile["train_samples"])["input_ids"].numel(),
                   typed.data.max_length) for path in context[2]]
    batch_index, indices = short_batch_indices(order, lengths)
    selected, identities = [], []
    for i in indices:
        path = context[2][i]
        entry = manifest["files"][i]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("Selected TRAIN feature content changed")
        sample = normalize_offline_sample(load_feature_file(path), max_len=typed.data.max_length)
        if sample["input_ids"].shape[1] != lengths[i]:
            raise ValueError("Actual TRAIN feature length disagrees with authenticated cache")
        import torch
        if any(sample[name].dtype != torch.bfloat16 for name in ("hidden_state", "target")):
            raise ValueError("Selected TRAIN features must retain canonical BF16 dtype")
        selected.append(sample)
        identities.append({"source": path, "sha256": entry["sha256"], "feature_index": i,
                           "capture_row": rows[i], "sample_id": f"{cfg['run_id']}:{i:08d}"})
    tensors = build_offline_collator()(selected)
    if tensors["input_ids"].shape != (2, max(lengths[i] for i in indices)):
        raise ValueError("Unexpected collated microbatch shape")
    identity = {"samples": identities, "epoch": 0, "sampler_seed": typed.training.seed,
                "batch_index": batch_index, "q_len": tensors["input_ids"].shape[1],
                "config": cfg, "artifact_key": key}
    return typed, tensors, identity


def run_replays(bundle, strategy, tensors, rng, report):
    import torch
    from specforge.runtime.contracts import TrainBatch
    from specforge.training.backend import FSDPTrainingBackend
    from specforge.modeling.draft import llama3_eagle as upstream
    from specforge.modeling.draft.flex_attention import (
        compile_friendly_create_block_mask, compile_friendly_flex_attention,
    )

    attention = bundle.draft_model.midlayer.self_attn
    original = attention.forward
    capture = None
    raw_mask = upstream.create_block_mask
    raw_attn = upstream.flex_attention
    pairs = {"A": (raw_mask, raw_attn),
             "B": (compile_friendly_create_block_mask, compile_friendly_flex_attention)}

    def record_mask(fn, **kwargs):
        mask = fn(**kwargs)
        if capture is not None:
            fields = ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
                      "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices")
            from torch.nn.attention.flex_attention import create_mask
            capture["masks"].append({
                "q_len": kwargs["Q_LEN"], "kv_len": kwargs["KV_LEN"],
                "block_size": tuple(mask.BLOCK_SIZE),
                "blocks": {name: getattr(mask, name).detach().cpu().clone()
                           if getattr(mask, name) is not None else None for name in fields},
                "logical": create_mask(kwargs["mask_mod"], kwargs["B"], kwargs["H"],
                                       kwargs["Q_LEN"], kwargs["KV_LEN"], kwargs["device"]).cpu(),
            })
        return mask

    def replay(label, collect=False):
        nonlocal capture
        capture = {"masks": [], "outputs": []} if collect else None
        mask_fn, attn_fn = pairs[label]
        attention.forward = isolated_forward(
            attention, (lambda **kw: record_mask(mask_fn, **kw)) if collect else mask_fn, attn_fn)
        hook = attention.register_forward_hook(
            lambda mod, args, out: capture["outputs"].append(out.detach().cpu().clone())) if collect else None
        # All replay inputs/weights/RNG are identical; no seed changes or optimizer exists.
        FSDPTrainingBackend._set_rng_state(rng)
        bundle.model.zero_grad(set_to_none=True)
        data = {key: value.detach().clone().to("cuda") for key, value in tensors.items()}
        data["hidden_state"].requires_grad_(True)
        batch = TrainBatch(sample_ids=report["selection"]["sample_ids"], strategy="eagle3",
                           tensors=data, metadata={"target_repr": "hidden_state", "ttt_length": 5})
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            result = strategy.forward_loss(batch)
            torch.cuda.synchronize()
            forward_s = time.perf_counter() - start
            start = time.perf_counter()
            result.loss.backward()
            torch.cuda.synchronize()
            backward_s = time.perf_counter() - start
            memory = {"allocated_bytes": torch.cuda.max_memory_allocated(),
                      "reserved_bytes": torch.cuda.max_memory_reserved()}
            if collect:
                capture["loss"] = result.loss.detach().cpu().clone()
                capture["plosses"] = torch.stack(result.metrics["plosses"]).detach().cpu()
                capture["input_grad"] = data["hidden_state"].grad.detach().cpu().clone()
                capture["parameter_grads"] = {
                    name: p.grad.detach().cpu().clone() for name, p in bundle.draft_model.named_parameters()
                    if p.requires_grad and p.grad is not None}
                expected = {name for name, p in bundle.draft_model.named_parameters() if p.requires_grad}
                if set(capture["parameter_grads"]) != expected:
                    raise ValueError("Missing trainable parameter gradient")
            return capture, {"forward_s": forward_s, "backward_s": backward_s, "peak_cuda": memory}
        finally:
            if hook is not None:
                hook.remove()

    try:
        results = {}
        report["first_call"] = {}
        import traceback
        failures = {}
        report["arms"] = {}
        for label in pairs:
            report["phase"] = f"correctness_{label}"
            try:
                results[label], report["first_call"][label] = replay(label, collect=True)
                result = results[label]
                report["arms"][label] = {
                    "executed": True,
                    "loss": result["loss"].item() if torch.isfinite(result["loss"]) else None,
                    "q_kv_lengths": [(m["q_len"], m["kv_len"]) for m in result["masks"]],
                    "finite": all(torch.isfinite(t).all().item() for t in
                                  result["outputs"] + [result["loss"], result["plosses"], result["input_grad"]]
                                  + list(result["parameter_grads"].values())),
                }
            except Exception as exc:
                failures[label] = {"type": type(exc).__name__, "message": str(exc),
                                   "traceback": traceback.format_exc()}
                report["arms"][label] = {"executed": False, "exception": failures[label],
                                          "q_kv_lengths": [(m["q_len"], m["kv_len"])
                                                           for m in (capture or {}).get("masks", [])]}
                torch.cuda.empty_cache()
        if failures:
            raise RuntimeError(f"Correctness replay failed in arm(s): {', '.join(failures)}; timing skipped")
        report["comparison"] = compare_results(results["A"], results["B"])
        # All correctness evidence is now collected. No equivalence PASS threshold is invented.
        report["phase"] = "timing"
        report["timing"] = {}
        for label in pairs:
            warmup = [replay(label)[1] for _ in range(2)]
            steady = [replay(label)[1] for _ in range(5)]
            report["timing"][label] = {"warmup": warmup, "steady_state": steady}
        report["phase"] = "done"
    finally:
        attention.forward = original
        bundle.model.zero_grad(set_to_none=True)
        FSDPTrainingBackend._set_rng_state(rng)


def compare_results(a, b):
    import torch
    if len(a["masks"]) != 5 or len(b["masks"]) != 5 or len(a["outputs"]) != 5 or len(b["outputs"]) != 5:
        raise ValueError("Expected exactly five TTT attention calls per arm")
    masks = []
    for i, (left, right) in enumerate(zip(a["masks"], b["masks"])):
        q, kv = left["q_len"], left["kv_len"]
        if not 0 < q <= 128 or kv != (i + 1) * q or (q, kv) != (right["q_len"], right["kv_len"]):
            raise ValueError("Unexpected q_len / KV progression")
        blocks_equal = all(
            (value is None and right["blocks"][name] is None)
            or (value is not None and right["blocks"][name] is not None
                and torch.equal(value, right["blocks"][name]))
            for name, value in left["blocks"].items())
        masks.append({"ttt_iteration": i, "q_len": q, "kv_len": kv,
                      "block_size_equal": left["block_size"] == right["block_size"],
                      "logical_equal": torch.equal(left["logical"], right["logical"]),
                      "block_metadata_equal": blocks_equal})
    if a["parameter_grads"].keys() != b["parameter_grads"].keys():
        raise ValueError("A/B parameter gradient sets differ")
    comparisons = {"outputs": [delta(x, y) for x, y in zip(a["outputs"], b["outputs"])],
                   "loss": delta(a["loss"], b["loss"]), "ttt_losses": delta(a["plosses"], b["plosses"]),
                   "input_gradient": delta(a["input_grad"], b["input_grad"]),
                   "parameter_gradients": {n: delta(a["parameter_grads"][n], b["parameter_grads"][n])
                                           for n in a["parameter_grads"]}}
    flat = comparisons["outputs"] + [comparisons[k] for k in ("loss", "ttt_losses", "input_gradient")]
    flat += list(comparisons["parameter_gradients"].values())
    return {**comparisons, "masks": masks, "all_finite": all(x["finite"] for x in flat),
            "loss_values": {label: r["loss"].item() if torch.isfinite(r["loss"]) else None
                            for label, r in (("A", a), ("B", b))},
            "ttt_loss_values": {label: [v.item() if torch.isfinite(v) else None for v in r["plosses"]]
                                for label, r in (("A", a), ("B", b))}}


def hardware_policy_checks(runtime):
    """Separate baseline-reference and diagnostic policies; no canonical caller."""
    common = {"single_device": runtime["device_count"] == 1,
              "cuda_13_0": runtime["torch_cuda"] == "13.0"}
    memory = runtime["total_memory_bytes"]
    low, high = DIAGNOSTIC_MEMORY_RANGE_BYTES
    return {
        "canonical_sxm4": {**common, "a100_sxm4_80gb_name": runtime["device_name"] == CANONICAL_HARDWARE_POLICY["device_name"]},
        DIAGNOSTIC_HARDWARE_POLICY: {
            **common, "known_a100_80gb_name": runtime["device_name"] in DIAGNOSTIC_DEVICE_NAMES,
            "compute_capability_8_0": runtime["compute_capability"] == [8, 0],
            "memory_80gb_class": memory is not None and low <= memory <= high,
        },
    }


def record_diagnostic_hardware(report):
    """Populate report even on early software/query failures; never allocate tensors.

    Called only by the noncanonical Modal entrypoint and this diagnostic. The
    entrypoint captures identity before software verification; run_diagnostic
    refreshes it before applying the diagnostic policy.
    """
    runtime = {"cuda_available": None, "device_count": None,
               "device_name": None, "total_memory_bytes": None, "compute_capability": None,
               "torch_cuda": None, "query_errors": {}}
    fields = ("device_name", "total_memory_bytes", "compute_capability", "torch_cuda", "device_count")
    report.update({name: None for name in fields})
    report.update(hardware_policy=DIAGNOSTIC_HARDWARE_POLICY,
                  performance_comparable_to_canonical_sxm4=False,
                  performance_comparability_scope="hardware_identity_only",
                  performance_comparability_note="Software/config/protocol comparability is not established")
    report["runtime_guard"] = runtime
    import torch
    runtime["torch_cuda"] = torch.version.cuda
    for name, query in {"cuda_available": torch.cuda.is_available, "device_count": torch.cuda.device_count}.items():
        try:
            runtime[name] = query()
        except Exception as exc:
            runtime["query_errors"][name] = f"{type(exc).__name__}: {exc}"
    if runtime["device_count"] is not None and runtime["device_count"] > 0:
        for name, query in {
            "device_name": lambda: torch.cuda.get_device_name(0),
            "total_memory_bytes": lambda: torch.cuda.get_device_properties(0).total_memory,
            "compute_capability": lambda: list(torch.cuda.get_device_capability(0)),
        }.items():
            try:
                runtime[name] = query()
            except Exception as exc:
                runtime["query_errors"][name] = f"{type(exc).__name__}: {exc}"
    policies = hardware_policy_checks(runtime)
    runtime.update(checks=policies[DIAGNOSTIC_HARDWARE_POLICY], canonical_policy_checks=policies["canonical_sxm4"],
                   hardware_policy=DIAGNOSTIC_HARDWARE_POLICY,
                   canonical_policy=dict(CANONICAL_HARDWARE_POLICY),
                   diagnostic_policy={"known_device_names": list(DIAGNOSTIC_DEVICE_NAMES),
                                      "memory_range_bytes_inclusive": list(DIAGNOSTIC_MEMORY_RANGE_BYTES),
                                      "compute_capability": [8, 0], "torch_cuda": "13.0", "device_count": 1})
    report.update({name: runtime[name] for name in fields})
    # True establishes only the baseline hardware identity. It never establishes
    # software/config/experiment comparability or authorizes a speedup claim.
    report["performance_comparable_to_canonical_sxm4"] = (
        all(policies["canonical_sxm4"].values()) and runtime["checks"]["compute_capability_8_0"]
        and runtime["checks"]["memory_80gb_class"])
    if runtime["device_name"] == "NVIDIA A100 80GB PCIe":
        report["performance_comparability_note"] = (
            "PCIe-local A/B evidence only; no speedup versus historical SXM4, "
            "canonical optimizer-step speedup, or final canonical hardware performance claim")
    return runtime


def run_diagnostic(report, check_features, *, compile_mode_ab=False):
    import json
    import torch
    from specforge.algorithms.eagle3.providers import create_registration
    from specforge.training.assembly import build_model_bundle
    from specforge.training.checkpoint import CheckpointManager

    runtime = record_diagnostic_hardware(report)
    if not all(runtime["checks"].values()):
        observed = json.dumps(runtime, sort_keys=True)
        print(json.dumps({"runtime_guard": runtime}, sort_keys=True), flush=True)
        raise ValueError("Requires a single known A100 80GB / CC 8.0 / CUDA 13.0 diagnostic runtime"
                         f"; runtime_guard={observed}")
    report["gpu"] = runtime["device_name"]
    report["tolerance"] = {"status": "FROZEN", "bfloat16": {"rtol": .016, "atol": 1e-5},
                           "float32": {"rtol": 1.3e-6, "atol": 1e-5}, "allowed_mismatches": 0,
                           "reference": "A", "comparison_dtype": "torch.float32"}
    report["experiment"] = "TTT0 independent CPU FP32 reference with FP64 precision audit; no backward or timing"
    cfg, tensors, identity = load_inputs(check_features)
    identity["sample_ids"] = [s["sample_id"] for s in identity["samples"]]
    report["selection"] = identity
    if identity["q_len"] != 120:
        raise ValueError("Root-cause replay requires the existing q_len=120 microbatch")
    checkpoint_cfg = copy.deepcopy(identity["config"])
    checkpoint = Path(cfg.output_dir) / (cfg.run_id + "-step3000")
    checkpoint_cfg["training"]["resume_from"] = str(checkpoint)
    # Reuse native full-state validation, but never instantiate a resuming trainer.
    validate_resume_checkpoint(checkpoint_cfg)
    checkpoint = Path(checkpoint_cfg["training"]["resume_from"])
    before = checkpoint_inventory(checkpoint)
    report["checkpoint"] = {"path": str(checkpoint), "before": before}
    try:
        state = CheckpointManager.read_resume_state(str(checkpoint), map_location="cpu", require_full_state=True)
        for name, value in {"sampler_seed": cfg.training.seed, "batch_size": 2,
                            "accumulation_steps": 4, "tp_size": 1,
                            "sp_ulysses_size": 1, "sp_ring_size": 1}.items():
            if state.get(name) != value:
                raise ValueError(f"Checkpoint training contract mismatch: {name}")
        algorithm = create_registration()
        bundle = build_model_bundle(cfg, algorithm=algorithm)
        draft = bundle.draft_model
        if (draft.config.num_attention_heads, draft.config.num_key_value_heads,
                draft.midlayer.self_attn.head_dim, draft.config.num_hidden_layers) != (28, 4, 128, 1):
            raise ValueError("Unexpected EAGLE3 attention architecture")
        for name in ("fc_norm", "norm_output", "use_post_norm_feedback"):
            if getattr(draft.config, name, None) is not True:
                raise ValueError(f"Missing EAGLE-3.1 invariant: {name}")
        for name, value in algorithm.providers.step.resume_contract(cfg, draft, bundle.model).items():
            if state.get(name) != value:
                raise ValueError(f"Checkpoint objective mismatch: {name}")
        loaded = draft.load_state_dict(state["draft_state_dict"], strict=False)
        allowed = algorithm.providers.step.allowed_missing_checkpoint_keys(cfg, draft, bundle.model)
        if loaded.unexpected_keys or set(loaded.missing_keys) - allowed or not state["draft_state_dict"]:
            raise ValueError("Checkpoint draft weight mismatch")
        # Mapping is persisted in checkpoint buffers: authenticate it again after loading.
        mapping = torch.load(cfg.model.vocab_mapping_path, weights_only=True, map_location="cpu")
        if any(not torch.equal(getattr(draft, key).cpu(), mapping[key]) for key in ("t2d", "d2t")):
            raise ValueError("Checkpoint vocabulary mapping differs from sealed TRAIN mapping")
        rng = state["backend"]["rng"]
        del state  # Discard optimizer payload; no optimizer object is ever built/restored.
        bundle.model.train()
        strategy = algorithm.providers.step.build(bundle.model, target_head=bundle.target_head,
                                                  **algorithm.providers.step.options(cfg))
        if compile_mode_ab:
            report["experiment"] = "TTT0 compiled-default vs compiled-max-autotune; independent FP32/FP64 reference; forward timing only"
        run_forward_isolation(bundle, strategy, tensors, rng, report, reference_ttt0=True,
                              compile_mode_ab=compile_mode_ab)
    finally:
        after = checkpoint_inventory(checkpoint)
        report["checkpoint"].update(after=after, unchanged=before == after)
        if before != after:
            raise RuntimeError("Checkpoint or latest/best evidence changed during diagnostic")
