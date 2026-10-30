"""Opt-in TTT0 compile-mode experiment; no training or persistence entrypoint."""
import os
import statistics
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import FunctionType
from unittest.mock import patch

from training.flex_attention_ab import tensor_identity, ttt0_reference


def compile_candidates(flex, compile_fn):
    """Distinct code objects prevent Dynamo's code-keyed cache sharing.

    A matches SpecForge fddcb4ad WrappedFlexAttention exactly (no kwargs).
    B follows the requested upstream recipe; dynamic=True is explicit only in B.
    No SpecForge singleton is read, reset or modified.
    """
    candidates = {}
    for name, options in (("default", {}),
                          ("max_autotune", {"dynamic": True, "mode": "max-autotune"})):
        code = flex.__code__.replace(co_name=f"diagnostic_flex_{name}")
        fn = FunctionType(code, flex.__globals__, code.co_name, flex.__defaults__, flex.__closure__)
        fn.__kwdefaults__ = flex.__kwdefaults__
        candidates[name] = compile_fn(fn, **options)
    return candidates


def numerical_classification(reference, *, inputs_identical):
    """Reuse frozen reference sensitivity ranking; speed never enters this rule."""
    if not inputs_identical:
        return "NUMERICALLY INCONCLUSIVE"
    ranking = reference["classification"]
    if ranking == "RAW CLOSER TO REFERENCE":
        return "MAX-AUTOTUNE NUMERICALLY WORSE"
    if ranking == "COMPILED CLOSER TO REFERENCE":
        a, b = reference["raw_vs_reference"], reference["compiled_vs_reference"]
        sensitivity = reference["reference_precision_envelope"]["l2"]
        if (b["max_abs"] > a["max_abs"] + 2 * sensitivity or
                b["fp32_screening_only"]["mismatched_elements"] >
                a["fp32_screening_only"]["mismatched_elements"]):
            return "NUMERICALLY INCONCLUSIVE"
        return "MAX-AUTOTUNE BETTER OR EQUIVALENT"
    # Only exact finite output equality establishes equivalence; overlapping or
    # conflicting sensitivity intervals alone must remain inconclusive.
    direct = reference.get("direct_a_vs_b", {})
    if (ranking == "BOTH COMPARABLY CLOSE" and direct.get("finite")
            and direct.get("max_abs") == 0):
        return "MAX-AUTOTUNE BETTER OR EQUIVALENT"
    return "NUMERICALLY INCONCLUSIVE"


@contextmanager
def candidate_cache(directory):
    # Separate disk/autotuning namespaces, restored after each invocation. This
    # single-threaded diagnostic never clears process-wide compiler caches.
    with patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": str(directory),
                               "TRITON_CACHE_DIR": str(Path(directory) / "triton")}):
        yield


def timed_call(fn, synchronize):
    synchronize()
    start = time.perf_counter()
    output = fn()
    synchronize()
    return output, (time.perf_counter() - start) * 1000


def timing_summary(samples):
    return {"unit": "ms", "samples": samples, "mean": statistics.mean(samples),
            "median": statistics.median(samples), "min": min(samples), "max": max(samples)}


def run_compile_mode_ab(inputs, block_mask, create_mask, report):
    """Called once inside the authenticated TTT0 boundary, before projection.

    Both candidates receive the SAME tensor and BlockMask objects. Authenticate
    values/metadata before and after every call, including timed calls. CPU
    authentication and transfers stay outside the measured interval.
    """
    import torch
    from torch.nn.attention.flex_attention import flex_attention

    evidence = report["compile_mode_ab"] = {
        "status": "RUNNING", "warmup_count": 2, "repeat_count": 5,
        "backward_executed": False, "timing_scope": "bounded TTT0 forward only",
        "hardware": {key: report.get(key) for key in (
            "device_name", "total_memory_bytes", "compute_capability", "torch_cuda",
            "device_count", "performance_comparable_to_canonical_sxm4")},
        "performance_scope": "same-device default vs max-autotune only; no historical or optimizer-step comparison",
        "compile_contract": {"default": {}, "max_autotune": {"dynamic": True, "mode": "max-autotune"}},
        "cache_isolation": "distinct Dynamo code objects and per-candidate Inductor/Triton disk directories",
    }
    snapshots = {n: t.detach().clone().requires_grad_(t.requires_grad) for n, t in inputs.items()}
    b, h, q, _ = inputs["query"].shape
    kv = inputs["key"].shape[-2]
    logical = create_mask(block_mask.mask_mod, b, h, q, kv, inputs["query"].device).clone()
    fields = ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
              "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices",
              "dq_write_order", "dq_write_order_full", "dq_kv_order")
    blocks = {n: getattr(block_mask, n).clone() if getattr(block_mask, n) is not None else None
              for n in fields}
    metadata = {n: getattr(block_mask, n) for n in ("seq_lengths", "BLOCK_SIZE", "dq_kv_order_spt")}

    def authenticate():
        identities = {n: tensor_identity(snapshots[n], t) for n, t in inputs.items()}
        mask_equal = all((getattr(block_mask, n) is None if saved is None else
                          torch.equal(saved, getattr(block_mask, n))) for n, saved in blocks.items())
        mask_equal = mask_equal and all(getattr(block_mask, n) == v for n, v in metadata.items())
        mask_equal = mask_equal and torch.equal(logical, create_mask(
            block_mask.mask_mod, b, h, q, kv, inputs["query"].device))
        if not mask_equal or not all(v["identical"] and v["A"]["finite"] for v in identities.values()):
            raise ValueError("Compile-mode A/B input or mask mutation; timing/attribution invalid")
        return identities

    evidence["inputs"] = authenticate()
    evidence["mask_authentication"] = {
        "logical": tensor_identity(logical, logical),
        "block_tensors": {n: tensor_identity(t, t) if t is not None else None for n, t in blocks.items()},
        "metadata": metadata, "checked_before_and_after_every_invocation": True}
    evidence["same_tensor_and_block_mask_objects"] = True
    outputs = {}
    with tempfile.TemporaryDirectory(prefix="flex-compile-ab-") as cache_root:
        candidates = compile_candidates(flex_attention, torch.compile)

        def invoke(name):
            authenticate()
            with candidate_cache(Path(cache_root) / name):
                output, milliseconds = timed_call(
                    lambda: candidates[name](**inputs, block_mask=block_mask, enable_gqa=True),
                    torch.cuda.synchronize)
            authenticate()
            return output, milliseconds

        for name in candidates:
            evidence[name] = {"compile_mode": name.replace("_", "-"),
                              "compile_options": evidence["compile_contract"][name]}
            output, elapsed = invoke(name)
            if (output.shape != inputs["query"].shape or output.dtype != inputs["query"].dtype
                    or output.device != inputs["query"].device):
                raise ValueError("Compile-mode output shape/dtype/device mismatch; timing skipped")
            outputs[name] = output.detach().cpu().clone()
            del output
            evidence[name]["cold_first_call"] = {"unit": "ms", "latency": elapsed,
                "includes": "compile/autotune/execution; process-local first invocation in isolated disk cache"}
        # Existing reference and frozen ranking/gate are reused without edits.
        reference = ttt0_reference(snapshots, block_mask, block_mask,
                                  outputs["default"], outputs["max_autotune"],
                                  inputs["query"].device, create_mask)
        evidence["reference"] = reference
        for name, label in (("default", "raw"), ("max_autotune", "compiled")):
            evidence[name]["reference_metrics"] = reference.get(f"{label}_vs_reference")
            evidence[name]["output"] = {"shape": list(outputs[name].shape), "dtype": str(outputs[name].dtype),
                                        "finite": bool(torch.isfinite(outputs[name]).all())}
        evidence["numerical_classification"] = numerical_classification(reference, inputs_identical=True)
        evidence["ratios"] = {f"{key}_max_autotune_over_default": value
                               for key, value in ((k.removesuffix("_B_over_A"), v)
                                                  for k, v in reference.get("ratios", {}).items())}
        if reference["classification"] == "REFERENCE INVALID / SEMANTICS MISMATCH":
            raise ValueError("Invalid compile-mode reference; steady-state timing skipped")
        # Correctness for BOTH candidates is now collected. Warm-up and timing
        # cannot affect the numerical acceptance rule or reference metrics.
        for name in candidates:
            for _ in range(evidence["warmup_count"]):
                invoke(name)
            samples = [invoke(name)[1] for _ in range(evidence["repeat_count"])]
            evidence[name]["steady_state"] = timing_summary(samples)
    a, b = (evidence[n]["steady_state"]["mean"] for n in ("default", "max_autotune"))
    evidence["steady_state_speedup"] = a / b if b else None
    evidence["steady_state_gain_clear"] = (
        evidence["max_autotune"]["steady_state"]["max"] < evidence["default"]["steady_state"]["min"])
    evidence["gain_rule"] = "All five max-autotune samples faster than every default sample; conservative local screening only"
    overhead = (evidence["max_autotune"]["cold_first_call"]["latency"] - b
                - evidence["default"]["cold_first_call"]["latency"] + a)
    evidence["break_even_estimate"] = {
        "invocations": max(0., overhead) / (a - b) if a > b else None,
        "compile_overhead_delta_ms": overhead, "saved_per_call_ms": a - b,
        "scope": "estimate only for this kernel replay; no full-training inference"}
    evidence["next_action"] = ("STOP this optimization branch" if not evidence["steady_state_gain_clear"] or
        evidence["numerical_classification"] != "MAX-AUTOTUNE BETTER OR EQUIVALENT" else
        "Propose a bounded optimizer-step benchmark for review; do not execute it automatically")
    evidence["status"] = "EXECUTED"
