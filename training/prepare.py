"""Canonical feature extraction with a project-local preformatted loader bridge.

Only the module-local JSONL loader binding is replaced. Upstream source files,
tokenization, shuffling, capture, loss masks and feature writing remain upstream.
"""
import argparse
import importlib.util
import sys
from data.prepare_specforge import preformatted_rows
from training.train import SPECFORGE_ROOT, load_training_config, verify_runtime


def canonical_preparation(argv, script=None):
    script = script or SPECFORGE_ROOT / "scripts/prepare_hidden_states.py"
    spec = importlib.util.spec_from_file_location("_specforge_project_preparation", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    old_argv = sys.argv
    try:
        spec.loader.exec_module(module)
        module.safe_conversations_generator = preformatted_rows
        sys.argv = [str(script)] + list(argv)
        module.main()
    finally:
        sys.argv = old_argv
        sys.modules.pop(spec.name, None)


def preparation_args(cfg, source, output, num_samples=None):
    args = [
        "--strategy", cfg["training"]["strategy"],
        "--target-model-path", cfg["model"]["target_model_path"],
        "--draft-model-config", cfg["model"]["draft_model_config"],
        "--data-path", str(source), "--output-path", str(output),
        "--chat-template", cfg["data"]["chat_template"],
        "--max-length", str(cfg["data"]["max_length"]),
        "--cache-dir", cfg["data"]["cache_dir"],
        "--is-preformatted", "--tp-size", "1", "--batch-size", "2",
    ]
    if num_samples is not None:
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        args.extend(["--num-samples", str(num_samples)])
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int)
    args = parser.parse_args()
    verify_runtime(check_toolchain=True)
    cfg = load_training_config(args.config)
    canonical_preparation(preparation_args(cfg, args.source, args.output, args.num_samples))


if __name__ == "__main__":
    main()
