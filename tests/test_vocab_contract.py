"""CPU/meta regressions against read-only excerpts of the exact pinned upstream."""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from training.train import SPECFORGE_SHA, feature_paths, load_training_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures/specforge_vocab_contract.py.txt"


@pytest.fixture
def upstream(monkeypatch):
    source = FIXTURE.read_text(encoding="utf-8")
    assert "Revision: " + SPECFORGE_SHA in source

    class Base(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
        def post_init(self):
            pass  # Weight initialization is irrelevant on meta.

    namespace = {
        "torch": torch, "nn": torch.nn, "LlamaConfig": LlamaConfig,
        "Eagle3DraftModel": Base,
        # Only the heavy decoder/RMS implementation is stubbed, never lm_head.
        "LlamaDecoderLayer": lambda *args, **kwargs: torch.nn.Identity(),
        "LlamaRMSNorm": lambda size, eps: torch.nn.LayerNorm(size, eps=eps),
        "os": os, "json": json, "hashlib": hashlib, "Counter": Counter,
        "FeatureMode": SimpleNamespace(OFFLINE="offline"),
    }
    exec(compile(source, str(FIXTURE), "exec"), namespace)

    def module(name, **attrs):
        result = ModuleType(name)
        result.__path__ = []
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:
            monkeypatch.setattr(sys.modules[parent], child, result, raising=False)
        return result

    for name in ("specforge", "specforge.modeling", "specforge.modeling.draft",
                 "specforge.data", "specforge.training", "specforge.runtime",
                 "specforge.runtime.data_plane"):
        module(name)
    model_class = namespace["LlamaForCausalLMEagle3"]
    module("specforge.modeling.draft.registry",
           DRAFT_REGISTRY={"LlamaForCausalLMEagle3": model_class},
           available_drafts=lambda: ["LlamaForCausalLMEagle3"])
    module("specforge.data.preprocessing",
           process_token_dict_to_mappings=namespace["process_token_dict_to_mappings"])
    module("specforge.training.vocab_mapping",
           count_effective_feature_tokens=namespace["count_effective_feature_tokens"])
    namespace["module"] = module
    return namespace


def test_pinned_resolver_and_real_meta_lm_head_output_16000(upstream):
    payload = json.loads((ROOT / "configs/eagle3.1_qwen2.5_7b.json").read_text())
    cfg = upstream["_draft_config_from_dict"](payload)
    assert cfg.vocab_size == 152064
    assert cfg.draft_vocab_size == 16000
    with torch.device("meta"):
        draft = upstream["LlamaForCausalLMEagle3"](cfg)
        assert isinstance(draft.lm_head, torch.nn.Linear)
        assert draft.lm_head.out_features == 16000
        assert tuple(draft.lm_head.weight.shape) == (16000, 3584)
        assert draft.embed_tokens.num_embeddings == 152064
        assert tuple(draft.compute_logits(torch.empty(1, 2, 3584)).shape) == (1, 2, 16000)
    assert not torch.cuda.is_initialized()


def test_old_full_vocab_identity_cannot_satisfy_new_smoke_contract(tmp_path):
    cfg = load_training_config()
    current = ROOT / "configs/eagle3.1_qwen2.5_7b.json"
    new = tmp_path / "new-draft.json"
    old = tmp_path / "old-draft.json"
    content = json.loads(current.read_text())
    new.write_text(json.dumps(content, sort_keys=True))
    content["draft_vocab_size"] = content["vocab_size"]  # Reproduce the rejected legacy contract.
    old.write_text(json.dumps(content, sort_keys=True))
    options = {"feature_scope": "smoke", "target_revision": "same-target-revision",
               "capture_layers": [1, 13, 24]}
    new_config, new_key, new_contract = feature_paths(cfg, new, **options)
    old_config, old_key, old_contract = feature_paths(cfg, old, **options)
    assert new_contract["draft_sha256"] != old_contract["draft_sha256"]
    assert new_key != old_key
    assert new_config["data"]["hidden_states_path"] != old_config["data"]["hidden_states_path"]
    full_config, _, _ = feature_paths(cfg, current)
    assert "/smoke/" in new_config["data"]["hidden_states_path"]
    assert "/smoke/" not in full_config["data"]["hidden_states_path"]
    assert current.read_bytes() != old.read_bytes()


def test_native_offline_mapping_is_16000_and_derived_from_train_only(upstream, tmp_path):
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    train_dir.mkdir()
    eval_dir.mkdir()
    train_file = train_dir / "0.ckpt"
    eval_file = eval_dir / "0.ckpt"
    torch.save({"input_ids": torch.tensor([152000, 152001, 42]),
                "loss_mask": torch.tensor([1, 0, 1])}, train_file)
    torch.save({"input_ids": torch.tensor([152050]),
                "loss_mask": torch.tensor([1])}, eval_file)
    paths_seen = []
    def list_features(path):
        paths_seen.append(str(path))
        assert str(path) == str(train_dir), "Eval must never select vocabulary"
        return [str(train_file)]
    upstream["module"]("specforge.runtime.data_plane.offline_reader", list_feature_files=list_features)
    upstream["module"]("specforge.runtime.data_plane.feature_store",
                       load_feature_file=lambda path: torch.load(path, map_location="cpu", weights_only=True))

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("t2d", torch.zeros(152064, dtype=torch.bool))
            self.register_buffer("d2t", torch.zeros(16000, dtype=torch.int64))
            self.vocab_mapping_loaded = False
        def load_vocab_mapping(self, path):
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.t2d.copy_(state["t2d"])
            self.d2t.copy_(state["d2t"])
            self.vocab_mapping_loaded = True

    cfg = SimpleNamespace(
        model=SimpleNamespace(vocab_mapping_path=""),
        data=SimpleNamespace(hidden_states_path=str(train_dir),
                             eval_hidden_states_path=str(eval_dir),
                             cache_dir=str(tmp_path / "cache"), max_length=2048),
    )
    draft = Draft()
    bundle = SimpleNamespace(draft_model=draft, target_vocab_size=152064, draft_vocab_size=16000)
    algorithm = SimpleNamespace(providers=SimpleNamespace(vocab_mapping_modes={"offline"}))
    upstream["_ensure_offline_vocab_mapping"](cfg, bundle, algorithm)
    assert draft.vocab_mapping_loaded
    assert draft.t2d.shape == (152064,)
    assert draft.t2d.sum().item() == 16000
    assert draft.d2t.shape == (16000,)
    assert torch.equal(torch.nonzero(draft.t2d).flatten(), draft.d2t + torch.arange(16000))
    assert draft.t2d[152000]  # Supervised train token included.
    assert not draft.t2d[152001]  # Masked train token not selected.
    assert not draft.t2d[152050]  # Eval-only token not selected.
    assert set(paths_seen) == {str(train_dir)}
    cached = list((tmp_path / "cache/vocab_mapping").glob("*.pt"))
    assert len(cached) == 1
    saved = torch.load(cached[0], map_location="cpu", weights_only=True)
    assert saved["t2d"].shape == (152064,) and saved["d2t"].shape == (16000,)


def test_native_mapping_selection_is_deterministic(upstream):
    counts = Counter({6: 10, 7: 9, 2: 3})
    first = upstream["process_token_dict_to_mappings"](counts.copy(), 4, 8)
    second = upstream["process_token_dict_to_mappings"](counts.copy(), 4, 8)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
