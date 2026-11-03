"""Discriminating CPU tests; no production loader or experiment changes."""
from collections import Counter
import gzip
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.test_vocab_contract import upstream  # Reuse exact pinned assembly excerpts.
from training.train import SPECFORGE_SHA
from training.native_vocab_mapping import cache_record_for_feature, cache_sample_index, count_cache_tokens


@pytest.fixture
def offline(upstream):
    path = Path(__file__).parent / "fixtures/specforge_offline_startup.py.txt"
    source = path.read_text(encoding="utf-8")
    assert "Revision: " + SPECFORGE_SHA in source
    namespace = {
        "os": os, "torch": torch, "gzip": gzip, "io": io, "Counter": Counter,
        "tqdm": lambda values, **kwargs: values,
        "process_token_dict_to_mappings": upstream["process_token_dict_to_mappings"],
        "_FEATURE_SUFFIXES": (".ckpt", ".ckpt.gz"),
        "_OFFLINE_EAGLE3_KEYS": ("input_ids", "loss_mask", "hidden_state", "aux_hidden_state"),
        "_VALIDATE_ENV": "SPECFORGE_VALIDATE_OFFLINE_FEATURES",
        "SampleRef": SimpleNamespace, "SCHEMA_VERSION": 1,
    }
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


def mapping_case(upstream, offline, tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    for index, token in enumerate((7, 2, 6)):
        torch.save({"input_ids": torch.tensor([token]), "loss_mask": torch.tensor([1])},
                   train / f"{index}.ckpt")
    loads = []
    def load(path):
        loads.append(path)
        return offline["load_feature_file"](path)
    upstream["module"]("specforge.runtime.data_plane.offline_reader",
                       list_feature_files=offline["list_feature_files"])
    upstream["module"]("specforge.runtime.data_plane.feature_store", load_feature_file=load)
    cfg = SimpleNamespace(model=SimpleNamespace(vocab_mapping_path=""),
                          data=SimpleNamespace(hidden_states_path=str(train),
                                               cache_dir=str(tmp_path / "cache"), max_length=2048))
    class Draft:
        def __init__(self):
            self.t2d = torch.zeros(8, dtype=torch.bool)
            self.d2t = torch.zeros(2, dtype=torch.int64)
        def load_vocab_mapping(self, path):
            state = torch.load(path, weights_only=True)
            self.t2d.copy_(state["t2d"])
            self.d2t.copy_(state["d2t"])
    draft = Draft()
    bundle = SimpleNamespace(draft_model=draft, target_vocab_size=8, draft_vocab_size=2)
    algorithm = SimpleNamespace(providers=SimpleNamespace(vocab_mapping_modes={"offline"}))
    return cfg, bundle, algorithm, loads


def test_native_cache_hit_still_opens_every_train_record(upstream, offline, tmp_path):
    cfg, bundle, algorithm, loads = mapping_case(upstream, offline, tmp_path)
    # Preparation's fixed-name map exists, but no configured mapping path points at it.
    prepared = Path(cfg.data.hidden_states_path) / "vocab_mapping/vocab_mapping.pt"
    prepared.parent.mkdir()
    torch.save({"t2d": torch.zeros(8, dtype=torch.bool), "d2t": torch.zeros(2)}, prepared)
    ensure = upstream["_ensure_offline_vocab_mapping"]
    ensure(cfg, bundle, algorithm)
    expected = offline["list_feature_files"](cfg.data.hidden_states_path)
    assert loads == expected
    assert len(list((tmp_path / "cache/vocab_mapping").glob("*.pt"))) == 1
    original = bundle.draft_model.t2d.clone()
    loads.clear()
    ensure(cfg, bundle, algorithm)
    assert loads == expected  # The cache lookup happens AFTER counting, even on retry.
    assert torch.equal(original, bundle.draft_model.t2d)


def test_explicit_mapping_skips_scan_before_loader_assembly(upstream, offline, tmp_path):
    cfg, bundle, algorithm, loads = mapping_case(upstream, offline, tmp_path)
    cfg.model.vocab_mapping_path = str(Path(cfg.data.hidden_states_path) / "vocab_mapping/vocab_mapping.pt")
    def forbidden(*args, **kwargs):
        pytest.fail("Explicit map must return before listing/stat/counting train features")
    upstream["module"]("specforge.runtime.data_plane.offline_reader", list_feature_files=forbidden)
    upstream["module"]("specforge.training.vocab_mapping", count_effective_feature_tokens=forbidden)
    upstream["_ensure_offline_vocab_mapping"](cfg, bundle, algorithm)
    assert loads == []
    # Loading/validating that explicit file belongs to the earlier model-provider path.


def test_mapping_counter_does_not_access_hidden_tensor_values(upstream):
    raw = {"input_ids": torch.tensor([7, 2, 6]), "loss_mask": torch.tensor([1, 0, 1]),
           "hidden_state": object(), "aux_hidden_state": object()}
    upstream["module"]("specforge.runtime.data_plane.offline_reader", list_feature_files=lambda path: ["one.ckpt"])
    upstream["module"]("specforge.runtime.data_plane.feature_store", load_feature_file=lambda path: raw)
    assert upstream["count_effective_feature_tokens"]("train", max_length=2, target_vocab_size=8) == Counter({7: 1})


def test_uncompressed_feature_open_requests_mmap(offline, monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: calls.append((args, kwargs)))
    offline["load_feature_file"]("one.ckpt")
    assert calls == [(("one.ckpt",), {"weights_only": False, "mmap": True})]


def test_default_reader_lists_all_files_without_loading_tensors(offline, tmp_path, monkeypatch):
    folder = tmp_path / "rows_0-2000"
    folder.mkdir()
    for name in ("data_2.ckpt", "data_10.ckpt", "data_1.ckpt.gz", "not-a-feature.pt"):
        (folder / name).touch()
    monkeypatch.delenv("SPECFORGE_VALIDATE_OFFLINE_FEATURES", raising=False)
    offline["load_feature_file"] = lambda path: pytest.fail("Default reader must not open feature tensors")
    refs = offline["OfflineManifestReader"](str(tmp_path)).read()
    assert [Path(ref.feature_store_uri).name for ref in refs] == ["data_1.ckpt.gz", "data_10.ckpt", "data_2.ckpt"]
    assert all(ref.feature_specs == {} for ref in refs)


def test_preparation_map_cannot_be_assumed_equal_when_ties_cross_cutoff(upstream, offline, tmp_path):
    # Same supervised token multiset, different first-seen order at a frequency tie.
    dataset = {"input_ids": [torch.tensor([7]), torch.tensor([2])],
               "loss_mask": [torch.tensor([1]), torch.tensor([1])]}
    class Dataset(dict):
        def __len__(self):
            return 2
    path = offline["generate_vocab_mapping_file"](Dataset(dataset), 8, 1, str(tmp_path / "prep"))
    prepared = torch.load(path, weights_only=True)
    features = {"rows_0-2000/data_10.ckpt": {"input_ids": torch.tensor([2]), "loss_mask": torch.tensor([1])},
                "rows_0-2000/data_2.ckpt": {"input_ids": torch.tensor([7]), "loss_mask": torch.tensor([1])}}
    upstream["module"]("specforge.runtime.data_plane.offline_reader", list_feature_files=lambda path: sorted(features))
    upstream["module"]("specforge.runtime.data_plane.feature_store", load_feature_file=features.__getitem__)
    counts = upstream["count_effective_feature_tokens"]("train", max_length=2048, target_vocab_size=8)
    _, trained_t2d = upstream["process_token_dict_to_mappings"](counts, 1, 8)
    assert prepared["t2d"][7] and trained_t2d[2]
    assert not torch.equal(prepared["t2d"], trained_t2d)


def test_worker_resolution_preserves_provider_default_and_explicit_zero(offline):
    algorithm = SimpleNamespace(providers=SimpleNamespace(model=SimpleNamespace(default_dataloader_num_workers=4)))
    cfg = SimpleNamespace(data=SimpleNamespace(dataloader_num_workers=None))
    assert offline["_dataloader_num_workers"](cfg, algorithm) == 4
    cfg.data.dataloader_num_workers = 0
    assert offline["_dataloader_num_workers"](cfg, algorithm) == 0


def test_cache_reconstruction_matches_native_with_order_sensitive_cutoff(upstream, offline, tmp_path):
    # Physical cache order is numeric sample order. Native path order visits
    # data_10 before data_2. Mask 2 must count; positions after max_length must not.
    rows = [{"input_ids": [[0]], "loss_mask": [[0]]} for _ in range(12)]
    rows[2] = {"input_ids": [[7]], "loss_mask": [[1]]}
    rows[10] = {"input_ids": [[2, 9]], "loss_mask": [[1, 2]]}
    rows[11] = {"input_ids": [[11, 12, 13, 14]], "loss_mask": [[0, 0, 1, 1]]}
    directory = tmp_path / "rows_0-2000"
    directory.mkdir()
    # Independently build the feature side using the pinned capture collator's
    # zero-pad-to-longest rule, rather than invoking the candidate helper.
    for start in range(0, 12, 2):
        width = max(len(rows[i]["input_ids"][0]) for i in (start, start + 1))
        for i in (start, start + 1):
            raw = {}
            for field in ("input_ids", "loss_mask"):
                values = rows[i][field][0]
                raw[field] = torch.tensor(values + [0] * (width - len(values)))
            torch.save(raw, directory / f"data_{i}.ckpt")
    native_order = offline["list_feature_files"](str(tmp_path))
    assert [Path(p).name for p in native_order[:5]] == [
        "data_0.ckpt", "data_1.ckpt", "data_10.ckpt", "data_11.ckpt", "data_2.ckpt"]
    upstream["module"]("specforge.runtime.data_plane.offline_reader",
                       list_feature_files=offline["list_feature_files"])
    feature_module = upstream["module"]("specforge.runtime.data_plane.feature_store",
                                        load_feature_file=offline["load_feature_file"])
    count = upstream["count_effective_feature_tokens"]
    native_counts = count(str(tmp_path), max_length=2, target_vocab_size=16)
    assert list(native_counts.items()) == [(2, 1), (9, 1), (7, 1)]
    selector = upstream["process_token_dict_to_mappings"]
    native_d2t, native_t2d = selector(native_counts.copy(), 2, 16)

    def reconstructed(path):
        actual = offline["load_feature_file"](path)
        candidate = cache_record_for_feature(path, rows.__getitem__, len(rows))
        assert all(torch.equal(candidate[k], actual[k]) for k in actual)
        return candidate
    feature_module.load_feature_file = reconstructed
    candidate_counts = count(str(tmp_path), max_length=2, target_vocab_size=16)
    assert list(candidate_counts.items()) == list(native_counts.items())
    production_counts = count_cache_tokens(native_order, rows.__getitem__, len(rows),
                                          max_length=2, target_vocab_size=16)
    assert list(production_counts.items()) == list(native_counts.items())
    candidate_d2t, candidate_t2d = selector(candidate_counts.copy(), 2, 16)
    assert torch.equal(candidate_t2d, native_t2d)
    assert torch.equal(candidate_d2t, native_d2t)
    assert torch.nonzero(candidate_t2d).flatten().tolist() == [2, 9]
    assert candidate_d2t.tolist() == [2, 8]

    # A physical-row permutation must fail the row/file correspondence gate.
    rows[2], rows[10] = rows[10], rows[2]
    with pytest.raises(AssertionError):
        reconstructed(str(directory / "data_2.ckpt"))
    rows[2], rows[10] = rows[10], rows[2]

    # Numeric order and frequency-then-token-ID tie breaking are both rejected.
    feature_module.load_feature_file = lambda path: cache_record_for_feature(path, rows.__getitem__, len(rows))
    upstream["module"]("specforge.runtime.data_plane.offline_reader",
                       list_feature_files=lambda path: sorted(native_order, key=lambda p: cache_sample_index(p, 12)))
    wrong_counts = count(str(tmp_path), max_length=2, target_vocab_size=16)
    wrong_d2t, wrong_t2d = selector(wrong_counts.copy(), 2, 16)
    assert list(wrong_counts.items()) == [(7, 1), (2, 1), (9, 1)]
    assert not torch.equal(wrong_t2d, native_t2d)
    assert not torch.equal(wrong_d2t, native_d2t)
    assert [token for token, _ in sorted(native_counts.items(), key=lambda item: (-item[1], item[0]))[:2]] == [2, 7]


@pytest.mark.parametrize("token", [-1, 16])
def test_cache_counter_retains_native_target_vocab_validation(upstream, token):
    rows = [{"input_ids": [[token]], "loss_mask": [[2]]}]
    path = "/train/rows_0-2000/data_0.ckpt"
    upstream["module"]("specforge.runtime.data_plane.offline_reader", list_feature_files=lambda root: [path])
    upstream["module"]("specforge.runtime.data_plane.feature_store",
                       load_feature_file=lambda p: cache_record_for_feature(p, rows.__getitem__, 1))
    with pytest.raises(ValueError):
        upstream["count_effective_feature_tokens"]("/train", max_length=2048, target_vocab_size=16)
    with pytest.raises(ValueError):
        count_cache_tokens([path], rows.__getitem__, 1, max_length=2048, target_vocab_size=16)


@pytest.mark.parametrize("path", ["/train/rows_2000-4000/data_0.ckpt", "/train/rows_0-2000/data_12.ckpt"])
def test_cache_identity_rejects_wrong_group_or_out_of_range(path):
    with pytest.raises(ValueError):
        cache_sample_index(path, 12)
