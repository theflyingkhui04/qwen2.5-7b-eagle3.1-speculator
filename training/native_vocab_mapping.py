"""Authenticated native-equivalent TRAIN mapping; no feature tensor reads.

Full artifacts require an accepted cache-to-feature proof profile. Smoke keeps
the existing small native scan and cannot consume an explicit full mapping.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import uuid

from training.train import REPO_ROOT, SPECFORGE_SHA, feature_counts

PROFILE_PATH = REPO_ROOT / "configs/native_vocab_mapping_proof.json"
VOLUME_ROOT = Path("/vol")


def cache_sample_index(feature_path, num_rows):
    """Inverse of the pinned capture writer's per-sample path."""
    match = re.search(r"(?:^|/)rows_(\d+)-(\d+)/data_(\d+)\.ckpt$", str(feature_path).replace("\\", "/"))
    if not match:
        raise ValueError("Invalid canonical feature identity")
    lower, upper, index = map(int, match.groups())
    if not 0 <= index < num_rows or lower != index // 2000 * 2000 or upper != lower + 2000:
        raise ValueError("Feature identity is outside its canonical row group")
    return index


def cache_record_for_feature(feature_path, get_row, num_rows):
    """Reconstruct ids/masks including the pinned capture batch-2 padding."""
    import torch
    index = cache_sample_index(feature_path, num_rows)
    row = get_row(index)
    ids = torch.as_tensor(row["input_ids"], dtype=torch.int64).reshape(-1)
    mask = torch.as_tensor(row["loss_mask"], dtype=torch.int64).reshape(-1)
    if ids.numel() != mask.numel():
        raise ValueError("Cache ids/mask lengths differ")
    start = index // 2 * 2
    width = max(torch.as_tensor(get_row(i)["input_ids"]).numel()
                for i in range(start, min(start + 2, num_rows)))
    result = {"input_ids": torch.zeros(width, dtype=torch.int64),
              "loss_mask": torch.zeros(width, dtype=torch.int64)}
    result["input_ids"][:ids.numel()] = ids
    result["loss_mask"][:mask.numel()] = mask
    return result


def count_cache_tokens(paths, get_row, num_rows, *, max_length, target_vocab_size):
    """Pinned count_effective_feature_tokens semantics over authenticated rows.

    Keep per-file unique ordering and Counter insertion order: both affect ties.
    The paired regression executes the unchanged pinned counter independently.
    """
    counts = Counter()
    for path in paths:
        raw = cache_record_for_feature(path, get_row, num_rows)
        input_ids = raw["input_ids"].reshape(-1)
        loss_mask = raw["loss_mask"].reshape(-1)
        if input_ids.numel() != loss_mask.numel():
            raise ValueError(f"{path} has mismatched ids/mask lengths")
        input_ids, loss_mask = input_ids[:max_length], loss_mask[:max_length]
        selected = input_ids[loss_mask.to(dtype=bool)]
        token_ids, frequencies = selected.unique(return_counts=True)
        for token_id, frequency in zip(token_ids.tolist(), frequencies.tolist()):
            token_id = int(token_id)
            if token_id < 0 or token_id >= target_vocab_size:
                raise ValueError(f"{path} contains token id {token_id} outside target vocab")
            counts[token_id] += int(frequency)
    return counts


def tensor_hashes(mapping):
    return {key: hashlib.sha256(mapping[key].contiguous().numpy().astype(dtype).tobytes()).hexdigest()
            for key, dtype in (("t2d", "u1"), ("d2t", "<i8"))}


def validate_tensors(mapping, profile):
    import torch
    if not isinstance(mapping, dict) or set(mapping) != {"t2d", "d2t"}:
        raise ValueError("Malformed vocabulary mapping")
    for key, size, dtype in (("t2d", profile["target_vocab_size"], torch.bool),
                             ("d2t", profile["draft_vocab_size"], torch.int64)):
        value = mapping[key]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.shape != (size,) or value.dtype != dtype:
            raise ValueError("Vocabulary mapping shape/dtype mismatch")
    if not torch.equal(torch.nonzero(mapping["t2d"]).flatten(),
                       mapping["d2t"] + torch.arange(profile["draft_vocab_size"])):
        raise ValueError("Vocabulary mapping offsets disagree")
    if tensor_hashes(mapping) != profile["tensor_sha256"]:
        raise ValueError("Vocabulary mapping differs from accepted native proof")


def select_mapping(counts, profile):
    from specforge.data.preprocessing import process_token_dict_to_mappings
    d2t, t2d = process_token_dict_to_mappings(
        counts, profile["draft_vocab_size"], profile["target_vocab_size"])
    mapping = {"d2t": d2t, "t2d": t2d}
    validate_tensors(mapping, profile)
    return mapping


def _path(value, relative):
    expected = VOLUME_ROOT / relative
    if Path(value) != expected or expected.resolve() != VOLUME_ROOT.resolve() / relative:
        raise ValueError("Vocabulary mapping namespace/path mismatch")
    return expected


def _context(payload):
    cfg, contract = payload["config"], payload["contract"]
    if contract.get("feature_scope") == "smoke":
        if cfg["model"].get("vocab_mapping_path"):
            raise ValueError("Smoke cannot consume an explicit full vocabulary mapping")
        key = payload["artifact_key"]
        if hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20] != key:
            raise ValueError("Smoke artifact identity mismatch")
        for label, field in (("train", "hidden_states_path"), ("eval", "eval_hidden_states_path")):
            _path(cfg["data"][field], Path("hidden_states/m3-04/smoke") / key / label)
        _path(cfg["data"]["cache_dir"], Path("datasets/m3-04/processed") / key / "cache")
        return None
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20]
    draft = payload["resolved_draft"]
    if (key != payload["artifact_key"] or key != profile["artifact_key"]
            or contract["specforge_sha"] != SPECFORGE_SHA or profile["specforge_sha"] != SPECFORGE_SHA
            or payload["target_revision"] != profile["target_revision"]
            or feature_counts(contract)["train"] != profile["train_samples"]
            or cfg["data"]["max_length"] != contract["max_length"] or contract["max_length"] != profile["max_length"]
            or draft["vocab_size"] != profile["target_vocab_size"] or draft["draft_vocab_size"] != profile["draft_vocab_size"]
            or cfg["model"]["target_model_path"] != contract["target"]
            or cfg["model"]["torch_dtype"] != contract["dtype"] or contract["dtype"] != "bfloat16"
            or cfg["training"]["strategy"] != "eagle3" or not cfg["data"]["is_preformatted"]):
        raise ValueError("Vocabulary mapping artifact/cache/contract is not covered by the accepted proof")
    train = _path(cfg["data"]["hidden_states_path"], Path("hidden_states/m3-04/train") / key)
    _path(cfg["data"]["eval_hidden_states_path"], Path("hidden_states/m3-04/eval") / key)
    cache = _path(cfg["data"]["cache_dir"], Path("datasets/m3-04/processed") / key / "cache")
    recipe = f"{cache.parent / 'train.jsonl'}-{profile['max_length']}-{cfg['data']['chat_template']}-{contract['target']}-None-True"
    # The cache recipe is a path string, including POSIX separators on Modal.
    cache_key = hashlib.md5(recipe.replace("\\", "/").encode()).hexdigest()
    if cache_key != profile["cache_identity"]:
        raise ValueError("Processed TRAIN cache identity mismatch")
    manifest_path = train / "artifact_manifest.json"
    data = manifest_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != profile["train_manifest_sha256"]:
        raise ValueError("TRAIN manifest is not the accepted mapping proof input")
    manifest = json.loads(data)
    if (manifest["contract"] != contract or manifest["target_revision"] != payload["target_revision"]
            or manifest["capture_layers"] != payload["capture_layers"]
            or len(manifest["files"]) != profile["train_samples"]):
        raise ValueError("Vocabulary mapping feature contract mismatch")
    paths = [str(train / entry["path"]) for entry in manifest["files"]]
    indices = [cache_sample_index(path, profile["train_samples"]) for path in paths]
    if len(set(indices)) != profile["train_samples"] or paths != sorted(paths):
        raise ValueError("Feature identities/order mismatch")
    output = cache / "vocab_mapping/native_canonical.pt"
    metadata = output.with_suffix(".json")
    for path in (output, metadata):
        _path(path, path.relative_to(VOLUME_ROOT))
    configured = cfg["model"].get("vocab_mapping_path")
    if configured and configured != str(output):
        raise ValueError("Explicit mapping must be the authenticated native mapping, never prepared mapping A")
    binding = {"kind": "native-offline-vocab-v1", "proof": profile, "contract": contract,
               "capture_layers": payload["capture_layers"], "target_revision": payload["target_revision"]}
    return profile, cache, paths, output, metadata, binding


def _load_cache(cache, profile):
    import pyarrow as pa
    directory = cache / "processed_dataset"
    expected = [item["name"] for item in profile["cache_shards"]]
    actual = sorted(p.name for p in directory.glob(profile["cache_identity"] + "_*.pkl"))
    if actual != expected:
        raise ValueError("Processed TRAIN cache shards mismatch")
    tables = []
    for item in profile["cache_shards"]:
        path = directory / item["name"]
        _path(path, path.relative_to(VOLUME_ROOT))
        data = path.read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Processed TRAIN cache content changed")
        table = pa.ipc.open_stream(pa.BufferReader(data)).read_all().select(["input_ids", "loss_mask"])
        if table.num_rows != item["rows"]:
            raise ValueError("Processed TRAIN cache row count mismatch")
        tables.append(table)
    table = pa.concat_tables(tables)
    if table.num_rows != profile["train_samples"]:
        raise ValueError("Processed TRAIN cache total count mismatch")
    return table


def _reload(output, metadata, binding, profile):
    import torch
    if json.loads(metadata.read_text(encoding="utf-8")) != binding:
        raise ValueError("Persisted vocabulary mapping binding mismatch")
    mapping = torch.load(output, map_location="cpu", weights_only=True)
    validate_tensors(mapping, profile)
    return mapping


def persist_mapping(mapping, output, metadata, binding, profile):
    """Publish standard EAGLE tensors and JSON evidence; never overwrite a seal."""
    import torch
    validate_tensors(mapping, profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A concurrent preflight may have published while this one was counting.
    if output.exists() or metadata.exists():
        return _reload(output, metadata, binding, profile)
    temporary = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(mapping, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=True)
        validate_tensors(reloaded, profile)
        # Exclusive creation prevents replacing any existing evidence.
        with output.open("xb") as stream:
            stream.write(temporary.read_bytes())
        with metadata.open("x", encoding="utf-8") as stream:
            json.dump(binding, stream, sort_keys=True, indent=2)
    finally:
        temporary.unlink(missing_ok=True)
    return _reload(output, metadata, binding, profile)


def prepare_native_vocab_mapping(payload):
    """CPU preflight only: authenticate cache, derive/reuse, bind explicit path."""
    context = _context(payload)
    if context is None:
        return
    profile, cache, paths, output, metadata, binding = context
    table = _load_cache(cache, profile)  # Also authenticate current inputs on reuse.
    if output.exists() or metadata.exists():
        _reload(output, metadata, binding, profile)
    else:
        from specforge.runtime.data_plane.offline_reader import list_feature_files
        if list_feature_files(str(Path(paths[0]).parents[1])) != paths:
            raise ValueError("Native feature discovery differs from authenticated inventory")
        def get_row(index):
            return {key: table[key][index].as_py() for key in ("input_ids", "loss_mask")}
        counts = count_cache_tokens(paths, get_row, profile["train_samples"],
                                    max_length=profile["max_length"], target_vocab_size=profile["target_vocab_size"])
        persist_mapping(select_mapping(counts, profile), output, metadata, binding, profile)
    payload["config"]["model"]["vocab_mapping_path"] = str(output)


def validate_native_vocab_mapping(payload):
    """Execution gate: check the seal and canonical tensor hashes, without scans."""
    context = _context(payload)
    if context is None:
        return
    profile, _, _, output, metadata, binding = context
    if payload["config"]["model"].get("vocab_mapping_path") != str(output):
        raise ValueError("Execution config requires the explicit native vocabulary mapping")
    _reload(output, metadata, binding, profile)
