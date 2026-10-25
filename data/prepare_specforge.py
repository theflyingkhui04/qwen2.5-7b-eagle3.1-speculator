"""Lossless deterministic serialization before explicit token-length truncation."""
import hashlib
import json
from pathlib import Path
from training.train import sha256_file

ROLE_MAP = {"human": "user", "gpt": "assistant"}
SYSTEM_PROMPT = "You are a helpful assistant."


def prepared_row(row, tokenizer):
    messages = []
    for turn in row["conversations"]:
        if set(turn) != {"from", "value"} or turn["from"] not in ROLE_MAP:
            raise ValueError(f"Unsupported message schema in {row['id']}")
        if not isinstance(turn["value"], str):
            raise ValueError("Message content must be a string")
        messages.append({"role": ROLE_MAP[turn["from"]], "content": turn["value"]})
    if not messages:
        raise ValueError("Empty conversation")
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        tokenize=False, add_generation_prompt=False,
    )
    expected = "".join(
        "<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n"
        for m in [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    )
    if text != expected:
        raise ValueError("Target chat template does not preserve the canonical Qwen turn serialization")
    return {"id": row["id"], "messages": messages, "text": text}


def validate_raw(path, expected):
    if sha256_file(path) != expected["sha256"]:
        raise ValueError(f"Raw hash mismatch: {Path(path).name}")
    rows = 0
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if set(row) != {"id", "conversations"} or not isinstance(row["id"], str):
                raise ValueError("Unsupported raw schema")
            for turn in row["conversations"]:
                if set(turn) != {"from", "value"} or turn["from"] not in ROLE_MAP or not isinstance(turn["value"], str):
                    raise ValueError("Unsupported raw turn")
            rows += 1
    if rows != expected["rows"]:
        raise ValueError("Raw row count mismatch")
    return {"rows": rows, "sha256": expected["sha256"]}


def audit_prepared(source, tokenizer, expected):
    """Read-only CPU audit using the real target tokenizer in Modal plan."""
    digest = hashlib.sha256()
    rows = turns = 0
    with open(source, encoding="utf-8") as stream:
        for line in stream:
            row = prepared_row(json.loads(line), tokenizer)
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            digest.update(encoded.encode("utf-8"))
            rows += 1
            turns += len(row["messages"])
    if rows != expected["rows"] or sha256_file(source) != expected["sha256"]:
        raise ValueError("Raw input changed or row count differs")
    return {**expected, "turns": turns, "prepared_sha256": digest.hexdigest()}


def prepare_file(source, destination, tokenizer, expected, *, raw_root):
    source, destination, raw_root = map(lambda p: Path(p).resolve(), (source, destination, raw_root))
    if destination.is_relative_to(raw_root) or destination == source:
        raise ValueError("Generated artifacts must be outside raw data")
    raw = validate_raw(source, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    turns = 0
    with source.open(encoding="utf-8") as inp, destination.open("x", encoding="utf-8", newline="\n") as out:
        for line in inp:
            row = prepared_row(json.loads(line), tokenizer)
            turns += len(row["messages"])
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    if sha256_file(source) != raw["sha256"]:
        raise ValueError("Raw source changed during preparation")
    return {**raw, "turns": turns, "prepared_sha256": sha256_file(destination)}


def preformatted_rows(file_path):
    """Bridge the pinned canonical loader, which drops its preformatted text field."""
    with open(file_path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or not row.get("messages"):
                raise ValueError("Expected project-prepared preformatted row")
            yield {"text": row["text"], "id": row["id"]}
