import json
from pathlib import Path
from unittest.mock import patch
import pytest
from data.prepare_specforge import (
    ROLE_MAP, SYSTEM_PROMPT, audit_prepared, prepared_row, prepare_file, preformatted_rows, validate_raw,
)
from training.train import sha256_file, load_training_config
from training.prepare import canonical_preparation, preparation_args

ROOT = Path(__file__).resolve().parents[1]


class ChatMLTokenizer:
    """Reference renderer for the plain-text Qwen ChatML contract."""
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is False
        return "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n" for m in messages)


def sample():
    return {"id": "regression", "conversations": [
        {"from": "gpt", "value": "Leading assistant\nGiữ nguyên"},
        {"from": "human", "value": "User"},
        {"from": "gpt", "value": "Assistant 1"},
        {"from": "gpt", "value": "Assistant 2"},
        {"from": "human", "value": "Final user"},
    ]}


def test_leading_and_consecutive_assistants_preserved_before_tokenization():
    raw = sample()
    result = prepared_row(raw, ChatMLTokenizer())
    assert len(result["messages"]) == len(raw["conversations"])
    assert [(m["role"], m["content"]) for m in result["messages"]] == [
        (ROLE_MAP[m["from"]], m["value"]) for m in raw["conversations"]
    ]
    assert result["text"].startswith("<|im_start|>system\n" + SYSTEM_PROMPT)
    for turn in raw["conversations"]:
        assert turn["value"] in result["text"]


def test_unknown_schema_fails_instead_of_silent_drop():
    row = sample()
    row["conversations"][0]["from"] = "unknown"
    with pytest.raises(ValueError):
        prepared_row(row, ChatMLTokenizer())


def test_changed_template_fails_closed():
    class BadTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "dropped everything"
    with pytest.raises(ValueError, match="preserve"):
        prepared_row(sample(), BadTokenizer())


def test_deterministic_files_counts_raw_protection_and_loader(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "train.jsonl"
    source.write_text(json.dumps(sample(), ensure_ascii=False) + "\n", encoding="utf-8")
    expected = {"rows": 1, "sha256": sha256_file(source)}
    outputs = [tmp_path / "processed" / name for name in ("one.jsonl", "two.jsonl")]
    results = [prepare_file(source, p, ChatMLTokenizer(), expected, raw_root=raw) for p in outputs]
    assert results[0] == results[1]
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert sha256_file(source) == expected["sha256"]
    assert results[0]["turns"] == 5
    assert audit_prepared(source, ChatMLTokenizer(), expected) == results[0]
    assert len(list(preformatted_rows(outputs[0]))) == 1
    assert list(preformatted_rows(outputs[0]))[0]["text"] == json.loads(outputs[0].read_text(encoding="utf-8"))["text"]
    with pytest.raises(FileExistsError):
        prepare_file(source, outputs[0], ChatMLTokenizer(), expected, raw_root=raw)
    with pytest.raises(ValueError, match="outside raw"):
        prepare_file(source, raw / "new.jsonl", ChatMLTokenizer(), expected, raw_root=raw)
    with pytest.raises(ValueError, match="hash"):
        validate_raw(source, {"rows": 1, "sha256": "bad"})


def test_loader_rejects_missing_text(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(sample()) + "\n")
    with pytest.raises(ValueError):
        list(preformatted_rows(path))


def test_canonical_loader_bridge_without_upstream_source_edit(tmp_path):
    # Model/extractor remain canonical; the wrapper swaps only the loader binding.
    script = tmp_path / "upstream.py"
    script.write_text(
        "import sys\n"
        "def safe_conversations_generator(path):\n"
        "    raise AssertionError('legacy loader must not be used')\n"
        "def main():\n"
        "    rows = list(safe_conversations_generator(sys.argv[1]))\n"
        "    assert len(rows) == 1 and 'Leading assistant' in rows[0]['text']\n",
        encoding="utf-8",
    )
    before = script.read_bytes()
    path = tmp_path / "prepared.jsonl"
    path.write_text(json.dumps(prepared_row(sample(), ChatMLTokenizer())) + "\n")
    canonical_preparation([str(path)], script)
    assert script.read_bytes() == before


def test_canonical_preparation_args_do_not_cap_or_relabel_dataset():
    args = preparation_args(load_training_config(), "prepared.jsonl", "features")
    assert "--is-preformatted" in args
    assert "--num-samples" not in args
    assert args[args.index("--max-length") + 1] == "2048"
    assert args[args.index("--chat-template") + 1] == "qwen"


@pytest.mark.parametrize("count", [64, 16])
def test_bounded_preparation_forwards_upstream_num_samples(count):
    args = preparation_args(load_training_config(), "prepared.jsonl", "smoke-features", count)
    assert args[args.index("--num-samples") + 1] == str(count)
    assert args[args.index("--data-path") + 1] == "prepared.jsonl"
    assert "--is-preformatted" in args


def test_preformatted_prefix_order_is_deterministic_and_read_only(tmp_path):
    path = tmp_path / "prepared.jsonl"
    rows = []
    for index in range(80):
        row = sample()
        row["id"] = str(index)
        rows.append(prepared_row(row, ChatMLTokenizer()))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = path.read_bytes()
    # Canonical --num-samples selects range(N) from this ordered input before shuffle.
    first = list(preformatted_rows(path))[:64]
    second = list(preformatted_rows(path))[:64]
    assert first == second
    assert [row["id"] for row in first] == [str(i) for i in range(64)]
    assert path.read_bytes() == before


def test_full_immutable_snapshot_all_turns_when_available():
    root = ROOT / "data/raw/m3_04_drive"
    if not root.exists():
        pytest.skip("Immutable raw snapshot is not distributed with the repository")
    manifest = json.loads((ROOT / "configs/m3_04_dataset.json").read_text())
    for split, expected in manifest["splits"].items():
        path = root / (split + ".jsonl")
        assert validate_raw(path, expected)["rows"] == expected["rows"]
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                raw = json.loads(line)
                out = prepared_row(raw, ChatMLTokenizer())
                assert len(out["messages"]) == len(raw["conversations"])
                assert [m["content"] for m in out["messages"]] == [m["value"] for m in raw["conversations"]]
                count += 1
        assert count == expected["rows"]
        assert sha256_file(path) == expected["sha256"]
