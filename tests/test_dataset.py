import json
import pytest
from pathlib import Path
from data.prepare_hybrid_dataset import (
    filter_and_format_conversation,
    build_hybrid_dataset,
    build_train_val_test_splits,
)


def test_filter_and_format_conversation_valid():
    raw_convo = [
        {"from": "human", "value": "Trích xuất thông tin hóa đơn sau: Cty ABC MST 0123456789"},
        {"from": "gpt", "value": '{"company": "Cty ABC", "tax_code": "0123456789"}'},
    ]
    res = filter_and_format_conversation(raw_convo, sample_id="sample_0")
    assert res is not None
    assert res["id"] == "sample_0"
    assert len(res["conversations"]) == 2
    assert res["conversations"][0]["from"] == "human"
    assert res["conversations"][0]["value"] == "Trích xuất thông tin hóa đơn sau: Cty ABC MST 0123456789"
    assert res["conversations"][1]["from"] == "gpt"
    assert res["conversations"][1]["value"] == '{"company": "Cty ABC", "tax_code": "0123456789"}'


def test_filter_and_format_conversation_with_role_content_input():
    raw_convo = [
        {"role": "user", "content": "Xin chào"},
        {"role": "assistant", "content": "Chào bạn, tôi có thể giúp gì cho bạn?"},
    ]
    res = filter_and_format_conversation(raw_convo, sample_id="sample_1")
    assert res is not None
    assert res["id"] == "sample_1"
    assert len(res["conversations"]) == 2
    assert res["conversations"][0]["from"] == "human"
    assert res["conversations"][0]["value"] == "Xin chào"
    assert res["conversations"][1]["from"] == "gpt"
    assert res["conversations"][1]["value"] == "Chào bạn, tôi có thể giúp gì cho bạn?"


def test_filter_empty_or_invalid_conversation():
    assert filter_and_format_conversation([]) is None
    assert filter_and_format_conversation([{"from": "human", "value": ""}]) is None
    assert filter_and_format_conversation([{"from": "human", "value": "Chỉ có 1 lượt hỏi"}]) is None


def test_sanitizes_unusual_line_terminators():
    raw_convo = [
        {"from": "human", "value": "Line 1\u2028Line 2"},
        {"from": "gpt", "value": "Paragraph 1\u2029Paragraph 2"},
    ]
    res = filter_and_format_conversation(raw_convo, sample_id="sample_line_term")
    assert res is not None
    assert "\u2028" not in res["conversations"][0]["value"]
    assert "\u2029" not in res["conversations"][1]["value"]
    assert "Line 1\nLine 2" == res["conversations"][0]["value"]
    assert "Paragraph 1\n\nParagraph 2" == res["conversations"][1]["value"]


def test_build_hybrid_dataset_outputs_jsonl(tmp_path: Path):
    sharegpt_file = tmp_path / "sharegpt.json"
    vi_file = tmp_path / "vi_invoice.json"
    out_file = tmp_path / "hybrid_train.jsonl"

    dummy_sg = [
        {
            "id": f"sg_{i}",
            "conversations": [
                {"from": "human", "value": f"Hi {i}"},
                {"from": "gpt", "value": f"Hello {i}"},
            ],
        }
        for i in range(5)
    ]
    dummy_vi = [
        {
            "id": f"vi_{i}",
            "conversations": [
                {"from": "human", "value": f"Hóa đơn {i}"},
                {"from": "gpt", "value": f"Thông tin {i}"},
            ],
        }
        for i in range(5)
    ]

    sharegpt_file.write_text(json.dumps(dummy_sg), encoding="utf-8")
    vi_file.write_text(json.dumps(dummy_vi), encoding="utf-8")

    stats = build_hybrid_dataset(
        sharegpt_path=str(sharegpt_file),
        vietnamese_path=str(vi_file),
        output_path=str(out_file),
        sharegpt_limit=3,
        vi_limit=2,
        seed=42,
    )
    assert stats["total_samples"] == 5
    assert stats["sharegpt_count"] == 3
    assert stats["vietnamese_count"] == 2
    assert out_file.exists()

    # Xác thực định dạng JSONL (mỗi dòng là một JSON object hợp lệ)
    lines = out_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5

    parsed_samples = [json.loads(line) for line in lines]
    for sample in parsed_samples:
        assert "id" in sample
        assert "conversations" in sample
        assert len(sample["conversations"]) >= 2
        for turn in sample["conversations"]:
            assert "from" in turn
            assert "value" in turn
            assert turn["from"] in ["human", "gpt"]


def test_build_train_val_test_splits(tmp_path: Path):
    vi_file = tmp_path / "vi.json"
    inv_file = tmp_path / "invoice.json"
    en_file = tmp_path / "en.json"
    out_dir = tmp_path / "processed"

    dummy_vi = [
        {"id": f"vi_{i}", "conversations": [{"from": "human", "value": f"Hỏi TV {i}"}, {"from": "gpt", "value": f"Đáp TV {i}"}]}
        for i in range(25)
    ]
    dummy_inv = [
        {"id": f"inv_{i}", "conversations": [{"from": "human", "value": f"Hóa đơn {i}"}, {"from": "gpt", "value": f'{{"mst": "{i}"}}'}]}
        for i in range(25)
    ]
    dummy_en = [
        {"id": f"en_{i}", "conversations": [{"from": "human", "value": f"Prompt {i}"}, {"from": "gpt", "value": f"Step by step {i}"}]}
        for i in range(25)
    ]

    vi_file.write_text(json.dumps(dummy_vi), encoding="utf-8")
    inv_file.write_text(json.dumps(dummy_inv), encoding="utf-8")
    en_file.write_text(json.dumps(dummy_en), encoding="utf-8")

    stats = build_train_val_test_splits(
        vietnamese_path=str(vi_file),
        invoice_path=str(inv_file),
        english_path=str(en_file),
        output_dir=str(out_dir),
        train_counts={"vietnamese": 10, "invoice": 8, "english": 6},
        val_counts={"vietnamese": 2, "invoice": 2, "english": 2},
        test_counts={"vietnamese": 3, "invoice": 4, "english": 2},
        seed=42,
    )

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    test_path = out_dir / "test.jsonl"

    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()

    assert stats["train"]["total"] == 24
    assert stats["val"]["total"] == 6
    assert stats["test"]["total"] == 9

    train_lines = [json.loads(line) for line in train_path.read_text(encoding="utf-8").strip().split("\n")]
    val_lines = [json.loads(line) for line in val_path.read_text(encoding="utf-8").strip().split("\n")]
    test_lines = [json.loads(line) for line in test_path.read_text(encoding="utf-8").strip().split("\n")]

    assert len(train_lines) == 24
    assert len(val_lines) == 6
    assert len(test_lines) == 9

    train_ids = {s["id"] for s in train_lines}
    val_ids = {s["id"] for s in val_lines}
    test_ids = {s["id"] for s in test_lines}

    # Đảm bảo các tập hoàn toàn phân tách (disjoint) - không rò rỉ dữ liệu (No Data Leakage)
    assert train_ids.isdisjoint(val_ids), "Tập Train và Val bị trùng lặp mẫu!"
    assert train_ids.isdisjoint(test_ids), "Tập Train và Test bị trùng lặp mẫu (Data Leakage)!"
    assert val_ids.isdisjoint(test_ids), "Tập Val và Test bị trùng lặp mẫu!"
