import json
import pytest
from pathlib import Path
from data.prepare_hybrid_dataset import (
    filter_and_format_conversation,
    build_hybrid_dataset,
)

def test_filter_and_format_conversation():
    raw_convo = [
        {"from": "human", "value": "Trích xuất thông tin hóa đơn sau: Cty ABC MST 0123456789"},
        {"from": "gpt", "value": '{"company": "Cty ABC", "tax_code": "0123456789"}'}
    ]
    res = filter_and_format_conversation(raw_convo)
    assert res is not None
    assert len(res["conversations"]) == 2
    assert res["conversations"][0]["role"] == "user"
    assert res["conversations"][1]["role"] == "assistant"

def test_filter_empty_or_invalid_conversation():
    assert filter_and_format_conversation([]) is None
    assert filter_and_format_conversation([{"from": "human", "value": ""}]) is None

def test_build_hybrid_dataset(tmp_path):
    sharegpt_file = tmp_path / "sharegpt.json"
    vi_file = tmp_path / "vi_invoice.json"
    out_file = tmp_path / "hybrid.json"

    dummy_sg = [
        {"conversations": [{"from": "human", "value": f"Hi {i}"}, {"from": "gpt", "value": f"Hello {i}"}]}
        for i in range(5)
    ]
    dummy_vi = [
        {"conversations": [{"from": "human", "value": f"Hóa đơn {i}"}, {"from": "gpt", "value": f"Thông tin {i}"}]}
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

    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == 5
    assert all("conversations" in sample for sample in loaded)
