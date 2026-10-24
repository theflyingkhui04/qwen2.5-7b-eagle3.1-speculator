"""
Script prepare_hybrid_dataset.py
Gộp dữ liệu ShareGPT_Vicuna_unfiltered (20k) và Tiếng Việt/Hóa đơn MC-OCR (10k)
Chuẩn hóa cấu trúc hội thoại đa lượt và lưu định dạng ShareGPT JSON Lines (.jsonl)
tương thích trực tiếp với SpecForge và SGLang data loader.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def filter_and_format_conversation(
    raw_convo: List[Dict[str, Any]],
    sample_id: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Chuẩn hóa một đoạn hội thoại sang định dạng ShareGPT:
    {
        "id": sample_id,
        "conversations": [{"from": "human"|"gpt", "value": str}, ...]
    }
    Lọc bỏ các đoạn hội thoại rỗng hoặc chỉ có 1 lượt.
    """
    if not raw_convo or not isinstance(raw_convo, list) or len(raw_convo) < 2:
        return None

    formatted_turns = []
    for turn in raw_convo:
        if not isinstance(turn, dict):
            continue

        # Hỗ trợ cả 2 chuẩn format: 'from'/'value' và 'role'/'content'
        raw_role = turn.get("from") or turn.get("role") or ""
        content = turn.get("value") or turn.get("content") or ""

        if not str(content).strip():
            continue

        raw_role_lower = str(raw_role).lower()
        if raw_role_lower in ["human", "user"]:
            role = "human"
        elif raw_role_lower in ["gpt", "assistant", "bot"]:
            role = "gpt"
        else:
            continue

        clean_content = str(content).replace("\u2028", "\n").replace("\u2029", "\n\n").strip()
        formatted_turns.append({"from": role, "value": clean_content})

    # Cần tối thiểu 1 cặp human - gpt
    if len(formatted_turns) < 2:
        return None

    # Kiểm tra đảm bảo có ít nhất 1 human và 1 gpt
    has_human = any(t["from"] == "human" for t in formatted_turns)
    has_gpt = any(t["from"] == "gpt" for t in formatted_turns)
    if not (has_human and has_gpt):
        return None

    return {
        "id": sample_id,
        "conversations": formatted_turns,
    }


def load_raw_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Đọc dữ liệu thô từ file JSON hoặc JSONL."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    if path.suffix.lower() == ".jsonl":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    samples.append(json.loads(line_str))
        return samples
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "data" in data:
                return data["data"]
            else:
                return [data]


def build_hybrid_dataset(
    sharegpt_path: str,
    vietnamese_path: str,
    output_path: str,
    sharegpt_limit: int = 20000,
    vi_limit: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Gộp và xáo trộn tập dữ liệu lai theo tỷ lệ xác định,
    xuất ra định dạng ShareGPT JSON Lines (.jsonl).
    """
    logger.info("Đang nạp tập dữ liệu ShareGPT từ: %s", sharegpt_path)
    raw_sharegpt = load_raw_dataset(sharegpt_path)

    logger.info("Đang nạp tập dữ liệu Tiếng Việt / MC-OCR từ: %s", vietnamese_path)
    raw_vietnamese = load_raw_dataset(vietnamese_path)

    cleaned_sharegpt = []
    for idx, item in enumerate(raw_sharegpt):
        convo = item.get("conversations") or item.get("messages") or []
        sample_id = str(item.get("id") or f"sharegpt_{idx}")
        formatted = filter_and_format_conversation(convo, sample_id=sample_id)
        if formatted:
            cleaned_sharegpt.append(formatted)
        if len(cleaned_sharegpt) >= sharegpt_limit:
            break

    cleaned_vietnamese = []
    for idx, item in enumerate(raw_vietnamese):
        convo = item.get("conversations") or item.get("messages") or []
        sample_id = str(item.get("id") or f"vi_invoice_{idx}")
        formatted = filter_and_format_conversation(convo, sample_id=sample_id)
        if formatted:
            cleaned_vietnamese.append(formatted)
        if len(cleaned_vietnamese) >= vi_limit:
            break

    logger.info("Lọc được: %d mẫu ShareGPT, %d mẫu Tiếng Việt / MC-OCR", len(cleaned_sharegpt), len(cleaned_vietnamese))

    hybrid = cleaned_sharegpt + cleaned_vietnamese
    random.seed(seed)
    random.shuffle(hybrid)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for sample in hybrid:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Đã ghi thành công %d mẫu vào file JSONL %s", len(hybrid), str(out_file))

    return {
        "total_samples": len(hybrid),
        "sharegpt_count": len(cleaned_sharegpt),
        "vietnamese_count": len(cleaned_vietnamese),
        "output_path": str(out_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị tập dữ liệu lai JSONL cho huấn luyện SpecForge EAGLE-3.1")
    parser.add_argument("--sharegpt-path", type=str, default="data/raw/sharegpt.json", help="Đường dẫn file ShareGPT JSON/JSONL")
    parser.add_argument("--vi-path", type=str, default="data/raw/vietnamese_invoices.json", help="Đường dẫn file Tiếng Việt JSON/JSONL")
    parser.add_argument("--output-path", type=str, default="data/processed/hybrid_train.jsonl", help="Đường dẫn file JSONL xuất ra")
    parser.add_argument("--sharegpt-limit", type=int, default=20000, help="Số lượng mẫu ShareGPT tối đa")
    parser.add_argument("--vi-limit", type=int, default=10000, help="Số lượng mẫu Tiếng Việt tối đa")
    parser.add_argument("--seed", type=int, default=42, help="Seed xáo trộn dữ liệu")
    args = parser.parse_args()

    build_hybrid_dataset(
        sharegpt_path=args.sharegpt_path,
        vietnamese_path=args.vi_path,
        output_path=args.output_path,
        sharegpt_limit=args.sharegpt_limit,
        vi_limit=args.vi_limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
