"""
Script prepare_hybrid_dataset.py
Gộp dữ liệu ShareGPT_Vicuna_unfiltered (20k) và Tiếng Việt/Hóa đơn MC-OCR (10k)
Chuẩn hóa cấu trúc hội thoại đa lượt và lưu định dạng JSON.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def filter_and_format_conversation(raw_convo: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Chuẩn hóa một đoạn hội thoại sang định dạng:
    {"conversations": [{"role": "user"|"assistant", "content": str}, ...]}
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
            role = "user"
        elif raw_role_lower in ["gpt", "assistant", "bot"]:
            role = "assistant"
        else:
            continue

        formatted_turns.append({"role": role, "content": str(content).strip()})

    # Cần tối thiểu 1 cặp user - assistant
    if len(formatted_turns) < 2:
        return None

    return {"conversations": formatted_turns}


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
    Gộp và xáo trộn tập dữ liệu lai theo tỷ lệ xác định.
    """
    logger.info("Đang nạp tập dữ liệu ShareGPT từ: %s", sharegpt_path)
    raw_sharegpt = load_raw_dataset(sharegpt_path)

    logger.info("Đang nạp tập dữ liệu Tiếng Việt / MC-OCR từ: %s", vietnamese_path)
    raw_vietnamese = load_raw_dataset(vietnamese_path)

    cleaned_sharegpt = []
    for item in raw_sharegpt:
        convo = item.get("conversations") or item.get("messages") or []
        formatted = filter_and_format_conversation(convo)
        if formatted:
            cleaned_sharegpt.append(formatted)
        if len(cleaned_sharegpt) >= sharegpt_limit:
            break

    cleaned_vietnamese = []
    for item in raw_vietnamese:
        convo = item.get("conversations") or item.get("messages") or []
        formatted = filter_and_format_conversation(convo)
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
        json.dump(hybrid, f, ensure_ascii=False, indent=2)

    logger.info("Đã ghi thành công %d mẫu vào %s", len(hybrid), str(out_file))

    return {
        "total_samples": len(hybrid),
        "sharegpt_count": len(cleaned_sharegpt),
        "vietnamese_count": len(cleaned_vietnamese),
        "output_path": str(out_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị tập dữ liệu lai cho huấn luyện EAGLE-3.1")
    parser.add_argument("--sharegpt-path", type=str, default="data/raw/sharegpt.json", help="Đường dẫn file ShareGPT JSON")
    parser.add_argument("--vi-path", type=str, default="data/raw/vietnamese_invoices.json", help="Đường dẫn file Tiếng Việt JSON")
    parser.add_argument("--output-path", type=str, default="data/processed/hybrid_dataset.json", help="Đường dẫn file xuất")
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
