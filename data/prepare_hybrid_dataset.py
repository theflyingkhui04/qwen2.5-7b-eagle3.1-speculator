"""
Script prepare_hybrid_dataset.py
Chuẩn hóa dữ liệu hội thoại đa lượt và xuất định dạng ShareGPT JSON Lines (.jsonl).
Hỗ trợ phân chia 3 tập Train / Val / Test độc lập theo chuẩn Machine Learning
nhằm triệt tiêu nguy cơ rò rỉ dữ liệu (data leakage) và tối ưu hóa cho SpecForge EAGLE-3.1.
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

    if len(formatted_turns) < 2:
        return None

    has_human = any(t["from"] == "human" for t in formatted_turns)
    has_gpt = any(t["from"] == "gpt" for t in formatted_turns)
    if not (has_human and has_gpt):
        return None

    return {
        "id": sample_id,
        "conversations": formatted_turns,
    }


def load_raw_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Đọc dữ liệu thô từ file JSON, JSONL hoặc Parquet."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        samples = []
        col_names = table.column_names
        for i in range(len(table)):
            instr = str(table["instruction"][i]) if "instruction" in col_names else ""
            inp = str(table["input"][i]) if "input" in col_names else ""
            out = str(table["output"][i]) if "output" in col_names else ""
            user_text = f"{instr}\n{inp}".strip() if inp else instr.strip()
            if user_text and out:
                samples.append({
                    "id": f"parquet_{i}",
                    "conversations": [
                        {"from": "human", "value": user_text},
                        {"from": "gpt", "value": out.strip()},
                    ],
                })
        return samples
    elif suffix == ".jsonl":
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


def _clean_and_format_source(raw_items: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    """Lọc sạch và gán tiền tố ID cho từng mẫu từ nguồn dữ liệu."""
    cleaned = []
    for idx, item in enumerate(raw_items):
        convo = item.get("conversations") or item.get("messages") or []
        sample_id = str(item.get("id") or f"{prefix}_{idx}")
        if not sample_id.startswith(f"{prefix}_"):
            sample_id = f"{prefix}_{sample_id}"
        formatted = filter_and_format_conversation(convo, sample_id=sample_id)
        if formatted:
            cleaned.append(formatted)
    return cleaned


def build_train_val_test_splits(
    vietnamese_path: str,
    invoice_path: Optional[str] = None,
    english_path: Optional[str] = None,
    output_dir: str = "data/processed",
    train_counts: Optional[Dict[str, int]] = None,
    val_counts: Optional[Dict[str, int]] = None,
    test_counts: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Phân chia dữ liệu thành 3 tập độc lập (Train, Val, Test) theo chuẩn Machine Learning,
    đảm bảo không rò rỉ dữ liệu (No Data Leakage).
    """
    rng = random.Random(seed)

    # Cấu hình mặc định theo Blueprint: 40k Train, 3k Val, 2k Test
    default_train = {"vietnamese": 16000, "invoice": 14000, "english": 10000}
    default_val = {"vietnamese": 1200, "invoice": 1050, "english": 750}
    default_test = {"vietnamese": 600, "invoice": 1000, "english": 400}

    tc = train_counts if train_counts is not None else default_train
    vc = val_counts if val_counts is not None else default_val
    tec = test_counts if test_counts is not None else default_test

    logger.info("Đang nạp nguồn Tiếng Việt từ: %s", vietnamese_path)
    vi_cleaned = _clean_and_format_source(load_raw_dataset(vietnamese_path), prefix="vi")

    if invoice_path and invoice_path != vietnamese_path:
        logger.info("Đang nạp nguồn Hóa đơn / Structured từ: %s", invoice_path)
        inv_cleaned = _clean_and_format_source(load_raw_dataset(invoice_path), prefix="inv")
    elif invoice_path and invoice_path == vietnamese_path:
        logger.info("Nguồn Hóa đơn trùng với Tiếng Việt: Phân chia 50-50 để tránh trùng lặp mẫu")
        half = len(vi_cleaned) // 2
        inv_raw = vi_cleaned[half:]
        vi_cleaned = vi_cleaned[:half]
        inv_cleaned = []
        for s in inv_raw:
            new_id = s["id"].replace("vi_", "inv_", 1)
            inv_cleaned.append({"id": new_id, "conversations": s["conversations"]})
    else:
        inv_cleaned = []

    if english_path:
        logger.info("Đang nạp nguồn Tiếng Anh / CoT từ: %s", english_path)
        en_cleaned = _clean_and_format_source(load_raw_dataset(english_path), prefix="en")
    else:
        en_cleaned = []

    # Xáo trộn độc lập từng nguồn trước khi cắt lát (slice)
    rng.shuffle(vi_cleaned)
    rng.shuffle(inv_cleaned)
    rng.shuffle(en_cleaned)

    sources = {
        "vietnamese": vi_cleaned,
        "invoice": inv_cleaned,
        "english": en_cleaned,
    }

    train_splits = []
    val_splits = []
    test_splits = []
    breakdown = {"train": {}, "val": {}, "test": {}}

    for name, items in sources.items():
        if not items:
            continue

        n_train = tc.get(name, 0)
        n_val = vc.get(name, 0)
        n_test = tec.get(name, 0)

        # Cắt lát phân tách hoàn toàn (strictly disjoint)
        train_slice = items[0:n_train]
        val_slice = items[n_train:n_train + n_val]
        test_slice = items[n_train + n_val:n_train + n_val + n_test]

        train_splits.extend(train_slice)
        val_splits.extend(val_slice)
        test_splits.extend(test_slice)

        breakdown["train"][name] = len(train_slice)
        breakdown["val"][name] = len(val_slice)
        breakdown["test"][name] = len(test_slice)

    # Xáo trộn nội bộ mỗi tập sau khi gộp
    rng.shuffle(train_splits)
    rng.shuffle(val_splits)
    rng.shuffle(test_splits)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = out_dir / "train.jsonl"
    val_file = out_dir / "val.jsonl"
    test_file = out_dir / "test.jsonl"

    for file_path, data in [(train_file, train_splits), (val_file, val_splits), (test_file, test_splits)]:
        with open(file_path, "w", encoding="utf-8") as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(
        "Hoàn tất phân chia: Train=%d mẫu, Val=%d mẫu, Test=%d mẫu tại %s",
        len(train_splits),
        len(val_splits),
        len(test_splits),
        str(out_dir),
    )

    return {
        "train": {"total": len(train_splits), "breakdown": breakdown["train"], "path": str(train_file)},
        "val": {"total": len(val_splits), "breakdown": breakdown["val"], "path": str(val_file)},
        "test": {"total": len(test_splits), "breakdown": breakdown["test"], "path": str(test_file)},
    }


def build_hybrid_dataset(
    sharegpt_path: str,
    vietnamese_path: str,
    output_path: str,
    sharegpt_limit: int = 20000,
    vi_limit: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Gộp và xáo trộn tập dữ liệu lai theo tỷ lệ xác định (tương thích backward),
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
    parser = argparse.ArgumentParser(description="Chuẩn bị tập dữ liệu cho huấn luyện SpecForge EAGLE-3.1")
    parser.add_argument("--mode", type=str, choices=["splits", "hybrid"], default="splits", help="Chế độ: splits (train/val/test) hoặc hybrid")
    parser.add_argument("--vi-path", type=str, default="data/raw/vi_alpaca_full.parquet", help="Đường dẫn file Tiếng Việt JSON/JSONL/Parquet")
    parser.add_argument("--inv-path", type=str, default="data/raw/vi_alpaca_full.parquet", help="Đường dẫn file Hóa đơn JSON/JSONL/Parquet")
    parser.add_argument("--en-path", type=str, default="data/raw/sharegpt.json", help="Đường dẫn file Tiếng Anh/ShareGPT JSON/JSONL")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Thư mục xuất các file JSONL")
    parser.add_argument("--seed", type=int, default=42, help="Seed xáo trộn dữ liệu")
    args = parser.parse_args()

    if args.mode == "splits":
        build_train_val_test_splits(
            vietnamese_path=args.vi_path,
            invoice_path=args.inv_path,
            english_path=args.en_path,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    else:
        build_hybrid_dataset(
            sharegpt_path=args.en_path,
            vietnamese_path=args.vi_path,
            output_path=f"{args.output_dir}/hybrid_train.jsonl",
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
