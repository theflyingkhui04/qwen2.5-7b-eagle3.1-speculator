"""
Script extract_hidden_states.py
Trích xuất Hidden States tầng 28 từ Target Model Qwen2.5-7B-Instruct.
Áp dụng Sequence Bucketing (512, 1024, 2048) và lưu chunks safetensors (bfloat16).
Tích hợp OOM Guard và Disk Overflow Guard (>= 50GB trống).
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_disk_space(path: str, min_free_gb: float = 50.0) -> bool:
    """
    Kiểm tra dung lượng đĩa khả dụng trước khi trích xuất vector.
    Bắt buộc dung lượng trống >= min_free_gb.
    """
    try:
        check_target = path if os.path.exists(path) else os.path.dirname(os.path.abspath(path))
        total, used, free = shutil.disk_usage(check_target)
        free_gb = free / (1024 ** 3)
        logger.info("Kiểm tra ổ đĩa tại '%s': Còn trống %.2f GB (Yêu cầu tối thiểu: %.2f GB)", check_target, free_gb, min_free_gb)
        return free_gb >= min_free_gb
    except Exception as e:
        logger.warning("Không thể kiểm tra dung lượng ổ đĩa: %s", e)
        return True


def scan_existing_chunks(features_dir: Path) -> Tuple[Dict[int, int], int]:
    """
    Quét các file features_chunk_*.safetensors có sẵn trong thư mục để phục vụ Resume.
    Trả về:
    - completed_per_bucket: Dict[int, int] {bucket_len: total_samples_saved}
    - next_chunk_idx: int (chỉ số chunk tiếp theo cần lưu, bắt đầu từ max_idx + 1 hoặc 0)
    """
    completed_per_bucket: Dict[int, int] = {}
    max_chunk_idx = -1

    if not features_dir.exists():
        return completed_per_bucket, 0

    chunk_files = sorted(features_dir.glob("features_chunk_*.safetensors"))
    if not chunk_files:
        return completed_per_bucket, 0

    try:
        from safetensors import safe_open
        has_safetensors = True
    except ImportError:
        logger.warning("Không thể import safe_open từ safetensors để quét metadata chunk cũ.")
        has_safetensors = False

    for f_path in chunk_files:
        match = re.search(r"features_chunk_(\d+)\.safetensors", f_path.name)
        if match:
            idx = int(match.group(1))
            if idx > max_chunk_idx:
                max_chunk_idx = idx

        if has_safetensors:
            try:
                with safe_open(str(f_path), framework="pt", device="cpu") as f:
                    if "hidden_states" in f.keys():
                        shape = f.get_slice("hidden_states").get_shape()
                        # shape: (num_samples, seq_len, hidden_dim)
                        num_samples = shape[0]
                        seq_len = shape[1]
                        completed_per_bucket[seq_len] = completed_per_bucket.get(seq_len, 0) + num_samples
            except Exception as e:
                logger.warning("File chunk %s không hợp lệ hoặc bị lỗi đọc: %s", f_path.name, e)

    next_chunk_idx = max_chunk_idx + 1 if max_chunk_idx >= 0 else 0
    return completed_per_bucket, next_chunk_idx


def bucket_sequences(samples: List[Dict[str, Any]], bucket_limits: Optional[List[int]] = None) -> Dict[int, List[Dict[str, Any]]]:
    """
    Phân loại các mẫu vào các bucket độ dài tương ứng (<= 512, <= 1024, <= 2048).
    Loại bỏ các mẫu dài quá giới hạn cao nhất (OOM Guard).
    """
    if bucket_limits is None:
        bucket_limits = [512, 1024, 2048]
    bucket_limits = sorted(bucket_limits)
    max_len = bucket_limits[-1]

    buckets: Dict[int, List[Dict[str, Any]]] = {b: [] for b in bucket_limits}

    for item in samples:
        length = item.get("length")
        if length is None:
            # Ước lượng chiều dài nếu chưa có
            text = item.get("text") or str(item.get("conversations", ""))
            length = len(text.split())

        if length > max_len:
            logger.debug("Loại bỏ mẫu có độ dài %d vượt trần max_len %d", length, max_len)
            continue

        for b in bucket_limits:
            if length <= b:
                buckets[b].append(item)
                break

    return buckets


def parse_extraction_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trích xuất Hidden States tầng 28 từ Qwen2.5-7B")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Tên hoặc đường dẫn target model")
    parser.add_argument("--dataset-path", type=str, default="data/processed/hybrid_dataset.json", help="Tập dữ liệu lai đầu vào")
    parser.add_argument("--output-dir", type=str, default="data/features", help="Thư mục lưu feature safetensors")
    parser.add_argument("--target-layer", type=int, default=28, help="Tầng trích xuất (tầng 28)")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Số mẫu trên mỗi file safetensors chunk")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size khi forward model lớn")
    parser.add_argument("--min-disk-gb", type=float, default=50.0, help="Dung lượng đĩa trống tối thiểu (GB)")
    parser.add_argument("--force", action="store_true", help="Trích xuất lại từ đầu, bỏ qua các chunk đã có")
    return parser.parse_args(args)


def run_extraction():
    args = parse_extraction_args()

    # Chốt chặn kiểm tra đĩa
    if not check_disk_space(args.output_dir, min_free_gb=args.min_disk_gb):
        logger.error("Dung lượng đĩa khả dụng dưới %.2f GB! Dừng để tránh Disk Overflow.", args.min_disk_gb)
        sys.exit(1)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from safetensors.torch import save_file
    except ImportError:
        logger.error("Cần cài đặt torch, transformers và safetensors trên môi trường chạy thực tế!")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info("Tổng số mẫu: %d. Bắt đầu phân nhóm độ dài...", len(dataset))
    buckets = bucket_sequences(dataset)
    total_needed = sum(len(s) for s in buckets.values())

    completed_per_bucket = {}
    chunk_idx = 0

    if not getattr(args, "force", False):
        completed_per_bucket, next_chunk_idx = scan_existing_chunks(out_dir)
        total_completed = sum(completed_per_bucket.get(b, 0) for b in buckets.keys())

        if total_completed >= total_needed and total_needed > 0:
            logger.info("Toàn bộ %d/%d mẫu đã được trích xuất đầy đủ (%d chunks). Bỏ qua trích xuất!",
                        total_completed, total_needed, len(list(out_dir.glob("features_chunk_*.safetensors"))))
            return

        if completed_per_bucket:
            logger.info("Phát hiện tiến trình trích xuất cũ tại %s:", out_dir)
            for b_len, count in sorted(completed_per_bucket.items()):
                target_count = len(buckets.get(b_len, []))
                logger.info("  - Bucket len <= %d: đã xong %d/%d mẫu", b_len, count, target_count)
            logger.info("  -> Tiếp tục từ chunk index: %d", next_chunk_idx)
        chunk_idx = next_chunk_idx
    else:
        logger.info("Chế độ force: Trích xuất lại từ đầu và ghi đè các chunk cũ.")

    logger.info("Đang nạp Target Model: %s ở định dạng bfloat16...", args.model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    buffer_hidden = []
    buffer_ids = []

    for bucket_len, samples in buckets.items():
        if not samples:
            continue

        done_in_bucket = completed_per_bucket.get(bucket_len, 0)
        if not getattr(args, "force", False) and done_in_bucket >= len(samples):
            logger.info("Bucket <= %d (%d mẫu) đã hoàn thành 100%%. Bỏ qua!", bucket_len, len(samples))
            continue
        elif not getattr(args, "force", False) and done_in_bucket > 0:
            logger.info("Bucket <= %d: đã có %d/%d mẫu. Chạy tiếp từ mẫu %d...", bucket_len, done_in_bucket, len(samples), done_in_bucket)
            samples = samples[done_in_bucket:]
        else:
            logger.info("Xử lý bucket <= %d: %d mẫu", bucket_len, len(samples))

        buffer_hidden = []
        buffer_ids = []
        for i in range(0, len(samples), args.batch_size):
            batch_samples = samples[i : i + args.batch_size]
            texts = [
                tokenizer.apply_chat_template(s["conversations"], tokenize=False, add_generation_prompt=False)
                for s in batch_samples
            ]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=bucket_len,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Tầng trích xuất 28
                layer_hidden = outputs.hidden_states[args.target_layer].cpu().to(torch.bfloat16)
                input_ids = inputs["input_ids"].cpu()

            buffer_hidden.append(layer_hidden)
            buffer_ids.append(input_ids)

            # Lưu chunk nếu chạm mốc chunk_size
            if len(buffer_hidden) * args.batch_size >= args.chunk_size:
                chunk_file = out_dir / f"features_chunk_{chunk_idx:04d}.safetensors"
                tensors = {
                    "hidden_states": torch.cat(buffer_hidden, dim=0),
                    "input_ids": torch.cat(buffer_ids, dim=0),
                }
                save_file(tensors, str(chunk_file))
                logger.info("Đã lưu chunk %d (len=%d) ra %s", chunk_idx, bucket_len, chunk_file)
                chunk_idx += 1
                buffer_hidden.clear()
                buffer_ids.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Lưu buffer còn lại của bucket này trước khi sang bucket mới
        if buffer_hidden:
            chunk_file = out_dir / f"features_chunk_{chunk_idx:04d}.safetensors"
            tensors = {
                "hidden_states": torch.cat(buffer_hidden, dim=0),
                "input_ids": torch.cat(buffer_ids, dim=0),
            }
            save_file(tensors, str(chunk_file))
            logger.info("Đã lưu chunk %d (len=%d) ra %s", chunk_idx, bucket_len, chunk_file)
            chunk_idx += 1
            buffer_hidden.clear()
            buffer_ids.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("Hoàn tất trích xuất Hidden States!")


if __name__ == "__main__":
    run_extraction()
