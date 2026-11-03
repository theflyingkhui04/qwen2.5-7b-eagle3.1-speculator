"""
Script package_and_upload.py
Xác thực tính toàn vẹn của thư mục checkpoint sinh ra từ SpecForge (config.json, model.safetensors),
đính kèm Model Card (README.md) và tự động tải lên Hugging Face Hub.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_hf_config(base_config: Dict[str, Any], precision: str = "bfloat16") -> Dict[str, Any]:
    """
    Xây dựng config.json chuẩn Hugging Face cho vLLM / SGLang Speculative Decoding.
    """
    config = dict(base_config)
    if config.get("architectures") != ["LlamaForCausalLMEagle3"]:
        raise ValueError("Export with native specforge export before packaging")
    if precision != "bfloat16" or config.get("dtype", config.get("torch_dtype")) != precision:
        raise ValueError("Packaging must preserve native BF16 weights and config")
    return config


def verify_package_integrity(export_dir: Any) -> bool:
    """
    Kiểm định gói xuất xưởng: bắt buộc phải có config.json và model.safetensors.
    """
    exp_path = Path(export_dir)
    if not exp_path.exists():
        return False

    config_file = exp_path / "config.json"
    weights_file = exp_path / "model.safetensors"

    if not config_file.exists() or not weights_file.exists():
        return False

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        if archs != ["LlamaForCausalLMEagle3"] or cfg.get("dtype", cfg.get("torch_dtype")) != "bfloat16":
            return False
    except Exception:
        return False

    return True


def attach_model_card(export_dir: str, model_card_path: str = "MODEL_CARD.md") -> bool:
    """
    Copy MODEL_CARD.md thành README.md trong thư mục checkpoint để hiển thị trên Hugging Face.
    """
    exp_path = Path(export_dir)
    exp_path.mkdir(parents=True, exist_ok=True)
    target = exp_path / "README.md"
    card_source = Path(model_card_path)

    if card_source.exists():
        shutil.copyfile(card_source, target)
        logger.info("Đã đính kèm Model Card vào: %s", target)
        return True
    else:
        logger.warning("Không tìm thấy %s để đính kèm.", model_card_path)
        return False


def parse_export_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh cho quy trình đóng gói và upload."""
    parser = argparse.ArgumentParser(description="Xác thực và tải checkpoint SpecForge lên Hugging Face")
    parser.add_argument(
        "--output-dir",
        "--checkpoint-dir",
        dest="output_dir",
        type=str,
        default="checkpoints/qwen2.5-7b-eagle3.1",
        help="Thư mục checkpoint cần kiểm tra và đóng gói",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="theflyingkhui04/qwen2.5-7b-eagle3.1-speculator",
        help="Repo ID trên Hugging Face Hub",
    )
    parser.add_argument(
        "--model-card",
        type=str,
        default="MODEL_CARD.md",
        help="Đường dẫn file Model Card markdown",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["bfloat16"],
        help="Định dạng số thực",
    )
    parser.add_argument("--upload", action="store_true", help="Bật cờ upload lên Hugging Face Hub")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm mà không tải lên Hugging Face thật")
    return parser.parse_args(args)


def upload_to_huggingface(
    export_dir: str,
    repo_id: str,
    token: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """
    Tải thư mục checkpoint lên Hugging Face Hub.
    Hỗ trợ chế độ dry-run để kiểm thử cục bộ an toàn.
    """
    if dry_run:
        logger.info("[DRY RUN] Bỏ qua bước upload thực tế lên Hugging Face repo: %s", repo_id)
        return True

    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("Cần cài đặt thư viện huggingface_hub để upload.")
        return False

    hf_token = token or os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("Không tìm thấy HF_TOKEN! Bỏ qua bước tải lên Hugging Face.")
        return False

    logger.info("Đang tải toàn bộ thư mục %s lên repo %s...", export_dir, repo_id)
    try:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=export_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="feat: upload EAGLE-3.1 draft head weights trained via SpecForge",
        )
        logger.info("Đã upload thành công lên: https://huggingface.co/%s", repo_id)
        return True
    except Exception as e:
        logger.error("Lỗi khi tải lên Hugging Face Hub: %s", e)
        return False


def main():
    args = parse_export_args()
    if not verify_package_integrity(args.output_dir):
        logger.error("Thư mục %s không hợp lệ hoặc thiếu checkpoint!", args.output_dir)
        sys.exit(1)

    attach_model_card(args.output_dir, args.model_card)

    if args.upload or args.dry_run:
        success = upload_to_huggingface(
            export_dir=args.output_dir,
            repo_id=args.repo_id,
            dry_run=args.dry_run,
        )
        if not success and not args.dry_run:
            sys.exit(1)


if __name__ == "__main__":
    main()
