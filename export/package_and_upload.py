"""
Script package_and_upload.py
Đóng gói trọng số đầu nháp EAGLE-3.1 sang định dạng model.safetensors và config.json
chuẩn vLLM ("EagleQwen2ForCausalLM").
Hỗ trợ xuất precision float16 tương thích 100% với kiến trúc Turing trên Tesla T4.
Tự động đẩy lên Hugging Face Hub (duckhoi/qwen2.5-7b-eagle3.1-speculator).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_hf_config(base_config: Dict[str, Any], precision: str = "float16") -> Dict[str, Any]:
    """
    Xây dựng config.json chuẩn Hugging Face cho vLLM Speculative Decoding.
    """
    config = dict(base_config)
    config["architectures"] = ["Eagle3_1Qwen2ForCausalLM", "EagleQwen2ForCausalLM"]
    config["model_type"] = "eagle_qwen2"
    config["eagle_version"] = "3.1"
    config["torch_dtype"] = precision
    config["hidden_size"] = config.get("hidden_size", 3584)
    config["num_hidden_layers"] = config.get("num_hidden_layers", 1)
    config["num_attention_heads"] = config.get("num_attention_heads", 28)
    config["num_key_value_heads"] = config.get("num_key_value_heads", 4)
    config["vocab_size"] = config.get("vocab_size", 152064)
    config["target_layer"] = config.get("target_layer", 28)
    config["use_fc_norm"] = config.get("use_fc_norm", True)
    config["use_post_norm_feedback"] = config.get("use_post_norm_feedback", True)
    return config


def verify_package_integrity(export_dir: Any) -> bool:
    """
    Kiểm định gói xuất: bắt buộc phải có config.json và model.safetensors.
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
        if "EagleQwen2ForCausalLM" not in cfg.get("architectures", []):
            return False
    except Exception:
        return False

    return True


def parse_export_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đóng gói và tải lên Hugging Face")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/checkpoint_epoch_20.pt", help="Đường dẫn file .pt")
    parser.add_argument("--config-path", type=str, default="configs/eagle3_qwen2.5_7b.json", help="Đường dẫn file cấu hình gốc")
    parser.add_argument("--output-dir", type=str, default="output/eagle3_qwen2.5_7b", help="Thư mục xuất thành phẩm")
    parser.add_argument("--repo-id", type=str, default="duckhoi/qwen2.5-7b-eagle3.1-speculator", help="Repo ID trên Hugging Face")
    parser.add_argument("--precision", type=str, default="float16", choices=["float16", "bfloat16"], help="Định dạng số thực (float16 tối ưu T4)")
    parser.add_argument("--upload", action="store_true", help="Bật cờ upload lên Hugging Face Hub")
    return parser.parse_args(args)


def package_model(checkpoint_path: str, config_path: str, output_dir: str, precision: str = "float16") -> str:
    """
    Chuyển đổi trọng số từ PyTorch .pt sang model.safetensors và xuất config.json.
    """
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        logger.error("torch và safetensors là bắt buộc để đóng gói trọng số.")
        sys.exit(1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Nạp và xử lý cấu hình
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    hf_cfg = build_hf_config(raw_cfg, precision=precision)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(hf_cfg, f, indent=2, ensure_ascii=False)

    # 2. Chuyển đổi trọng số
    logger.info("Đang đọc checkpoint từ: %s", checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu")
    weights = state.get("model_state_dict", state)

    target_dtype = torch.float16 if precision == "float16" else torch.bfloat16
    converted_weights = {}
    for k, v in weights.items():
        if isinstance(v, torch.Tensor):
            converted_weights[k] = v.to(target_dtype)
        else:
            converted_weights[k] = v

    weights_file = out_dir / "model.safetensors"
    save_file(converted_weights, str(weights_file))
    logger.info("Đã lưu trọng số safetensors (%s) ra: %s", precision, str(weights_file))

    # 3. Kiểm định tính toàn vẹn
    if not verify_package_integrity(out_dir):
        raise RuntimeError("Gói đóng gói không vượt qua bài kiểm tra toàn vẹn!")

    logger.info("Gói xuất bản đã sẵn sàng tại: %s", str(out_dir))
    return str(out_dir)


def upload_to_huggingface(export_dir: str, repo_id: str, token: Optional[str] = None):
    """
    Tải thư mục checkpoint lên Hugging Face Hub.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("Cần cài đặt thư viện huggingface_hub để upload.")
        sys.exit(1)

    hf_token = token or os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("Không tìm thấy HF_TOKEN! Bỏ qua bước tải lên Hugging Face.")
        return

    logger.info("Đang tải toàn bộ thư mục %s lên repo %s...", export_dir, repo_id)
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=export_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="feat: upload EAGLE-3.1 draft head weights for Qwen2.5-7B-Instruct",
    )
    logger.info("Đã upload thành công lên: https://huggingface.co/%s", repo_id)


def main():
    args = parse_export_args()
    packaged_dir = package_model(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        precision=args.precision,
    )

    if args.upload:
        upload_to_huggingface(export_dir=packaged_dir, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
