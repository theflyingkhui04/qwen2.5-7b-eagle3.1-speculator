"""
Modal.com Serverless Pipeline for Qwen2.5-7B EAGLE-3.1 Speculator Training
Framework: sgl-project/SpecForge
Quy trình 3 bước tinh gọn:
- Nạp tự động configs/ và data/processed/ (ShareGPT JSONL)
- Huấn luyện unroll autoregressive với FlashAttention trên GPU NVIDIA A100-80GB
- Checkpoint tự động commit vào Modal Persistent Volume (/vol)
- Tự động đồng bộ Weights & Biases (WandB) và upload Hugging Face Hub
"""

import os
import sys
from pathlib import Path
from typing import Optional

import modal

# Khắc phục lỗi mã hóa ký tự Unicode charmap trên console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 1. Định nghĩa Persistent Storage Volume trên Modal Cloud
volume = modal.Volume.from_name("qwen-eagle3-volume", create_if_missing=True)

# 2. Xây dựng Container Image với SpecForge, SGLang và PyTorch
app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "safetensors>=0.4.4",
        "datasets>=2.20.0",
        "pyyaml>=6.0.2",
        "tqdm>=4.66.5",
        "huggingface_hub>=0.24.0",
        "wandb>=0.16.0",
        "sglang[all]",
        "git+https://github.com/sgl-project/SpecForge.git",
    )
    .add_local_python_source("training", "data", "export")
    .add_local_dir("configs", remote_path="/root/configs")
)

# Nạp dữ liệu processed và MODEL_CARD.md nếu có sẵn ở local
local_processed_dir = Path("data/processed")
if local_processed_dir.exists():
    app_image = app_image.add_local_dir(
        str(local_processed_dir),
        remote_path="/root/data/processed",
    )

local_model_card = Path("MODEL_CARD.md")
if local_model_card.exists():
    app_image = app_image.add_local_file(
        str(local_model_card),
        remote_path="/root/MODEL_CARD.md",
    )

app = modal.App("qwen2-5-eagle3.1-speculator", image=app_image)


# ------------------------------------------------------------------------------
# Function 1: Huấn luyện với SpecForge trên NVIDIA A100-80GB
# ------------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    timeout=21600,  # 6.0 giờ tối đa (khớp Cost Ceiling Guard)
    volumes={"/vol": volume},
    secrets=[
        modal.Secret.from_name("my-huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_specforge(
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    resume: bool = False,
    test_mode: bool = False,
    dry_run: bool = False,
) -> str:
    import yaml
    from training.train import run_specforge_training

    print("=" * 70)
    print(">>> [MODAL] Bắt đầu khởi chạy SpecForge Training trên NVIDIA A100-80GB")
    print("=" * 70)

    try:
        volume.reload()
    except Exception:
        pass

    config_path = "/root/configs/train_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Định tuyến đường dẫn đầu ra và dữ liệu trên Modal Volume
    vol_output_dir = Path("/vol/checkpoints/qwen2.5-7b-eagle3.1")
    vol_output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(vol_output_dir)

    if Path("/root/data/processed/train.jsonl").exists():
        cfg["train_data_path"] = "/root/data/processed/train.jsonl"
    if Path("/root/data/processed/val.jsonl").exists():
        cfg["val_data_path"] = "/root/data/processed/val.jsonl"

    if epochs is not None:
        cfg["num_epochs"] = epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size

    if test_mode:
        print("[TEST MODE] Cấu hình chạy nhanh 1 epoch, unroll 3 bước để kiểm thử môi trường...")
        cfg["num_epochs"] = 1
        cfg["ttt_length"] = 3

    # Lưu tạm cấu hình đã cập nhật
    temp_config = "/tmp/modal_train_config.yaml"
    with open(temp_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    cmd = run_specforge_training(
        config_path=temp_config,
        num_gpus=1,
        resume=resume,
        dry_run=dry_run,
    )

    volume.commit()
    print(">>> [MODAL] Đã commit toàn bộ checkpoint vào Modal Volume (/vol) thành công!")
    return str(vol_output_dir)


# ------------------------------------------------------------------------------
# Function 2: Đóng gói và upload lên Hugging Face Hub (Chạy CPU container)
# ------------------------------------------------------------------------------
@app.function(
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret", required_keys=["HF_TOKEN"])],
    timeout=1800,
)
def export_and_upload(
    repo_id: str = "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator",
    dry_run: bool = False,
) -> str:
    from export.package_and_upload import (
        attach_model_card,
        upload_to_huggingface,
        verify_package_integrity,
    )

    try:
        volume.reload()
    except Exception:
        pass

    export_dir = Path("/vol/checkpoints/qwen2.5-7b-eagle3.1")
    best_dir = export_dir / "best"
    target_dir = str(best_dir if best_dir.exists() else export_dir)

    print(f">>> [MODAL] Xác thực tính toàn vẹn của checkpoint tại: {target_dir}")
    if not verify_package_integrity(target_dir):
        print(f"[CẢNH BÁO] Chưa tìm thấy checkpoint đầy đủ tại {target_dir}.")

    # Đính kèm Model Card nếu có
    model_card = "/root/MODEL_CARD.md"
    if Path(model_card).exists():
        attach_model_card(target_dir, model_card)

    print(f">>> [MODAL] Tải checkpoint lên Hugging Face Hub: {repo_id}")
    upload_to_huggingface(
        export_dir=target_dir,
        repo_id=repo_id,
        dry_run=dry_run,
    )

    volume.commit()
    print(f">>> [MODAL] Hoàn tất upload lên: https://huggingface.co/{repo_id}")
    return repo_id


# ------------------------------------------------------------------------------
# Entrypoint chạy từ Terminal: modal run modal_pipeline.py [--step test|train|export|all]
# ------------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    step: str = "train",
    repo_id: str = "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    resume: bool = False,
    dry_run: bool = False,
):
    print("=" * 70)
    print(">>> [MODAL] Khởi chạy EAGLE-3.1 SpecForge Pipeline trên Modal.com")
    print(f"Bước thực hiện (Step): {step.upper()} | Resume: {resume} | Dry-run: {dry_run}")
    print("=" * 70)

    if step == "test":
        print(">>> [TEST RUN] Kiểm thử nhanh môi trường huấn luyện SpecForge (1 epoch)...")
        train_specforge.remote(test_mode=True, dry_run=dry_run)
        print(">>> [SUCCESS] Test run hoàn tất thành công!")

    elif step == "train":
        print(">>> [STEP: TRAIN] Huấn luyện đầu nháp EAGLE-3.1 với SpecForge trên A100-80GB...")
        train_specforge.remote(
            epochs=epochs,
            batch_size=batch_size,
            resume=resume,
            dry_run=dry_run,
        )

    elif step == "export":
        print(f">>> [STEP: EXPORT] Đóng gói và upload lên Hugging Face ({repo_id})...")
        export_and_upload.remote(repo_id=repo_id, dry_run=dry_run)

    elif step == "all":
        print(">>> [STEP: ALL] Chạy liên tục Huấn luyện -> Đóng gói & Upload...")
        train_specforge.remote(
            epochs=epochs,
            batch_size=batch_size,
            resume=resume,
            dry_run=dry_run,
        )
        export_and_upload.remote(repo_id=repo_id, dry_run=dry_run)
        print(">>> [SUCCESS] HOÀN TẤT TOÀN BỘ PIPELINE HUẤN LUYỆN TRÊN MODAL.COM!")

    else:
        print(f"❌ Lỗi: Bước không hợp lệ '{step}'. Vui lòng chọn: test, train, export, all.")
