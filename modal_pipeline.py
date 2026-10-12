"""
Modal.com Serverless Pipeline for Qwen2.5-7B EAGLE-3.1 Speculator Training
Hỗ trợ:
- Tự động mount mã nguồn và dữ liệu hybrid_dataset.json
- Tích hợp Persistent Volume (/vol) lưu features và checkpoints
- Cung cấp GPU A100-80GB serverless (tính tiền theo giây thực tế)
- Chế độ --step test (chạy thử 50 mẫu trong 2 phút) và --step all (full pipeline)
"""

import os
import sys
from pathlib import Path
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

# 2. Xây dựng Container Image với đầy đủ thư viện AI/ML cần thiết
app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "safetensors>=0.4.2",
        "datasets>=2.18.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "huggingface_hub>=0.21.0",
        "wandb>=0.16.0",
    )
    # Tự động nạp mã nguồn từ local vào cloud container
    .add_local_python_source("training", "extraction", "data", "export")
    .add_local_dir("configs", remote_path="/root/configs")
)

# Thêm hybrid_dataset.json nếu tồn tại ở local
local_data_path = Path("data/processed/hybrid_dataset.json")
if local_data_path.exists():
    app_image = app_image.add_local_file(
        str(local_data_path),
        remote_path="/root/data/processed/hybrid_dataset.json",
    )

app = modal.App("qwen2-5-eagle3.1-speculator", image=app_image)


# ------------------------------------------------------------------------------
# Function 1: Feature Extraction (Tầng 28 Target Model trên A100-80GB)
# ------------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    timeout=14400,  # 4 giờ tối đa
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
def extract_features(
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    target_layer: int = 28,
    chunk_size: int = 5000,
    batch_size: int = 8,
    test_mode: bool = False,
    force: bool = False,
):
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import save_file
    from extraction.extract_hidden_states import bucket_sequences

    dataset_path = "/root/data/processed/hybrid_dataset.json"
    features_dir = Path("/vol/features")
    features_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Không tìm thấy dataset tại {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if test_mode:
        existing_chunks = list(features_dir.glob("features_chunk_*.safetensors"))
        if existing_chunks and not force:
            print(f"[TEST MODE] Da tim thay {len(existing_chunks)} chunk features co san trong {features_dir}. Bo qua trich xuat de tiet kiem thoi gian va chi phi GPU!")
            return str(features_dir)
        print("[TEST MODE] Chi chay tren 50 mau dau tien...")
        dataset = dataset[:50]
        chunk_size = 50

    print(f"Bat dau trich xuat features tu {len(dataset)} mau...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiet bi kha dung: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    buckets = bucket_sequences(dataset)
    chunk_idx = 0
    buffer_hidden = []
    buffer_ids = []

    for bucket_len, samples in buckets.items():
        if not samples:
            continue
        print(f"Dang xu ly bucket len <= {bucket_len} ({len(samples)} mau)...")
        buffer_hidden = []
        buffer_ids = []
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i : i + batch_size]
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
                layer_hidden = outputs.hidden_states[target_layer].cpu().to(torch.bfloat16)
                input_ids = inputs["input_ids"].cpu()

            buffer_hidden.append(layer_hidden)
            buffer_ids.append(input_ids)

            if len(buffer_hidden) * batch_size >= chunk_size:
                chunk_file = features_dir / f"features_chunk_{chunk_idx:04d}.safetensors"
                save_file(
                    {
                        "hidden_states": torch.cat(buffer_hidden, dim=0),
                        "input_ids": torch.cat(buffer_ids, dim=0),
                    },
                    str(chunk_file),
                )
                print(f"Da luu chunk {chunk_idx} (len={bucket_len}) -> {chunk_file}")
                chunk_idx += 1
                buffer_hidden.clear()
                buffer_ids.clear()
                torch.cuda.empty_cache()

        # Lưu phần còn lại của bucket hiện tại (đảm bảo không trộn lẫn các bucket khác độ dài)
        if buffer_hidden:
            chunk_file = features_dir / f"features_chunk_{chunk_idx:04d}.safetensors"
            save_file(
                {
                    "hidden_states": torch.cat(buffer_hidden, dim=0),
                    "input_ids": torch.cat(buffer_ids, dim=0),
                },
                str(chunk_file),
            )
            print(f"Da luu chunk {chunk_idx} (len={bucket_len}) -> {chunk_file}")
            chunk_idx += 1
            buffer_hidden.clear()
            buffer_ids.clear()
            torch.cuda.empty_cache()

    # Commit thay đổi vào volume
    volume.commit()
    print("Hoan tat trich xuat features va da commit vao Modal Volume!")
    return str(features_dir)


# ------------------------------------------------------------------------------
# Function 2: Train EAGLE-3.1 Speculator Head (trên A100-80GB)
# ------------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    timeout=14400,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_head(
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 3e-4,
    test_mode: bool = False,
):
    import yaml
    from pathlib import Path
    from training.train import run_training

    config_path = "/root/configs/train_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["config_path"] = "/root/configs/eagle3_qwen2.5_7b.json"
    cfg["data"]["features_dir"] = "/vol/features"
    cfg["training"]["output_dir"] = "/vol/checkpoints"
    cfg["training"]["epochs"] = 1 if test_mode else epochs
    cfg["training"]["batch_size"] = 4 if test_mode else batch_size
    cfg["training"]["learning_rate"] = lr

    checkpoint_dir = Path("/vol/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_finished(ep: int):
        volume.commit()
        print(f">>> [CHECKPOINT] Da commit an toan checkpoint_epoch_{ep}.pt vao Modal Volume!")

    print(f"Bat dau huan luyen EAGLE-3.1 ({cfg['training']['epochs']} epochs)...")
    run_training(cfg, resume=True, on_epoch_end=on_epoch_finished)

    volume.commit()
    print("Hoan tat huan luyen va da commit toan bo checkpoint vao Modal Volume!")
    return str(checkpoint_dir)


# ------------------------------------------------------------------------------
# Function 3: Package & Upload to Hugging Face Hub (Chạy trên CPU)
# ------------------------------------------------------------------------------
@app.function(
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret", required_keys=["HF_TOKEN"])],
    timeout=1800,
)
def export_and_upload(
    repo_id: str = "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator",
    target_epoch: int = 20,
    test_mode: bool = False,
):
    import os
    from pathlib import Path
    from export.package_and_upload import package_model, upload_to_huggingface

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Chưa cấu hình HF_TOKEN trong Modal Secret!")

    export_dir = "/vol/output/eagle3_qwen2.5_7b"
    chk_epoch = 1 if test_mode else target_epoch
    checkpoint_path = Path(f"/vol/checkpoints/checkpoint_epoch_{chk_epoch}.pt")
    if not checkpoint_path.exists():
        checkpoints = list(Path("/vol/checkpoints").glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            raise FileNotFoundError("Không tìm thấy checkpoint nào trong /vol/checkpoints!")
        checkpoint_path = sorted(checkpoints)[-1]
        print(f"Su dung checkpoint moi nhat: {checkpoint_path}")

    print("Bat dau dong goi model sang float16 chuan vLLM...")
    package_model(
        checkpoint_path=str(checkpoint_path),
        config_path="/root/configs/eagle3_qwen2.5_7b.json",
        output_dir=export_dir,
        precision="float16",
    )

    print(f"Bat dau tai len Hugging Face Hub: {repo_id}...")
    upload_to_huggingface(
        export_dir=export_dir,
        repo_id=repo_id,
        token=hf_token,
    )
    volume.commit()
    print(f"Da upload thanh cong len https://huggingface.co/{repo_id}")
    return repo_id


# ------------------------------------------------------------------------------
# Entrypoint chạy từ Terminal: modal run modal_pipeline.py [--step test|extract|train|export|all]
# ------------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    step: str = "test",
    repo_id: str = "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator",
    epochs: int = 20,
    force: bool = False,
):
    print("=" * 70)
    print(f"[START] Khoi chay EAGLE-3.1 Serverless Pipeline tren Modal.com")
    print(f"Mode/Step: {step.upper()}")
    print("=" * 70)

    if step == "test":
        print(">>> [TEST RUN] Kiem thu toan bo pipeline khep kin (50 mau, 1 epoch)...")
        extract_features.remote(test_mode=True, force=force)
        train_head.remote(test_mode=True)
        print(">>> [SUCCESS] Test Run thanh cong hoan hao! San sang cho Full Pipeline.")

    elif step == "extract":
        print(">>> [BUOC 1/3] Trich xuat Hidden States tang 28 tren A100-80GB...")
        extract_features.remote(force=force)

    elif step == "train":
        print(f">>> [BUOC 2/3] Huan luyen dau nhap EAGLE-3.1 ({epochs} Epochs)...")
        train_head.remote(epochs=epochs)

    elif step == "export":
        print(f">>> [BUOC 3/3] Dong goi float16 va upload len Hugging Face ({repo_id})...")
        export_and_upload.remote(repo_id=repo_id)

    elif step == "all":
        print(">>> [FULL PIPELINE] Chay tuan tu Trich xuat -> Huan luyen -> Upload...")
        extract_features.remote()
        train_head.remote(epochs=epochs)
        export_and_upload.remote(repo_id=repo_id, target_epoch=epochs)
        print(">>> [SUCCESS] HOAN TAT TOAN BO PIPELINE HUAN LUYEN TREN MODAL.COM!")

    else:
        print(f"[ERROR] Buoc khong hop le: '{step}'. Vui long chon trong: test, extract, train, export, all.")
