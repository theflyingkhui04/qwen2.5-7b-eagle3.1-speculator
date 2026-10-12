"""
Script train.py
Vòng lặp huấn luyện chính cho đầu nháp suy đoán EAGLE-3.1 (Qwen2.5-7B).
Tối ưu hóa bằng AdamW, Cosine Annealing decay, Gradient Accumulation.
Hỗ trợ tự động phục hồi Checkpoint và thu gom rác bộ nhớ GPU (OOM Guard).
"""

import argparse
import glob
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from training.loss import EagleDualLoss
from training.model import Eagle3SpeculatorConfig, Eagle3SpeculatorHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_training_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện đầu nháp EAGLE-3.1")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--resume", action="store_true", help="Tự động tiếp tục từ checkpoint gần nhất")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Đường dẫn checkpoint cụ thể cần nạp")
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epoch huấn luyện")
    parser.add_argument("--batch-size", type=int, default=None, help="Ghi đè batch size")
    return parser.parse_args(args)


def load_training_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_latest_checkpoint(checkpoint_dir: Any) -> Optional[Path]:
    """Tìm checkpoint có số epoch lớn nhất trong thư mục."""
    chk_dir = Path(checkpoint_dir)
    if not chk_dir.exists():
        return None

    chk_files = list(chk_dir.glob("checkpoint_epoch_*.pt"))
    if not chk_files:
        return None

    def extract_epoch(p: Path) -> int:
        match = re.search(r"checkpoint_epoch_(\d+)\.pt", p.name)
        return int(match.group(1)) if match else -1

    chk_files.sort(key=extract_epoch)
    return chk_files[-1]


def save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    step: int,
    output_dir: str,
) -> str:
    """Lưu checkpoint tiến trình huấn luyện."""
    try:
        import torch
    except ImportError:
        return ""

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"checkpoint_epoch_{epoch}.pt"

    state = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    torch.save(state, str(file_path))
    logger.info("Đã lưu checkpoint epoch %d vào: %s", epoch, str(file_path))
    return str(file_path)


def run_training(
    config: Dict[str, Any],
    resume: bool = True,
    checkpoint_path: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    on_epoch_end: Optional[Any] = None,
):
    try:
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from safetensors.torch import load_file
    except ImportError:
        logger.error("torch và safetensors là bắt buộc để huấn luyện trên GPU.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    logger.info("Khởi tạo môi trường huấn luyện trên thiết bị: %s (dtype=%s)", device, dtype)

    # 1. Khởi tạo mô hình
    model_cfg_path = config["model"]["config_path"]
    model_config = Eagle3SpeculatorConfig.from_json_file(model_cfg_path)
    model = Eagle3SpeculatorHead(model_config).to(device=device, dtype=dtype)

    # 2. Loss & Optimizer
    loss_cfg = config.get("loss", {})
    loss_fn = EagleDualLoss(
        smooth_l1_weight=loss_cfg.get("smooth_l1_weight", 0.5),
        ce_weight=loss_cfg.get("ce_weight", 1.0),
    )

    train_cfg = config["training"]
    total_epochs = epochs or train_cfg.get("epochs", 20)
    effective_batch_size = batch_size or train_cfg.get("batch_size", 16)
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 2)
    lr = float(train_cfg.get("learning_rate", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    betas = (float(train_cfg.get("adam_beta1", 0.9)), float(train_cfg.get("adam_beta2", 0.95)))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=float(train_cfg.get("min_learning_rate", 1e-5)))

    # 3. Phục hồi checkpoint nếu có
    start_epoch = 1
    global_step = 0
    checkpoint_to_load = checkpoint_path
    if not checkpoint_to_load and resume:
        checkpoint_to_load = get_latest_checkpoint(train_cfg.get("output_dir", "checkpoints"))

    if checkpoint_to_load:
        logger.info("Đang nạp checkpoint từ: %s", checkpoint_to_load)
        chk = torch.load(checkpoint_to_load, map_location=device)
        model.load_state_dict(chk["model_state_dict"])
        if "optimizer_state_dict" in chk and chk["optimizer_state_dict"]:
            optimizer.load_state_dict(chk["optimizer_state_dict"])
        if "scheduler_state_dict" in chk and chk["scheduler_state_dict"]:
            scheduler.load_state_dict(chk["scheduler_state_dict"])
        start_epoch = chk.get("epoch", 0) + 1
        global_step = chk.get("step", 0)
        logger.info("Tiếp tục huấn luyện từ epoch %d, step %d", start_epoch, global_step)

    # 4. Tìm kiếm các file feature chunks
    features_dir = Path(config["data"]["features_dir"])
    feature_files = sorted(features_dir.glob("features_chunk_*.safetensors"))
    if not feature_files:
        logger.warning("Không tìm thấy file feature nào trong %s. Hãy chạy extraction trước!", features_dir)
        return

    logger.info("Tìm thấy %d file feature chunk. Bắt đầu huấn luyện %d epochs...", len(feature_files), total_epochs)

    # 5. Khởi tạo Weights & Biases nếu có API key
    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb_entity = os.environ.get("WANDB_ENTITY", None)
            init_kwargs = {
                "project": "qwen2.5-eagle3.1-speculator",
                "name": f"eagle3.1-qwen2.5-7b-{total_epochs}ep",
                "config": config,
            }
            if wandb_entity:
                init_kwargs["entity"] = wandb_entity
            wandb_run = wandb.init(**init_kwargs)
            logger.info("Đã kết nối W&B thành công: %s", wandb_run.url)
        except Exception as e:
            logger.warning("Không thể khởi tạo W&B (%s), tiếp tục huấn luyện với logger tiêu chuẩn.", e)

    output_dir = train_cfg.get("output_dir", "checkpoints")
    checkpoint_interval = train_cfg.get("checkpoint_interval_epochs", 1)

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        total_ce_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0

        for f_path in feature_files:
            data = load_file(str(f_path))
            hidden_states = data["hidden_states"].to(device=device, dtype=dtype)
            input_ids = data["input_ids"].to(device)
            num_samples = hidden_states.size(0)

            for b in range(0, num_samples, effective_batch_size):
                b_hidden = hidden_states[b : b + effective_batch_size]
                b_ids = input_ids[b : b + effective_batch_size]

                # Chuẩn bị input/target (đảm bảo contiguous memory layout)
                token_embeddings = b_hidden.clone()
                target_tokens = b_ids[:, 1:].contiguous()
                target_hidden = b_hidden[:, 1:, :].contiguous()
                input_hidden = b_hidden[:, :-1, :].contiguous()
                input_embeds = token_embeddings[:, :-1, :].contiguous()

                logits, pred_hidden = model(input_hidden, input_embeds)

                loss, metrics = loss_fn(logits, target_tokens, pred_hidden, target_hidden)
                loss = loss / grad_accum_steps
                loss.backward()

                if (global_step + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_epoch_loss += metrics["total_loss"]
                total_ce_loss += metrics["ce_loss"]
                total_reg_loss += metrics["reg_loss"]
                num_batches += 1
                global_step += 1

                # Log step định kỳ lên W&B
                if wandb_run and global_step % 20 == 0:
                    wandb.log({
                        "step": global_step,
                        "step/total_loss": metrics["total_loss"],
                        "step/ce_loss": metrics["ce_loss"],
                        "step/reg_loss": metrics["reg_loss"],
                        "step/learning_rate": scheduler.get_last_lr()[0],
                    })

        scheduler.step()
        avg_loss = total_epoch_loss / max(num_batches, 1)
        avg_ce = total_ce_loss / max(num_batches, 1)
        avg_reg = total_reg_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch [%d/%d] - Avg Loss: %.4f (CE: %.4f, SmoothL1: %.4f) - LR: %.6f",
            epoch, total_epochs, avg_loss, avg_ce, avg_reg, current_lr,
        )

        if wandb_run:
            wandb.log({
                "epoch": epoch,
                "epoch/total_loss": avg_loss,
                "epoch/ce_loss": avg_ce,
                "epoch/reg_loss": avg_reg,
                "epoch/learning_rate": current_lr,
            })

        # OOM Guard: Thu gom rác bộ nhớ sau mỗi epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Lưu checkpoint theo chu kỳ (mỗi epoch) và kích hoạt on_epoch_end
        if epoch % checkpoint_interval == 0 or epoch == total_epochs:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, output_dir)
            if on_epoch_end:
                try:
                    on_epoch_end(epoch)
                except Exception as e:
                    logger.warning("Lỗi trong on_epoch_end callback: %s", e)

    if wandb_run:
        wandb.finish()

    logger.info("Quá trình huấn luyện đã hoàn thành thành công!")


def main():
    args = parse_training_args()
    config = load_training_config(args.config)
    run_training(
        config=config,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
