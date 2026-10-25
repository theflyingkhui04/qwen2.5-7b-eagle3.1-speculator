"""
Script train.py
Launcher nạp cấu hình configs/train_config.yaml và khởi chạy tiến trình huấn luyện
SpecForge (sgl-project/SpecForge) cho đầu nháp EAGLE-3.1 (Qwen2.5-7B).
Hỗ trợ lưu checkpoint định kỳ, đồng bộ Weights & Biases (WandB) và tự động resume.
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_training_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh cho launcher."""
    parser = argparse.ArgumentParser(description="Khởi chạy SpecForge Training Launcher cho EAGLE-3.1")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--gpus", type=int, default=1, help="Số lượng GPU sử dụng (nproc_per_node)")
    parser.add_argument("--resume", action="store_true", help="Tự động tiếp tục từ checkpoint trong output_dir")
    parser.add_argument("--resume-best", action="store_true", help="Ưu tiên tiếp tục từ best checkpoint thay vì latest")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Đường dẫn checkpoint cụ thể để resume/warm-start")
    parser.add_argument("--save-total-limit", type=int, default=None, help="Số lượng checkpoint tối đa lưu trên đĩa (giữ best + N latest)")
    parser.add_argument("--wandb", action="store_true", help="Kích hoạt ghi log lên Weights & Biases")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ sinh câu lệnh mà không thực thi subprocess")
    return parser.parse_args(args)


def load_training_config(config_path: str) -> Dict[str, Any]:
    """Nạp file cấu hình YAML huấn luyện."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_best_checkpoint(checkpoint_dir: Any) -> Optional[Path]:
    """
    Tìm checkpoint tốt nhất (dựa trên thư mục best/ hoặc file model_best/checkpoint-best).
    """
    chk_dir = Path(checkpoint_dir)
    if not chk_dir.exists():
        return None

    # 1. Kiểm tra các thư mục best ưu tiên
    for best_name in ["best", "best_checkpoint", "checkpoint-best"]:
        best_path = chk_dir / best_name
        if best_path.exists():
            return best_path

    # 2. Kiểm tra file đánh dấu metadata best_metric.json nếu có
    meta_file = chk_dir / "best_metric.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                b_path = meta.get("best_checkpoint_path")
                if b_path and Path(b_path).exists():
                    return Path(b_path)
        except Exception:
            pass

    return None


def find_latest_checkpoint(checkpoint_dir: Any) -> Optional[Path]:
    """
    Tìm checkpoint có số bước hoặc số epoch lớn nhất trong thư mục đầu ra.
    Hỗ trợ cả thư mục con (checkpoint-*) và file (.pt, .safetensors).
    """
    chk_dir = Path(checkpoint_dir)
    if not chk_dir.exists():
        return None

    candidates: List[Tuple[int, Path]] = []

    for p in chk_dir.iterdir():
        if p.name in ["best", "best_checkpoint", "checkpoint-best", "best_metric.json"]:
            continue
        if p.is_dir():
            match = re.search(r"(?:checkpoint|step|epoch)[-_](\d+)", p.name, re.IGNORECASE)
            if match:
                candidates.append((int(match.group(1)), p))
            elif (p / "model.safetensors").exists() or (p / "training_state.pt").exists():
                candidates.append((0, p))
        elif p.is_file():
            match = re.search(r"(?:checkpoint|step|epoch)[-_](\d+)\.(?:pt|safetensors|bin)", p.name, re.IGNORECASE)
            if match:
                candidates.append((int(match.group(1)), p))
            elif p.name in ["model.safetensors", "training_state.pt"]:
                candidates.append((0, p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def find_resume_checkpoint(checkpoint_dir: Any, prefer_best: bool = True) -> Optional[Path]:
    """
    Tìm checkpoint tối ưu để tiếp tục huấn luyện:
    - Nếu prefer_best=True: Ưu tiên tìm best_checkpoint trước, nếu không có mới lấy latest.
    - Nếu prefer_best=False: Lấy checkpoint có step/epoch cao nhất (latest).
    """
    if prefer_best:
        best = find_best_checkpoint(checkpoint_dir)
        if best:
            return best
    return find_latest_checkpoint(checkpoint_dir)


def prune_old_checkpoints(checkpoint_dir: Any, keep_limit: int = 2) -> List[Path]:
    """
    Dọn dẹp các checkpoint cũ dư thừa để tiết kiệm dung lượng đĩa và băng thông tải về.
    Luôn bảo vệ tuyệt đối thư mục 'best' và giữ lại tối đa `keep_limit` checkpoint mới nhất.
    """
    chk_dir = Path(checkpoint_dir)
    if not chk_dir.exists():
        return []

    candidates: List[Tuple[int, Path]] = []
    best_path = find_best_checkpoint(chk_dir)

    for p in chk_dir.iterdir():
        # Không bao giờ xóa thư mục best hoặc file metadata
        if best_path and (p == best_path or p.resolve() == best_path.resolve()):
            continue
        if p.name in ["best", "best_checkpoint", "checkpoint-best", "best_metric.json"]:
            continue

        if p.is_dir():
            match = re.search(r"(?:checkpoint|step|epoch)[-_](\d+)", p.name, re.IGNORECASE)
            if match:
                candidates.append((int(match.group(1)), p))
        elif p.is_file():
            match = re.search(r"(?:checkpoint|step|epoch)[-_](\d+)\.(?:pt|safetensors|bin)", p.name, re.IGNORECASE)
            if match:
                candidates.append((int(match.group(1)), p))

    candidates.sort(key=lambda x: x[0])

    deleted: List[Path] = []
    if len(candidates) > keep_limit:
        to_remove = candidates[:-keep_limit]
        for _, path_to_del in to_remove:
            try:
                if path_to_del.is_dir():
                    shutil.rmtree(path_to_del)
                else:
                    path_to_del.unlink()
                deleted.append(path_to_del)
                logger.info("Đã dọn dẹp checkpoint cũ dư thừa: %s", path_to_del.name)
            except Exception as e:
                logger.warning("Không thể xóa checkpoint cũ %s: %s", path_to_del, e)

    return deleted


def setup_wandb_metrics(cfg: Dict[str, Any]) -> bool:
    """
    Tự động cấu hình trước các metric, trục hoành độ (step_metric) và
    chỉ số tóm tắt (summary min/max) trên giao diện Weights & Biases bằng code.
    """
    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled = False
    if isinstance(wandb_cfg, dict) and wandb_cfg.get("enabled"):
        wandb_enabled = True
    elif cfg.get("report_to") == "wandb":
        wandb_enabled = True

    if not wandb_enabled:
        return False

    try:
        import os
        import wandb

        project = (wandb_cfg.get("project") if isinstance(wandb_cfg, dict) else None) or cfg.get(
            "wandb_project", "qwen2.5-7b-eagle3.1-speculator"
        )
        run_name = (wandb_cfg.get("run_name") if isinstance(wandb_cfg, dict) else None) or cfg.get(
            "wandb_run_name", "eagle3.1-qwen2.5-7b-train"
        )

        run = wandb.init(
            project=project,
            name=run_name,
            resume="allow",
            reinit=True,
        )

        # Định nghĩa trục thời gian chuẩn cho các nhóm metric
        run.define_metric("train/step")
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="train/step")
        run.define_metric("eval/*", step_metric="train/step")
        run.define_metric("val/*", step_metric="train/step")
        run.define_metric("acceptance/*", step_metric="train/step")

        # Định nghĩa các chỉ số tóm tắt tốt nhất (Best Summaries)
        run.define_metric("train/loss", summary="min")
        run.define_metric("train/ce_loss", summary="min")
        run.define_metric("train/reg_loss", summary="min")
        run.define_metric("eval/loss", summary="min")
        run.define_metric("val/dual_loss", summary="min")
        run.define_metric("acceptance/alpha_overall", summary="max")
        for k in range(1, 6):
            run.define_metric(f"acceptance/alpha_step_{k}", summary="max")

        # Kế thừa run ID cho tiến trình subprocess torchrun
        if hasattr(run, "id") and run.id:
            os.environ["WANDB_RUN_ID"] = str(run.id)
            os.environ["WANDB_RESUME"] = "allow"

        run.finish()
        logger.info("Đã cấu hình trước biểu đồ và metric trên WandB: %s (run: %s)", project, run_name)
        return True
    except Exception as e:
        logger.warning("Bỏ qua tiền cấu hình WandB metrics: %s", e)
        return False


def build_specforge_command(
    cfg: Dict[str, Any],
    num_gpus: int = 1,
    resume: bool = False,
    prefer_best: bool = True,
    checkpoint_path: Optional[str] = None,
    wandb_override: Optional[bool] = None,
) -> List[str]:
    """
    Chuyển đổi các siêu tham số từ train_config.yaml sang command line gọi torchrun specforge.train.
    Hỗ trợ cờ resume checkpoint (ưu tiên best_checkpoint), lưu định kỳ save-interval và Weights & Biases (WandB).
    """
    target_model_path = cfg.get("target_model_path") or cfg.get("model_path") or "Qwen/Qwen2.5-7B-Instruct"
    draft_model_config = cfg.get("draft_model_config") or cfg.get("draft_config") or "configs/eagle3.1_qwen2.5_7b.json"
    train_data_path = cfg.get("train_data_path") or cfg.get("data_path") or "data/processed/train.jsonl"
    output_dir = cfg.get("output_dir", "checkpoints/qwen2.5-7b-eagle3.1")
    num_epochs = cfg.get("num_epochs", 3)
    batch_size = cfg.get("batch_size") or cfg.get("micro_batch_size", 2)
    grad_accum_steps = cfg.get("gradient_accumulation_steps") or cfg.get("grad_accum_steps", 4)
    lr = cfg.get("learning_rate") or cfg.get("lr", 1.0e-4)
    max_seq_len = cfg.get("max_seq_len", 2048)
    ttt_length = cfg.get("ttt_length", 5)
    tp_size = cfg.get("tp_size", 1)

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "-m", "specforge.train",
        "--model-path", str(target_model_path),
        "--draft-config", str(draft_model_config),
        "--data-path", str(train_data_path),
        "--output-dir", str(output_dir),
        "--num-epochs", str(num_epochs),
        "--micro-batch-size", str(batch_size),
        "--grad-accum-steps", str(grad_accum_steps),
        "--lr", str(lr),
        "--max-seq-len", str(max_seq_len),
        "--ttt-length", str(ttt_length),
        "--tp-size", str(tp_size),
    ]

    # Cấu hình lưu checkpoint định kỳ
    save_interval = cfg.get("save_interval") or cfg.get("save_interval_epochs")
    if save_interval:
        cmd.extend(["--save-interval", str(save_interval)])

    # Xử lý tự động tiếp tục (Resume) hoặc nạp checkpoint khởi đầu (Warm-start)
    resolved_checkpoint: Optional[str] = None
    if checkpoint_path:
        resolved_checkpoint = checkpoint_path
    elif resume or cfg.get("resume") or cfg.get("resume_best"):
        chosen = find_resume_checkpoint(output_dir, prefer_best=prefer_best)
        if chosen:
            resolved_checkpoint = str(chosen)
            tag = "best" if chosen == find_best_checkpoint(output_dir) else "latest"
            logger.info("Đã phát hiện checkpoint (%s) để tiếp tục: %s", tag, resolved_checkpoint)
        else:
            logger.warning("Không tìm thấy checkpoint nào trong %s để resume. Bắt đầu huấn luyện từ đầu.", output_dir)
    elif cfg.get("draft_checkpoint_path") or cfg.get("resume_from_checkpoint"):
        resolved_checkpoint = str(cfg.get("draft_checkpoint_path") or cfg.get("resume_from_checkpoint"))

    if resolved_checkpoint:
        cmd.extend(["--draft-checkpoint-path", str(resolved_checkpoint)])

    # Xử lý kết nối Weights & Biases (WandB)
    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled = False
    if wandb_override is True:
        wandb_enabled = True
    elif isinstance(wandb_cfg, dict) and wandb_cfg.get("enabled"):
        wandb_enabled = True
    elif cfg.get("report_to") == "wandb":
        wandb_enabled = True

    if wandb_enabled:
        cmd.extend(["--report-to", "wandb"])
        project = (wandb_cfg.get("project") if isinstance(wandb_cfg, dict) else None) or cfg.get("wandb_project")
        if project:
            cmd.extend(["--wandb-project", str(project)])
        run_name = (wandb_cfg.get("run_name") if isinstance(wandb_cfg, dict) else None) or cfg.get("wandb_run_name")
        if run_name:
            cmd.extend(["--wandb-name", str(run_name)])

    return cmd


def run_specforge_training(
    config_path: str,
    num_gpus: int = 1,
    resume: bool = False,
    prefer_best: bool = True,
    checkpoint_path: Optional[str] = None,
    save_total_limit: Optional[int] = None,
    wandb: Optional[bool] = None,
    dry_run: bool = False,
) -> List[str]:
    """Nạp cấu hình, dựng lệnh và thực thi tiến trình huấn luyện."""
    cfg = load_training_config(config_path)
    cmd = build_specforge_command(
        cfg,
        num_gpus=num_gpus,
        resume=resume,
        prefer_best=prefer_best,
        checkpoint_path=checkpoint_path,
        wandb_override=wandb,
    )
    logger.info("Lệnh khởi chạy SpecForge: %s", " ".join(cmd))

    if dry_run:
        logger.info("[DRY RUN] Bỏ qua thực thi lệnh subprocess.")
        return cmd

    # Tiền cấu hình các metric và biểu đồ WandB nếu được bật
    wandb_flag = wandb if wandb is not None else cfg.get("wandb", {}).get("enabled", False)
    if wandb_flag:
        setup_wandb_metrics(cfg)

    subprocess.run(cmd, check=True)

    # Dọn dẹp checkpoint cũ vượt quá giới hạn lưu trữ (chỉ giữ best + N latest)
    limit = save_total_limit or cfg.get("save_total_limit", 2)
    output_dir = cfg.get("output_dir", "checkpoints/qwen2.5-7b-eagle3.1")
    prune_old_checkpoints(output_dir, keep_limit=limit)

    return cmd


def main():
    args = parse_training_args()
    run_specforge_training(
        config_path=args.config,
        num_gpus=args.gpus,
        resume=args.resume or args.resume_best,
        prefer_best=not (args.resume and not args.resume_best),
        checkpoint_path=args.checkpoint_path,
        save_total_limit=args.save_total_limit,
        wandb=args.wandb if args.wandb else None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
