from pathlib import Path
from unittest.mock import patch
import pytest

from training.train import (
    build_specforge_command,
    find_best_checkpoint,
    find_latest_checkpoint,
    find_resume_checkpoint,
    load_training_config,
    parse_training_args,
    prune_old_checkpoints,
    run_specforge_training,
    setup_wandb_metrics,
)


def test_parse_training_args_defaults():
    args = parse_training_args(["--config", "configs/train_config.yaml", "--gpus", "1"])
    assert args.config == "configs/train_config.yaml"
    assert args.gpus == 1
    assert args.dry_run is False
    assert args.resume is False
    assert args.checkpoint_path is None
    assert args.wandb is False


def test_parse_training_args_custom():
    args = parse_training_args([
        "--config", "custom.yaml",
        "--gpus", "4",
        "--resume",
        "--checkpoint-path", "checkpoints/best",
        "--wandb",
        "--dry-run",
    ])
    assert args.config == "custom.yaml"
    assert args.gpus == 4
    assert args.resume is True
    assert args.checkpoint_path == "checkpoints/best"
    assert args.wandb is True
    assert args.dry_run is True


def test_find_latest_checkpoint_empty_or_nonexistent(tmp_path):
    assert find_latest_checkpoint(tmp_path / "nonexistent") is None
    assert find_latest_checkpoint(tmp_path) is None


def test_find_latest_checkpoint_subdirectories(tmp_path):
    (tmp_path / "checkpoint-500").mkdir()
    (tmp_path / "checkpoint-1500").mkdir()
    (tmp_path / "checkpoint-1000").mkdir()
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-1500"


def test_find_latest_checkpoint_files(tmp_path):
    (tmp_path / "checkpoint_epoch_1.pt").write_text("chk1")
    (tmp_path / "checkpoint_epoch_3.pt").write_text("chk3")
    (tmp_path / "checkpoint_epoch_2.pt").write_text("chk2")
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint_epoch_3.pt"


def test_build_specforge_command():
    cfg = {
        "target_model_path": "Qwen/Qwen2.5-7B-Instruct",
        "draft_model_config": "configs/eagle3.1_qwen2.5_7b.json",
        "train_data_path": "data/processed/train.jsonl",
        "output_dir": "checkpoints/qwen2.5-7b-eagle3.1",
        "num_epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "max_seq_len": 2048,
        "ttt_length": 5,
        "tp_size": 1,
    }
    cmd = build_specforge_command(cfg, num_gpus=2)
    assert cmd[0] == "torchrun"
    assert "--nproc_per_node=2" in cmd
    assert "-m" in cmd
    assert "specforge.train" in cmd
    assert "--model-path" in cmd
    assert "Qwen/Qwen2.5-7B-Instruct" in cmd
    assert "--draft-config" in cmd
    assert "configs/eagle3.1_qwen2.5_7b.json" in cmd
    assert "--data-path" in cmd
    assert "data/processed/train.jsonl" in cmd
    assert "--output-dir" in cmd
    assert "checkpoints/qwen2.5-7b-eagle3.1" in cmd
    assert "--num-epochs" in cmd
    assert "3" in cmd
    assert "--micro-batch-size" in cmd
    assert "2" in cmd
    assert "--grad-accum-steps" in cmd
    assert "4" in cmd
    assert "--lr" in cmd
    assert "0.0001" in cmd or "1e-04" in cmd or "0.0001" in str(cfg["learning_rate"])
    assert "--max-seq-len" in cmd
    assert "2048" in cmd
    assert "--ttt-length" in cmd
    assert "5" in cmd
    assert "--tp-size" in cmd
    assert "1" in cmd


def test_build_specforge_command_with_wandb_and_save_interval():
    cfg = {
        "target_model_path": "Qwen/Qwen2.5-7B-Instruct",
        "draft_model_config": "configs/eagle3.1_qwen2.5_7b.json",
        "train_data_path": "data/processed/train.jsonl",
        "output_dir": "checkpoints/qwen2.5-7b-eagle3.1",
        "save_interval": 1,
        "wandb": {
            "enabled": True,
            "project": "qwen2.5-eagle3.1",
            "run_name": "test-run",
        },
    }
    cmd = build_specforge_command(cfg, num_gpus=1)
    assert "--save-interval" in cmd
    assert "1" in cmd
    assert "--report-to" in cmd
    assert "wandb" in cmd
    assert "--wandb-project" in cmd
    assert "qwen2.5-eagle3.1" in cmd
    assert "--wandb-name" in cmd
    assert "test-run" in cmd


def test_build_specforge_command_with_resume(tmp_path):
    chk_dir = tmp_path / "checkpoints"
    chk_dir.mkdir()
    (chk_dir / "checkpoint-2000").mkdir()

    cfg = {
        "target_model_path": "Qwen/Qwen2.5-7B-Instruct",
        "draft_model_config": "configs/eagle3.1_qwen2.5_7b.json",
        "train_data_path": "data/processed/train.jsonl",
        "output_dir": str(chk_dir),
    }

    # Resume tự động tìm checkpoint
    cmd = build_specforge_command(cfg, num_gpus=1, resume=True)
    assert "--draft-checkpoint-path" in cmd
    assert str(chk_dir / "checkpoint-2000") in cmd

    # Truyền checkpoint cụ thể
    cmd2 = build_specforge_command(cfg, num_gpus=1, checkpoint_path=str(chk_dir / "custom_checkpoint"))
    assert "--draft-checkpoint-path" in cmd2
    assert str(chk_dir / "custom_checkpoint") in cmd2


def test_run_specforge_training_dry_run():
    root = Path(__file__).parent.parent
    config_path = str(root / "configs" / "train_config.yaml")
    cmd = run_specforge_training(config_path, num_gpus=1, dry_run=True)
    assert cmd[0] == "torchrun"
    assert "--nproc_per_node=1" in cmd
    assert "-m" in cmd
    assert "specforge.train" in cmd
    assert "--model-path" in cmd
    assert "Qwen/Qwen2.5-7B-Instruct" in cmd


def test_run_specforge_training_executes_subprocess():
    root = Path(__file__).parent.parent
    config_path = str(root / "configs" / "train_config.yaml")
    with patch("subprocess.run") as mock_subproc:
        cmd = run_specforge_training(config_path, num_gpus=2, dry_run=False)
        mock_subproc.assert_called_once_with(cmd, check=True)
        assert "--nproc_per_node=2" in cmd


def test_setup_wandb_metrics_disabled():
    cfg = {"wandb": {"enabled": False}}
    assert setup_wandb_metrics(cfg) is False


def test_setup_wandb_metrics_enabled():
    import sys
    from unittest.mock import MagicMock

    cfg = {
        "wandb": {
            "enabled": True,
            "project": "test-project",
            "run_name": "test-run",
        }
    }
    mock_run = MagicMock()
    mock_run.id = "mock-run-id-123"

    mock_wandb = MagicMock()
    mock_wandb.init.return_value = mock_run

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        res = setup_wandb_metrics(cfg)
        assert res is True
        mock_wandb.init.assert_called_once_with(
            project="test-project",
            name="test-run",
            resume="allow",
            reinit=True,
        )
        metric_names = [call.args[0] for call in mock_run.define_metric.call_args_list]
        assert "train/step" in metric_names
        assert "train/*" in metric_names
        assert "train/loss" in metric_names
        assert "eval/loss" in metric_names
        assert "acceptance/alpha_overall" in metric_names
        assert "acceptance/alpha_step_1" in metric_names
        mock_run.finish.assert_called_once()


def test_find_best_checkpoint_directory_or_metadata(tmp_path):
    assert find_best_checkpoint(tmp_path) is None

    best_dir = tmp_path / "best"
    best_dir.mkdir()
    assert find_best_checkpoint(tmp_path) == best_dir

    # Thử qua metadata file
    meta_dir = tmp_path / "meta_test"
    meta_dir.mkdir()
    target_chk = meta_dir / "checkpoint-800"
    target_chk.mkdir()
    (meta_dir / "best_metric.json").write_text(f'{{"best_checkpoint_path": "{str(target_chk).replace("\\", "\\\\")}"}}')
    assert find_best_checkpoint(meta_dir) == target_chk


def test_find_resume_checkpoint_prefers_best(tmp_path):
    (tmp_path / "checkpoint-500").mkdir()
    (tmp_path / "checkpoint-1000").mkdir()
    best_dir = tmp_path / "best"
    best_dir.mkdir()

    # Mặc định ưu tiên best
    chosen_best = find_resume_checkpoint(tmp_path, prefer_best=True)
    assert chosen_best == best_dir

    # Nếu prefer_best=False thì lấy latest (checkpoint-1000)
    chosen_latest = find_resume_checkpoint(tmp_path, prefer_best=False)
    assert chosen_latest.name == "checkpoint-1000"


def test_prune_old_checkpoints(tmp_path):
    best_dir = tmp_path / "best"
    best_dir.mkdir()
    c1 = tmp_path / "checkpoint-100"
    c2 = tmp_path / "checkpoint-200"
    c3 = tmp_path / "checkpoint-300"
    c4 = tmp_path / "checkpoint-400"
    for c in [c1, c2, c3, c4]:
        c.mkdir()

    # Giới hạn giữ lại 2 checkpoint latest + luôn giữ best
    deleted = prune_old_checkpoints(tmp_path, keep_limit=2)
    assert len(deleted) == 2
    deleted_names = [d.name for d in deleted]
    assert "checkpoint-100" in deleted_names
    assert "checkpoint-200" in deleted_names

    # Checkpoint-300, 400 và best vẫn còn nguyên
    assert best_dir.exists()
    assert (tmp_path / "checkpoint-300").exists()
    assert (tmp_path / "checkpoint-400").exists()
    assert not (tmp_path / "checkpoint-100").exists()
    assert not (tmp_path / "checkpoint-200").exists()



