import pytest
from unittest.mock import patch
from extraction.extract_hidden_states import (
    bucket_sequences,
    check_disk_space,
    parse_extraction_args,
)

def test_bucket_sequences():
    samples = [
        {"id": 1, "length": 120},
        {"id": 2, "length": 450},
        {"id": 3, "length": 700},
        {"id": 4, "length": 1800},
        {"id": 5, "length": 3000},  # Vượt max 2048 -> loại bỏ
    ]
    buckets = bucket_sequences(samples, bucket_limits=[512, 1024, 2048])
    assert len(buckets[512]) == 2
    assert len(buckets[1024]) == 1
    assert len(buckets[2048]) == 1
    assert sum(len(v) for v in buckets.values()) == 4

def test_check_disk_space_pass():
    with patch("shutil.disk_usage") as mock_usage:
        # Mock 100GB free
        mock_usage.return_value = (200 * (1024**3), 100 * (1024**3), 100 * (1024**3))
        assert check_disk_space("/workspace", min_free_gb=50) is True

def test_check_disk_space_fail():
    with patch("shutil.disk_usage") as mock_usage:
        # Mock 20GB free
        mock_usage.return_value = (200 * (1024**3), 180 * (1024**3), 20 * (1024**3))
        assert check_disk_space("/workspace", min_free_gb=50) is False

def test_parse_extraction_args():
    args = parse_extraction_args(["--target-layer", "28", "--chunk-size", "5000"])
    assert args.target_layer == 28
    assert args.chunk_size == 5000
    assert args.model_id == "Qwen/Qwen2.5-7B-Instruct"
