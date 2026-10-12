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
    args = parse_extraction_args(["--target-layer", "28", "--chunk-size", "5000", "--force"])
    assert args.target_layer == 28
    assert args.chunk_size == 5000
    assert args.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert args.force is True


def test_scan_existing_chunks_empty_and_non_existent(tmp_path):
    from extraction.extract_hidden_states import scan_existing_chunks
    completed, next_idx = scan_existing_chunks(tmp_path)
    assert completed == {}
    assert next_idx == 0

    completed_ne, next_idx_ne = scan_existing_chunks(tmp_path / "not_found")
    assert completed_ne == {}
    assert next_idx_ne == 0


def test_scan_existing_chunks_with_mocked_safetensors(tmp_path):
    from unittest.mock import MagicMock
    from pathlib import Path
    import sys
    from extraction.extract_hidden_states import scan_existing_chunks

    chunk_shapes = {
        "features_chunk_0000.safetensors": (5000, 512, 3584),
        "features_chunk_0001.safetensors": (5000, 512, 3584),
        "features_chunk_0002.safetensors": (5000, 512, 3584),
        "features_chunk_0003.safetensors": (229, 512, 3584),
        "features_chunk_0004.safetensors": (5000, 1024, 3584),
        "features_chunk_0005.safetensors": (463, 1024, 3584),
    }

    for fname in chunk_shapes:
        (tmp_path / fname).write_text("dummy")

    class MockSafeOpen:
        def __init__(self, path, framework, device):
            self.fname = Path(path).name

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def keys(self):
            return ["hidden_states", "input_ids"]

        def get_slice(self, key):
            mock_slice = MagicMock()
            mock_slice.get_shape.return_value = chunk_shapes[self.fname]
            return mock_slice

    mock_mod = MagicMock()
    mock_mod.safe_open = MockSafeOpen

    with patch.dict(sys.modules, {"safetensors": mock_mod}):
        completed, next_idx = scan_existing_chunks(tmp_path)

    assert completed[512] == 15229
    assert completed[1024] == 5463
    assert next_idx == 6
