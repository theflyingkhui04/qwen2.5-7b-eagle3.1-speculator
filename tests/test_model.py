import pytest
from training.model import Eagle3SpeculatorConfig, FCNormalization

def test_config_defaults():
    config = Eagle3SpeculatorConfig()
    assert config.hidden_size == 3584
    assert config.intermediate_size == 18944
    assert config.num_hidden_layers == 1
    assert config.num_attention_heads == 28
    assert config.num_key_value_heads == 4
    assert config.target_layer == 28
    assert config.use_fc_norm is True
    assert config.vocab_size == 152064

def test_config_from_dict():
    cfg_dict = {
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "use_fc_norm": False,
        "target_layer": 24,
    }
    config = Eagle3SpeculatorConfig.from_dict(cfg_dict)
    assert config.hidden_size == 2048
    assert config.num_attention_heads == 16
    assert config.use_fc_norm is False
    assert config.target_layer == 24

def test_fc_norm_initialization():
    norm = FCNormalization(hidden_size=3584)
    assert norm.hidden_size == 3584
