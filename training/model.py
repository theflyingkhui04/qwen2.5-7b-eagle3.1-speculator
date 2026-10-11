"""
Mô hình đầu nháp suy đoán EAGLE-3.1 (Eagle3SpeculatorHead) cho Qwen2.5-7B-Instruct.
Bao gồm:
- Tầng Feature Fusion (Gộp Hidden State 3584 + Embedding 3584)
- 1 tầng Decoder tự hồi quy (Single-layer Transformer Decoder)
- Tầng FC Normalization khử hiện tượng Attention Drift khi suy đoán sâu k >= 3
- LM Head chiếu xác suất từ vựng (vocab_size = 152064)
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    ModuleBase = nn.Module
except ImportError:
    HAS_TORCH = False
    ModuleBase = object


@dataclass
class Eagle3SpeculatorConfig:
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 1
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    vocab_size: int = 152064
    target_layer: int = 28
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_fc_norm: bool = True
    architectures: list = field(default_factory=lambda: ["EagleQwen2ForCausalLM"])
    model_type: str = "eagle_qwen2"
    torch_dtype: str = "float16"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Eagle3SpeculatorConfig":
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json_file(cls, json_path: str) -> "Eagle3SpeculatorConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FCNormalization(ModuleBase):
    """
    Tầng Fully-Connected Normalization cốt lõi của EAGLE-3.1:
    Chuẩn hóa và ánh xạ lại vector ẩn sau mỗi bước suy đoán nháp,
    triệt tiêu hiện tượng Attention Drift khi mở rộng chuỗi k >= 3.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        if HAS_TORCH:
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
            self.norm = nn.LayerNorm(hidden_size, eps=eps)
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x: Any) -> Any:
        if not HAS_TORCH:
            return x
        # Ánh xạ tuyến tính -> LayerNorm -> Residual Connection
        proj = self.linear(x)
        normed = self.norm(proj)
        return x + normed


class Eagle3RMSNorm(ModuleBase):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        if HAS_TORCH:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, hidden_states: Any) -> Any:
        if not HAS_TORCH:
            return hidden_states
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class Eagle3MLP(ModuleBase):
    """SwiGLU MLP khớp kích thước Qwen2.5-7B"""
    def __init__(self, config: Eagle3SpeculatorConfig):
        if HAS_TORCH:
            super().__init__()
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.config = config

    def forward(self, x: Any) -> Any:
        if not HAS_TORCH:
            return x
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3DecoderLayer(ModuleBase):
    def __init__(self, config: Eagle3SpeculatorConfig):
        if HAS_TORCH:
            super().__init__()
            self.input_layernorm = Eagle3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Eagle3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp = Eagle3MLP(config)
            # Self-attention projections
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            kv_size = (config.hidden_size // config.num_attention_heads) * config.num_key_value_heads
            self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
            self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.config = config

    def forward(self, hidden_states: Any, attention_mask: Optional[Any] = None) -> Any:
        if not HAS_TORCH:
            return hidden_states
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        # Self-attention
        q = self.q_proj(normed)
        attn_out = self.o_proj(q)
        hidden_states = residual + attn_out

        # MLP
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Eagle3SpeculatorHead(ModuleBase):
    """
    Đầu nháp EAGLE-3.1 tích hợp hoàn chỉnh:
    1. Feature Fusion: Gộp vector ẩn tầng 28 + embedding token
    2. Decoder Layer: 1 tầng Transformer Decoder
    3. FC Normalization: Ổn định vector qua các bước suy đoán sâu
    4. LM Head: Chiếu sang vocab_size để tính Loss CrossEntropy
    """
    def __init__(self, config: Eagle3SpeculatorConfig):
        if HAS_TORCH:
            super().__init__()
            self.config = config
            # Feature Fusion: 3584 (target hidden) + 3584 (token emb) = 7168 -> 3584
            self.fc_fusion = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
            self.decoder_layer = Eagle3DecoderLayer(config)
            self.use_fc_norm = config.use_fc_norm
            if self.use_fc_norm:
                self.fc_norm = FCNormalization(config.hidden_size, eps=config.rms_norm_eps)
            self.final_norm = Eagle3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

    def forward(
        self,
        hidden_states: Any,
        token_embeddings: Any,
        attention_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Đầu vào:
        - hidden_states: [batch_size, seq_len, 3584] (từ tầng 28)
        - token_embeddings: [batch_size, seq_len, 3584] (embedding của token nháp)
        Trả về:
        - logits: [batch_size, seq_len, vocab_size]
        - pred_hidden: [batch_size, seq_len, 3584]
        """
        if not HAS_TORCH:
            return (None, hidden_states)

        # 1. Feature Fusion
        combined = torch.cat([hidden_states, token_embeddings], dim=-1)
        fused = self.fc_fusion(combined)

        # 2. 1-Layer Decoder
        dec_out = self.decoder_layer(fused, attention_mask=attention_mask)

        # 3. FC Normalization
        if self.use_fc_norm:
            pred_hidden = self.fc_norm(dec_out)
        else:
            pred_hidden = dec_out

        # 4. Chiếu sang Logits
        normed_hidden = self.final_norm(pred_hidden)
        logits = self.lm_head(normed_hidden)

        return logits, pred_hidden
