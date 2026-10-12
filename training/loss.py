"""
Hàm mất mát kết hợp (Dual Loss Function) cho EAGLE-3.1:
Total_Loss = Loss_CrossEntropy(token_prediction) + 0.5 * Loss_SmoothL1(predicted_hidden, target_hidden)
"""

from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    ModuleBase = nn.Module
except ImportError:
    HAS_TORCH = False
    ModuleBase = object


class EagleDualLoss(ModuleBase):
    def __init__(self, smooth_l1_weight: float = 0.5, ce_weight: float = 1.0, beta: float = 1.0):
        if HAS_TORCH:
            super().__init__()
            self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            self.reg_loss_fn = nn.SmoothL1Loss(beta=beta)
        self.smooth_l1_weight = smooth_l1_weight
        self.ce_weight = ce_weight
        self.beta = beta

    def forward(
        self,
        logits: Any,
        target_token_ids: Any,
        pred_hidden: Any,
        target_hidden: Any,
        loss_mask: Optional[Any] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Tính toán Dual Loss:
        - logits: [batch_size, seq_len, vocab_size]
        - target_token_ids: [batch_size, seq_len]
        - pred_hidden: [batch_size, seq_len, hidden_size]
        - target_hidden: [batch_size, seq_len, hidden_size]
        """
        if not HAS_TORCH:
            return 0.0, {"ce_loss": 0.0, "reg_loss": 0.0, "total_loss": 0.0}

        vocab_size = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = target_token_ids.reshape(-1)

        # 1. Classification Cross-Entropy Loss
        ce_loss = self.ce_loss_fn(flat_logits.float(), flat_targets)

        # 2. Regression Smooth L1 Loss trên không gian vector ẩn
        if loss_mask is not None:
            mask = loss_mask.unsqueeze(-1).expand_as(pred_hidden)
            masked_pred = pred_hidden * mask
            masked_target = target_hidden * mask
            reg_loss = self.reg_loss_fn(masked_pred.float(), masked_target.float())
        else:
            reg_loss = self.reg_loss_fn(pred_hidden.float(), target_hidden.float())

        # 3. Kết hợp Dual Loss
        total_loss = self.ce_weight * ce_loss + self.smooth_l1_weight * reg_loss

        metrics = {
            "ce_loss": float(ce_loss.detach().item()),
            "reg_loss": float(reg_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }

        return total_loss, metrics
