"""
Training losses for LEEN.

Since LEEN has exact antisymmetry by construction (energy-difference
formulation), no antisymmetry loss is needed. The training objective
combines a primary regression loss with an OOD-margin regularizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedMSE(nn.Module):
    """Balanced Mean Squared Error (BMC) from Ren et al. (CVPR 2022).

    Treats regression as distributional-aware contrastive learning.
    Dynamically upweights underrepresented regions of label space
    (e.g. rare stabilizing mutations).

    The noise scale σ is learnable but clamped to [sigma_min, sigma_max]
    to prevent numerical collapse.
    """

    def __init__(self, sigma_init: float = 1.0, sigma_min: float = 0.1, sigma_max: float = 10.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(sigma_init).log())
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp().clamp(min=self.sigma_min, max=self.sigma_max)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B] predicted ΔΔG.
            target: [B] experimental ΔΔG.

        Returns:
            Scalar loss.
        """
        sigma_sq = self.sigma ** 2

        diff = pred.unsqueeze(1) - target.unsqueeze(0)  # [B, B]
        logits = -(diff ** 2) / (2 * sigma_sq)           # [B, B]

        labels = torch.arange(pred.size(0), device=pred.device)
        loss = F.cross_entropy(logits, labels)

        return loss


class OODMarginLoss(nn.Module):
    """OOD-margin consistency loss.

    Penalizes the squared difference between predictions from clean
    and noise-perturbed representations. Encourages the energy head
    to be robust to small feature drift, improving OOD generalization.

    With the energy-difference formulation, this is applied to the
    ΔΔG output (which depends on the representation at the mutation site).
    """

    def forward(
        self, ddg_clean: torch.Tensor, ddg_perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            ddg_clean: [B] predictions from unperturbed representation.
            ddg_perturbed: [B] predictions from noised representation.

        Returns:
            Scalar loss.
        """
        return F.mse_loss(ddg_clean, ddg_perturbed)


class LEENLoss(nn.Module):
    """Combined training objective for LEEN.

    L_total = L_primary + λ_ood * L_ood

    The primary loss is either MSE, Huber, or Balanced MSE (BMC).
    No antisymmetry loss is needed — it's satisfied by the architecture.
    """

    def __init__(
        self,
        primary: str = "bmc",
        bmc_sigma_init: float = 1.0,
        ood_weight: float = 0.5,
    ):
        super().__init__()

        self.ood_weight = ood_weight
        self.ood_loss = OODMarginLoss()

        # Primary regression loss
        if primary == "bmc":
            self.primary_loss = BalancedMSE(sigma_init=bmc_sigma_init)
        elif primary == "mse":
            self.primary_loss = nn.MSELoss()
        elif primary == "huber":
            self.primary_loss = nn.HuberLoss(delta=1.0)
        else:
            raise ValueError(f"Unknown primary loss: {primary}")

        self.primary_name = primary

    def forward(
        self,
        ddg_pred: torch.Tensor,
        ddg_target: torch.Tensor,
        ddg_perturbed: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            ddg_pred: [B] predicted ΔΔG from clean forward pass.
            ddg_target: [B] experimental ΔΔG.
            ddg_perturbed: [B] predicted ΔΔG from perturbed pass (optional).

        Returns:
            Dict with 'total', 'primary', and 'ood' loss scalars.
        """
        loss_primary = self.primary_loss(ddg_pred, ddg_target)

        result = {
            "primary": loss_primary,
            "total": loss_primary,
        }

        if ddg_perturbed is not None and self.ood_weight > 0:
            loss_ood = self.ood_loss(ddg_pred.detach(), ddg_perturbed)
            result["ood"] = loss_ood
            result["total"] = loss_primary + self.ood_weight * loss_ood
        else:
            result["ood"] = torch.tensor(0.0, device=ddg_pred.device)

        return result
