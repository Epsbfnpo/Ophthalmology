from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from new.modeling.hc_gdrnet import HCGDRNet


@dataclass
class LossWeights:
    cls: float = 1.0
    seg: float = 10.0
    distill: float = 1.0
    concept: float = 1.0
    reg: float = 1.0
    ib: float = 1.0


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def update_ema(student: nn.Module, teacher: nn.Module, momentum: float = 0.99) -> None:
    with torch.no_grad():
        for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)


def _concept_alignment_loss(
    concept_vec: torch.Tensor,
    concept_bank: Optional[torch.Tensor],
    labels: torch.Tensor,
) -> torch.Tensor:
    if concept_bank is None:
        return torch.tensor(0.0, device=concept_vec.device)
    concept_bank = F.normalize(concept_bank, dim=1)
    concept_vec = F.normalize(concept_vec, dim=1)
    normal_idx = concept_bank.shape[0] - 1
    lesion_proto = concept_bank[:-1].mean(dim=0)
    targets = torch.where(
        labels[:, None] == 0,
        concept_bank[normal_idx],
        lesion_proto,
    )
    similarity = (concept_vec * targets).sum(dim=1)
    return (1 - similarity).mean()


class HCMTLGGDRNetTrainer:
    def __init__(
        self,
        concept_bank: Optional[torch.Tensor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        weights: Optional[LossWeights] = None,
    ) -> None:
        self.device = device
        self.weights = weights or LossWeights()
        self.student = HCGDRNet(concept_bank=concept_bank).to(device)
        self.teacher = HCGDRNet(concept_bank=concept_bank).to(device)
        self.teacher.load_state_dict(self.student.state_dict())
        for param in self.teacher.parameters():
            param.requires_grad = False

    def update(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        has_masks: bool,
    ) -> dict:
        images, masks, labels = batch
        images = images.to(self.device)
        masks = masks.to(self.device)
        labels = labels.to(self.device)

        pred_mask, concept_vec, logits = self.student(images)

        cls_loss = F.cross_entropy(logits, labels)
        seg_loss = dice_loss(pred_mask, masks) if has_masks else torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            teacher_mask, teacher_concept, _ = self.teacher(images)

        distill_loss = F.mse_loss(pred_mask, teacher_mask)
        concept_loss = _concept_alignment_loss(concept_vec, self.student.concept_bank, labels)

        reg_loss = torch.relu(concept_vec).sum(dim=1)[labels == 0].mean() if (labels == 0).any() else torch.tensor(0.0, device=self.device)
        ib_loss = concept_vec.abs().mean()

        total_loss = (
            self.weights.cls * cls_loss
            + self.weights.seg * seg_loss
            + self.weights.distill * distill_loss
            + self.weights.concept * concept_loss
            + self.weights.reg * reg_loss
            + self.weights.ib * ib_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        update_ema(self.student, self.teacher)

        return {
            "loss": total_loss.item(),
            "cls": cls_loss.item(),
            "seg": seg_loss.item(),
            "distill": distill_loss.item(),
            "concept": concept_loss.item(),
            "reg": reg_loss.item(),
            "ib": ib_loss.item(),
        }
