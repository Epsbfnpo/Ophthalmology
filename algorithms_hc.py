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
    distill: float = 2.0
    concept: float = 5.0
    reg: float = 2.0
    ib: float = 0.1


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2.0 * intersection + eps) / (union + eps)


def update_ema(student: nn.Module, teacher: nn.Module, momentum: float = 0.999):
    with torch.no_grad():
        for sp, tp in zip(student.parameters(), teacher.parameters()):
            tp.data.mul_(momentum).add_(sp.data, alpha=1 - momentum)


class HCMTLGGDRNetTrainer:
    def __init__(self, concept_bank, device="cuda", weights: Optional[LossWeights] = None):
        self.device = device
        self.weights = weights or LossWeights()
        self.student = HCGDRNet(concept_bank=concept_bank, num_l1_concepts=2, num_l2_concepts=4).to(device)
        self.teacher = HCGDRNet(concept_bank=concept_bank, num_l1_concepts=2, num_l2_concepts=4).to(device)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.bank_l1 = concept_bank[:2].to(device) if concept_bank is not None else None
        self.bank_l2 = concept_bank[2:].to(device) if concept_bank is not None else None

    def update(self, batch, optimizer, has_masks):
        images, masks, labels = [x.to(self.device) for x in batch]

        s_feat, s_mask, s_z_l1, s_z_l2, s_logits, s_vis_emb = self.student(images)
        with torch.no_grad():
            t_feat, _, _, _, _, _ = self.teacher(images)

        loss_cls = F.cross_entropy(s_logits, labels)
        loss_seg = dice_loss(s_mask, masks) if has_masks else torch.tensor(0.0, device=self.device)
        loss_distill = F.mse_loss(s_feat, t_feat)

        if self.bank_l1 is not None:
            with torch.no_grad():
                target_sim_l1 = s_vis_emb @ self.bank_l1.T
                target_sim_l2 = s_vis_emb @ self.bank_l2.T

            loss_concept_l1 = F.mse_loss(torch.sigmoid(s_z_l1), target_sim_l1)
            loss_concept_l2 = F.mse_loss(torch.sigmoid(s_z_l2), target_sim_l2)
            loss_concept = loss_concept_l1 + loss_concept_l2
        else:
            loss_concept = torch.tensor(0.0, device=self.device)

        probs_l2 = torch.sigmoid(s_z_l2)
        healthy_mask = labels == 0
        if healthy_mask.any():
            loss_reg = probs_l2[healthy_mask].mean()
        else:
            loss_reg = torch.tensor(0.0, device=self.device)

        loss_ib = probs_l2.abs().mean()

        total_loss = (
            self.weights.cls * loss_cls
            + self.weights.seg * loss_seg
            + self.weights.distill * loss_distill
            + self.weights.concept * loss_concept
            + self.weights.reg * loss_reg
            + self.weights.ib * loss_ib
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        update_ema(self.student, self.teacher)

        return {
            "loss": total_loss.item(),
            "cls": loss_cls.item(),
            "seg": loss_seg.item(),
            "concept": loss_concept.item(),
        }
