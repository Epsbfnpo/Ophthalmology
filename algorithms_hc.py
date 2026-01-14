from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import cohen_kappa_score, accuracy_score
import numpy as np

from modeling.hc_gdrnet import HCGDRNet


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
    s_model = student.module if hasattr(student, "module") else student
    t_model = teacher.module if hasattr(teacher, "module") else teacher

    with torch.no_grad():
        for sp, tp in zip(s_model.parameters(), t_model.parameters()):
            tp.data.mul_(momentum).add_(sp.data, alpha=1 - momentum)


class HCMTLGGDRNetTrainer:
    def __init__(self, concept_bank, device="cuda", weights: Optional[LossWeights] = None):
        self.device = device
        self.weights = weights or LossWeights()

        self.student = HCGDRNet(concept_bank=concept_bank, num_l1_concepts=2, num_l2_concepts=4)
        self.teacher = HCGDRNet(concept_bank=concept_bank, num_l1_concepts=2, num_l2_concepts=4)

        if torch.cuda.device_count() > 1:
            print(f"🔥 Trainer Detected {torch.cuda.device_count()} GPUs! Wrapping with DataParallel.")
            self.student = nn.DataParallel(self.student)
            self.teacher = nn.DataParallel(self.teacher)

        self.student.to(device)
        self.teacher.to(device)

        if concept_bank is not None:
            self.bank_l1 = concept_bank[:2].to(device)
            self.bank_l2 = concept_bank[2:].to(device)
        else:
            self.bank_l1 = None
            self.bank_l2 = None

        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

    def update(self, batch, optimizer, has_masks=True):
        self.student.train()
        images, masks, labels = batch
        images = images.to(self.device)
        masks = masks.to(self.device)
        labels = labels.to(self.device)

        # 1. Student Forward
        model_core = self.student.module if isinstance(self.student, nn.DataParallel) else self.student

        s_feat = model_core.backbone(images)
        s_pred_mask = model_core.decoder(s_feat, target_size=images.shape[-2:])

        clean_mask = s_pred_mask.detach()
        clean_mask = torch.sigmoid(clean_mask)
        mask_small = F.interpolate(clean_mask, size=s_feat.shape[-2:], mode="nearest")
        gated_feat = s_feat * mask_small

        # [CRITICAL FIX]: 这里必须做 GAP (Global Average Pooling) 把 [B, C, H, W] 变成 [B, C]
        s_pooled_feat = gated_feat.mean(dim=(2, 3))

        # 喂给 Projector 和 Adapter
        s_z_l1, s_z_l2 = model_core.projector(s_pooled_feat)
        s_vis_emb = model_core.feat_adapter(s_pooled_feat)

        s_pred_cls = model_core.classifier(s_z_l2)

        # 2. Teacher Forward (EMA)
        with torch.no_grad():
            teacher_core = self.teacher.module if isinstance(self.teacher, nn.DataParallel) else self.teacher
            t_feat = teacher_core.backbone(images)

            t_pred_mask = teacher_core.decoder(t_feat, target_size=images.shape[-2:])
            t_clean_mask = torch.sigmoid(t_pred_mask)
            t_mask_small = F.interpolate(t_clean_mask, size=t_feat.shape[-2:], mode="nearest")
            t_gated_feat = t_feat * t_mask_small

            # [CRITICAL FIX]: Teacher 也要做 GAP
            t_pooled_feat = t_gated_feat.mean(dim=(2, 3))
            t_vis_emb = teacher_core.feat_adapter(t_pooled_feat)

        # 3. Calculate Losses
        loss_cls = F.cross_entropy(s_pred_cls, labels)
        loss_seg = dice_loss(s_pred_mask, masks) if has_masks else torch.tensor(0.0, device=self.device)
        loss_distill = F.mse_loss(s_vis_emb, t_vis_emb)

        if self.bank_l1 is not None:
            s_vis_norm = F.normalize(s_vis_emb, dim=1)
            bank_l1_norm = F.normalize(self.bank_l1, dim=1)
            bank_l2_norm = F.normalize(self.bank_l2, dim=1)

            with torch.no_grad():
                target_sim_l1 = s_vis_norm @ bank_l1_norm.T
                target_sim_l2 = s_vis_norm @ bank_l2_norm.T
                target_sim_l1 = target_sim_l1.clamp(0, 1)
                target_sim_l2 = target_sim_l2.clamp(0, 1)

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
            "loss_seg": loss_seg.item(),
            "loss_cls": loss_cls.item(),
            "loss_concept": loss_concept.item(),
            "loss_reg": loss_reg.item(),
            "loss_distill": loss_distill.item(),
            "loss_ib": loss_ib.item()
        }

    def validate(self, dataloader):
        self.student.eval()
        all_preds = []
        all_labels = []

        model_core = self.student.module if isinstance(self.student, nn.DataParallel) else self.student

        with torch.no_grad():
            for batch in dataloader:
                images, _, labels = batch
                images = images.to(self.device)

                feat = model_core.backbone(images)
                pred_mask = model_core.decoder(feat, target_size=images.shape[-2:])

                clean_mask = torch.sigmoid(pred_mask)
                mask_small = F.interpolate(clean_mask, size=feat.shape[-2:], mode="nearest")
                gated_feat = feat * mask_small

                # [CRITICAL FIX]: Validation 时也要做 GAP
                pooled_feat = gated_feat.mean(dim=(2, 3))

                _, z_l2 = model_core.projector(pooled_feat)
                pred_cls = model_core.classifier(z_l2)

                preds = torch.argmax(pred_cls, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        self.student.train()

        acc = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

        return {"Accuracy": acc, "Kappa": kappa}
