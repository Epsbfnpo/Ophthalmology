from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import cohen_kappa_score, accuracy_score
import numpy as np

from modeling.hc_gdrnet import HCGDRNet


@dataclass
class LossWeights:
    cls: float = 1.0
    seg: float = 1.0
    distill: float = 0.0
    concept: float = 0.5
    reg: float = 2.0
    ib: float = 0.1


def update_ema(student: nn.Module, teacher: nn.Module, momentum: float = 0.999):
    s_model = student.module if hasattr(student, "module") else student
    t_model = teacher.module if hasattr(teacher, "module") else teacher

    with torch.no_grad():
        for sp, tp in zip(s_model.parameters(), t_model.parameters()):
            tp.data.mul_(momentum).add_(sp.data, alpha=1 - momentum)


class HCMTLGGDRNetTrainer:
    def __init__(
            self,
            concept_bank,
            device="cuda",
            weights: Optional[LossWeights] = None,
            class_weights: Optional[torch.Tensor] = None
    ):
        self.device = device
        self.weights = weights if weights is not None else LossWeights()
        self.class_weights = class_weights

        self.student = HCGDRNet(concept_bank=concept_bank, seg_classes=6)
        self.teacher = HCGDRNet(concept_bank=concept_bank, seg_classes=6)

        if torch.cuda.device_count() > 1:
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

        # Segmentation weights: Background(0) low, Retina(1) low, Lesions(2-5) high
        self.seg_weights = torch.tensor([0.1, 0.2, 2.0, 2.0, 2.0, 2.0], device=device)

        self.global_step = 0

    def update_weights(self, new_weights: LossWeights):
        self.weights = new_weights
        print(
            f"🔄 [Trainer] Weights Updated: CLS={self.weights.cls} | SEG={self.weights.seg} | CON={self.weights.concept}")

    def update(self, batch, optimizer, has_masks=True):
        self.student.train()
        images, masks, labels = batch
        images = images.to(self.device)
        masks = masks.to(self.device)
        labels = labels.to(self.device)

        s_pred_cls, s_pred_mask, s_vis_emb, s_z_l1, s_z_l2 = self.student(images)

        with torch.no_grad():
            _, _, t_vis_emb, _, _ = self.teacher(images)

        # --- Losses ---
        if self.class_weights is not None:
            loss_cls = F.cross_entropy(s_pred_cls, labels, weight=self.class_weights, label_smoothing=0.1)
        else:
            loss_cls = F.cross_entropy(s_pred_cls, labels, label_smoothing=0.1)

        if has_masks:
            # We train background (class 0) to force the decoder to map zero-inputs to background
            loss_seg = F.cross_entropy(s_pred_mask, masks, weight=self.seg_weights)
        else:
            loss_seg = torch.tensor(0.0, device=self.device)

        loss_distill = F.mse_loss(s_vis_emb, t_vis_emb)

        if self.bank_l1 is not None:
            s_vis_norm = F.normalize(s_vis_emb, dim=1)
            bank_l1_norm = F.normalize(self.bank_l1, dim=1)
            bank_l2_norm = F.normalize(self.bank_l2, dim=1)
            with torch.no_grad():
                target_sim_l1 = (s_vis_norm @ bank_l1_norm.T).clamp(0, 1)
                target_sim_l2 = (s_vis_norm @ bank_l2_norm.T).clamp(0, 1)
            loss_concept_l1 = F.mse_loss(torch.sigmoid(s_z_l1), target_sim_l1)
            loss_concept_l2 = F.mse_loss(torch.sigmoid(s_z_l2), target_sim_l2)
            loss_concept = loss_concept_l1 + loss_concept_l2
        else:
            loss_concept = torch.tensor(0.0, device=self.device)

        probs_l2 = torch.sigmoid(s_z_l2)
        loss_reg = probs_l2[labels == 0].mean() if (labels == 0).any() else torch.tensor(0.0, device=self.device)
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

        grad_norm = 0.0
        for p in self.student.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        update_ema(self.student, self.teacher)

        with torch.no_grad():
            train_preds = torch.argmax(s_pred_cls, dim=1)
            train_acc = (train_preds == labels).float().mean().item()
            pred_counts = Counter(train_preds.cpu().numpy())
            label_counts = Counter(labels.cpu().numpy())

        self.global_step += 1

        if self.global_step % 50 == 0:
            print(f"\n🩺 [TRAIN] Step {self.global_step}")
            print(f"   Losses: Total={total_loss.item():.3f} | CLS={loss_cls.item():.3f} | SEG={loss_seg.item():.3f}")
            print(f"   Pred Dist : {dict(sorted(pred_counts.items()))}")
            print(f"   Label Dist: {dict(sorted(label_counts.items()))}")
            print(f"   Grad Norm : {grad_norm:.4f}")

        return {
            "loss": total_loss.item(),
            "loss_seg": loss_seg.item(),
            "loss_cls": loss_cls.item(),
            "train_acc": train_acc
        }

    def validate(self, dataloader, domain_name="Unknown"):
        self.student.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                images, masks, labels = batch
                images = images.to(self.device)

                # Inference
                pred_cls, _, _, _, _ = self.student(images)
                preds = torch.argmax(pred_cls, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        self.student.train()

        acc = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

        val_pred_counts = Counter(all_preds)
        print(f"📊 [VAL DIST] {domain_name} Predictions: {dict(sorted(val_pred_counts.items()))}")

        return {"Accuracy": acc, "Kappa": kappa}