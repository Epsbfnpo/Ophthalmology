import argparse
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from collections import defaultdict

from algorithms_hc import HCMTLGGDRNetTrainer, LossWeights
from dataset.medical_dataset import DomainConfig, MedicalDataset


class RobustCurriculumScheduler:
    def __init__(self, weights: LossWeights, total_epochs: int):
        self.weights = weights
        self.total_epochs = total_epochs
        self.phase1_end = int(total_epochs * 0.3)
        self.phase2_end = int(total_epochs * 0.7)

    def step(self, epoch: int):
        if epoch < self.phase1_end:
            self.weights.seg = 10.0
            self.weights.distill = 1.0
            self.weights.concept = 0.0
            self.weights.ib = 0.0
            self.weights.cls = 0.0
            self.weights.reg = 0.0
            return "Phase 1: Vision Warm-up (Seg Focus)"
        elif epoch < self.phase2_end:
            self.weights.seg = 1.0
            self.weights.distill = 1.0
            self.weights.concept = 5.0
            self.weights.ib = 0.0
            self.weights.cls = 0.0
            self.weights.reg = 0.0
            return "Phase 2: Semantic Alignment (CLIP Focus)"
        else:
            self.weights.seg = 1.0
            self.weights.distill = 1.0
            self.weights.concept = 1.0
            self.weights.ib = 0.01
            self.weights.cls = 1.0
            self.weights.reg = 1.0
            return "Phase 3: Final Logic Tuning (Diagnosis)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--source_domains", nargs="+", required=True)
    parser.add_argument("--target_domains", nargs="+", required=True)
    parser.add_argument("--concept_bank", type=str, default="concepts.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # 1. 配置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Using device: {device}")

    # 2. 准备数据集配置
    source_domains = [DomainConfig(name=name, root=os.path.join(args.data_root, name), has_masks=True) for name in args.source_domains]
    target_domains = [DomainConfig(name=name, root=os.path.join(args.data_root, name), has_masks=False) for name in args.target_domains]

    # 3. 加载 Concept Bank
    print(f"[Init] Loading concept bank from {args.concept_bank}")
    # weights_only=True 防止新版 pytorch 警告
    concept_bank = torch.load(args.concept_bank, map_location=device, weights_only=True)

    # 4. 初始化训练器
    weights = LossWeights()
    trainer = HCMTLGGDRNetTrainer(concept_bank=concept_bank, device=device, weights=weights)

    # 优化器
    optimizer = torch.optim.AdamW(trainer.student.parameters(), lr=args.lr, weight_decay=1e-4)

    # 5. DataLoader 准备
    # Source (Train)
    print(f"[Init] Loading Source Domains: {args.source_domains}")
    source_datasets = [MedicalDataset(domain, augment=True) for domain in source_domains]
    source_dataset = ConcatDataset(source_datasets)
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Target (Validation) - 验证集不增强
    print(f"[Init] Loading Target Domains (Validation): {args.target_domains}")
    target_loaders = {}
    for domain in target_domains:
        val_ds = MedicalDataset(domain, augment=False)
        target_loaders[domain.name] = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    scheduler = RobustCurriculumScheduler(trainer.weights, total_epochs=args.epochs)
    best_kappa = -1.0

    print("\n>>> Start Training with Robust Curriculum Schedule <<<")

    for epoch in range(args.epochs):
        phase_name = scheduler.step(epoch)
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} | {phase_name} ===")

        # 定义累加器，用于统计所有 Loss
        epoch_metrics = defaultdict(float)
        steps = 0

        # --- Training Loop ---
        for batch in source_loader:
            # update 返回的是一个包含所有 loss 分项的字典
            step_metrics = trainer.update(batch, optimizer, has_masks=True)

            # 累加所有指标
            for k, v in step_metrics.items():
                epoch_metrics[k] += v
            steps += 1

            # 简单的进度展示
            if steps % 50 == 0:
                print(f"    Step {steps:03d}: Loss={step_metrics['loss']:.4f}", end="\r")

        # --- Epoch Summary (打印所有 Loss) ---
        print(f"\n    >>> Epoch {epoch + 1} Summary:")
        log_str = "    "
        for k, v in epoch_metrics.items():
            avg_val = v / steps
            log_str += f"{k}={avg_val:.4f} | "
        print(log_str)

        # --- Validation Loop (每 5 Epoch) ---
        if (epoch + 1) % 5 == 0:
            print(f"\n    --- Validation (Classification) ---")
            total_kappa = 0

            for name, loader in target_loaders.items():
                val_res = trainer.validate(loader)
                print(f"    [{name}] Acc: {val_res['Accuracy']:.4f} | Kappa: {val_res['Kappa']:.4f}")
                total_kappa += val_res['Kappa']

            avg_kappa = total_kappa / len(target_loaders)
            print(f"    >>> Avg Kappa: {avg_kappa:.4f}")

            # 保存最佳模型 (Phase 3 之后才开始算最佳，因为 Phase 1/2 分类头没训练)
            if epoch > scheduler.phase2_end and avg_kappa > best_kappa:
                best_kappa = avg_kappa
                save_path = f"checkpoints/best_kappa_{best_kappa:.4f}.pth"
                torch.save(trainer.student.state_dict(), save_path)
                print(f"    [Save] 🌟 New Best Model saved to {save_path}")

        # 定期保存 Checkpoint
        if (epoch + 1) % 5 == 0:
             torch.save(trainer.student.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
