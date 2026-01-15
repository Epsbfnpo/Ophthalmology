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

    os.makedirs("checkpoints", exist_ok=True)
    print(f"✅ Created checkpoint directory: {os.path.abspath('checkpoints')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Using device: {device}")

    source_domains = [DomainConfig(name=name, root=os.path.join(args.data_root, name), has_masks=True) for name in args.source_domains]
    target_domains = [DomainConfig(name=name, root=os.path.join(args.data_root, name), has_masks=False) for name in args.target_domains]

    print(f"[Init] Loading concept bank from {args.concept_bank}")
    concept_bank = torch.load(args.concept_bank, map_location=device, weights_only=True)

    weights = LossWeights()
    trainer = HCMTLGGDRNetTrainer(concept_bank=concept_bank, device=device, weights=weights)

    optimizer = torch.optim.AdamW(trainer.student.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"[Init] Loading Source Domains: {args.source_domains}")
    source_datasets = [MedicalDataset(domain, augment=True) for domain in source_domains]
    source_dataset = ConcatDataset(source_datasets)
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

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

        epoch_metrics = defaultdict(float)
        steps = 0

        for batch in source_loader:
            step_metrics = trainer.update(batch, optimizer, has_masks=True)
            for k, v in step_metrics.items():
                epoch_metrics[k] += v
            steps += 1

            if steps % 50 == 0:
                print(f"    Step {steps:03d}: Loss={step_metrics['loss']:.4f}", end="\r")

        print(f"\n    >>> Epoch {epoch + 1} Summary:")
        log_str = "    "
        for k, v in epoch_metrics.items():
            avg_val = v / steps
            log_str += f"{k}={avg_val:.4f} | "
        print(log_str)

        if (epoch + 1) % 5 == 0:
            print(f"\n    --- Validation (Classification) ---")
            total_kappa = 0

            for name, loader in target_loaders.items():
                val_res = trainer.validate(loader)
                print(f"    [{name}] Acc: {val_res['Accuracy']:.4f} | Kappa: {val_res['Kappa']:.4f}")
                total_kappa += val_res['Kappa']

            avg_kappa = total_kappa / len(target_loaders)
            print(f"    >>> Avg Kappa: {avg_kappa:.4f}")

            if epoch > scheduler.phase2_end and avg_kappa > best_kappa:
                best_kappa = avg_kappa
                save_path = f"checkpoints/best_kappa_{best_kappa:.4f}.pth"
                state_dict = trainer.student.module.state_dict() if isinstance(trainer.student, torch.nn.DataParallel) else trainer.student.state_dict()
                torch.save(state_dict, save_path)
                print(f"    [Save] 🌟 New Best Model saved to {save_path}")

        if (epoch + 1) % 5 == 0:
            save_path = f"checkpoints/epoch_{epoch + 1}.pth"
            state_dict = trainer.student.module.state_dict() if isinstance(trainer.student, torch.nn.DataParallel) else trainer.student.state_dict()
            torch.save(state_dict, save_path)
            print(f"    [Save] Checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()