import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from collections import defaultdict, Counter

from algorithms_hc import HCMTLGGDRNetTrainer, LossWeights
from dataset.medical_dataset import DomainConfig, MedicalDataset


class RobustCurriculumScheduler:
    def __init__(self, trainer: HCMTLGGDRNetTrainer, total_epochs: int):
        self.trainer = trainer
        self.total_epochs = total_epochs
        self.phase1_end = 5
        self.phase2_end = 15

    def step(self, epoch: int):
        current_weights = LossWeights()

        if epoch < self.phase1_end:
            # Phase 1: Vision Warm-up. CLS weight set to 1.0 to ensure classifier training starts.
            current_weights.seg = 5.0
            current_weights.distill = 1.0
            current_weights.concept = 0.0
            current_weights.ib = 0.0
            current_weights.cls = 1.0
            current_weights.reg = 0.0
            phase_name = "Phase 1: Vision Warm-up"
        elif epoch < self.phase2_end:
            # Phase 2: Semantic Alignment
            current_weights.seg = 1.0
            current_weights.distill = 0.5
            current_weights.concept = 1.0
            current_weights.ib = 0.01
            current_weights.cls = 1.0
            current_weights.reg = 0.1
            phase_name = "Phase 2: Semantic Alignment"
        else:
            # Phase 3: Final Logic Tuning
            current_weights.seg = 1.0
            current_weights.distill = 0.0
            current_weights.concept = 0.5
            current_weights.ib = 0.01
            current_weights.cls = 2.0
            current_weights.reg = 1.0
            phase_name = "Phase 3: Final Logic Tuning"

        self.trainer.update_weights(current_weights)
        return phase_name


def calculate_class_weights(datasets):
    print("[Init] Calculating Class Weights...")
    all_labels = []
    for ds in datasets:
        all_labels.extend(list(ds.labels.values()))

    counter = Counter(all_labels)
    total_count = len(all_labels)
    num_classes = 5

    weights = []
    for i in range(num_classes):
        count = counter[i]
        if count == 0:
            w = 1.0
        else:
            w = total_count / (num_classes * count)
        weights.append(w)

    weights = np.array(weights)
    weights = weights / weights.mean()
    return torch.FloatTensor(weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--source_domains", nargs="+", required=True)
    parser.add_argument("--target_domains", nargs="+", required=True)
    parser.add_argument("--concept_bank", type=str, default="concepts.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Using device: {device}")

    source_domains = [DomainConfig(name=name, root=os.path.join(args.data_root, name), has_masks=True) for name in
                      args.source_domains]
    target_domains = [DomainConfig(name=name, root=os.path.join(args.data_root, name), has_masks=False) for name in
                      args.target_domains]

    print(f"[Init] Loading concept bank from {args.concept_bank}")
    concept_bank = torch.load(args.concept_bank, map_location=device, weights_only=True)

    print(f"[Init] Loading Source Domains: {args.source_domains}")
    source_datasets_list = [MedicalDataset(domain, augment=True) for domain in source_domains]

    class_weights_loss = calculate_class_weights(source_datasets_list).to(device)
    print(f"⚖️ [Loss Weights] {class_weights_loss}")

    # WeightedRandomSampler setup
    print("[Init] Building WeightedRandomSampler for Balanced Batching...")

    all_targets = []
    for ds in source_datasets_list:
        all_targets.extend(list(ds.labels.values()))

    class_counts = Counter(all_targets)
    class_sample_weights = {}
    for c, count in class_counts.items():
        class_sample_weights[c] = 1.0 / count

    print(f"📊 [Sampler] Class Counts: {dict(sorted(class_counts.items()))}")

    samples_weights = [class_sample_weights[t] for t in all_targets]
    samples_weights = torch.DoubleTensor(samples_weights)

    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    trainer = HCMTLGGDRNetTrainer(
        concept_bank=concept_bank,
        device=device,
        weights=LossWeights(),
        class_weights=class_weights_loss
    )

    optimizer = torch.optim.AdamW(trainer.student.parameters(), lr=args.lr, weight_decay=1e-4)

    source_dataset = ConcatDataset(source_datasets_list)

    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"[Init] Loading Target Domains (Validation): {args.target_domains}")
    target_loaders = {}
    for domain in target_domains:
        val_ds = MedicalDataset(domain, augment=False)
        target_loaders[domain.name] = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                                 pin_memory=True)

    scheduler = RobustCurriculumScheduler(trainer, total_epochs=args.epochs)
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
                val_res = trainer.validate(loader, domain_name=name)
                print(f"    [{name}] Acc: {val_res['Accuracy']:.4f} | Kappa: {val_res['Kappa']:.4f}")
                total_kappa += val_res['Kappa']

            avg_kappa = total_kappa / len(target_loaders)
            print(f"    >>> Avg Kappa: {avg_kappa:.4f}")

            if avg_kappa > best_kappa:
                best_kappa = avg_kappa
                save_path = f"checkpoints/best_kappa_{best_kappa:.4f}.pth"
                state_dict = trainer.student.module.state_dict() if isinstance(trainer.student,
                                                                               torch.nn.DataParallel) else trainer.student.state_dict()
                torch.save(state_dict, save_path)
                print(f"    [Save] 🌟 New Best Model saved to {save_path}")

        if (epoch + 1) % 5 == 0:
            save_path = f"checkpoints/epoch_{epoch + 1}.pth"
            state_dict = trainer.student.module.state_dict() if isinstance(trainer.student,
                                                                           torch.nn.DataParallel) else trainer.student.state_dict()
            torch.save(state_dict, save_path)
            print(f"    [Save] Checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()