import argparse
import os
from typing import List

import torch
from torch.utils.data import DataLoader, ConcatDataset

from new.algorithms_hc import HCMTLGGDRNetTrainer, LossWeights
from new.dataset.medical_dataset import DomainConfig, MedicalDataset


def build_domains(root: str, names: List[str], has_masks: bool) -> List[DomainConfig]:
    return [DomainConfig(name=name, root=os.path.join(root, name), has_masks=has_masks) for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="HC-MT-LG-GDRNet training")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--source_domains", nargs="+", required=True)
    parser.add_argument("--target_domains", nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lambda_seg", type=float, default=10.0)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--algorithm", type=str, default="HC_MT_LG_GDRNet")
    parser.add_argument("--concept_bank", type=str, default="./concepts.pth")
    args = parser.parse_args()

    concept_bank = None
    if os.path.exists(args.concept_bank):
        concept_bank = torch.load(args.concept_bank, map_location="cpu")

    weights = LossWeights(seg=args.lambda_seg, reg=args.lambda_reg)
    trainer = HCMTLGGDRNetTrainer(concept_bank=concept_bank, weights=weights)

    source_domains = build_domains(args.data_root, args.source_domains, has_masks=True)
    source_datasets = [MedicalDataset(domain, augment=True) for domain in source_domains]
    source_loader = DataLoader(ConcatDataset(source_datasets), batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(trainer.student.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        for batch in source_loader:
            metrics = trainer.update(batch, optimizer, has_masks=True)
        print(f"Epoch {epoch + 1}: loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
