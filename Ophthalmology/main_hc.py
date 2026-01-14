import argparse
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset

from algorithms_hc import HCMTLGGDRNetTrainer, LossWeights
from dataset.medical_dataset import DomainConfig, MedicalDataset


class RobustCurriculumScheduler:
    def __init__(self, weights: LossWeights, total_epochs: int):
        self.weights = weights
        self.total_epochs = total_epochs

        # 自动计算阶段边界 (基于 30% - 40% - 30% 的黄金比例)
        self.phase1_end = int(total_epochs * 0.3)  # 约第 15 Epoch
        self.phase2_end = int(total_epochs * 0.7)  # 约第 35 Epoch

    def step(self, epoch: int):
        if epoch < self.phase1_end:
            # === Phase 1: 视觉热身 (Vision Warm-up) ===
            # 策略: 哪怕天塌下来，也先把分割搞定。
            # 权重: Seg=10 (绝对主导), Distill=1 (辅助)
            self.weights.seg = 10.0
            self.weights.distill = 1.0

            # 关闭干扰项
            self.weights.concept = 0.0
            self.weights.ib = 0.0
            self.weights.cls = 0.0
            self.weights.reg = 0.0

            return "Phase 1: Vision Warm-up (Seg Focus)"

        elif epoch < self.phase2_end:
            # === Phase 2: 语义对齐 (Semantic Alignment) ===
            # 策略: 视觉好了，现在强迫 Projector 去对齐 BioMedCLIP。
            # 权重: Concept=5 (拉大，让MLP快速收敛), IB=0.1 (去噪)
            self.weights.seg = 1.0      # 降低 Seg 权重，维持即可
            self.weights.distill = 1.0

            self.weights.concept = 5.0  # 核心任务
            self.weights.ib = 0.1

            # 依然不诊断
            self.weights.cls = 0.0
            self.weights.reg = 0.0

            return "Phase 2: Semantic Alignment (CLIP Focus)"

        else:
            # === Phase 3: 全局微调 (Global Fine-tuning) ===
            # 策略: 特征和概念都好了，现在开分类头，并加上逻辑锁。
            # 权重: Cls=1 (主任务), Reg=1 (逻辑约束)
            self.weights.seg = 1.0
            self.weights.distill = 1.0
            self.weights.concept = 1.0  # 恢复正常
            self.weights.ib = 0.1

            self.weights.cls = 1.0      # 终于开始看病了
            self.weights.reg = 1.0      # KCCL 逻辑锁开启

            return "Phase 3: Final Logic Tuning (Diagnosis)"


def build_domains(root: str, names: list, has_masks: bool) -> list:
    return [DomainConfig(name=name, root=os.path.join(root, name), has_masks=has_masks) for name in names]


def main():
    parser = argparse.ArgumentParser(description="HC-MT-LG-GDRNet One-Shot Training")
    parser.add_argument("--data_root", type=str, default="./data", help="数据根目录")
    parser.add_argument("--source_domains", nargs="+", required=True, help="源域数据集列表")
    parser.add_argument("--target_domains", nargs="+", required=True, help="目标域数据集列表(仅测试用)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50, help="建议设置为 50 以保证课程完整")
    parser.add_argument("--concept_bank", type=str, default="./concepts.pth")
    # 注意：这里不再需要 lambda 参数了，Scheduler 全权接管
    args = parser.parse_args()

    # 1. 加载 BioMedCLIP 概念库
    concept_bank = None
    if os.path.exists(args.concept_bank):
        concept_bank = torch.load(args.concept_bank, map_location="cpu")
        print(f"[Init] Loaded concept bank: {concept_bank.shape}")
    else:
        print("[Warning] Concept bank not found! Language guidance will be disabled.")

    # 2. 初始化权重对象 (初始值不重要，Scheduler 会覆盖)
    weights = LossWeights()

    # 3. 初始化 Trainer
    trainer = HCMTLGGDRNetTrainer(concept_bank=concept_bank, weights=weights)

    # 4. 准备数据
    print(f"[Init] Loading Source Domains: {args.source_domains}")
    source_domains = build_domains(args.data_root, args.source_domains, has_masks=True)
    source_datasets = [MedicalDataset(domain, augment=True) for domain in source_domains]
    # 这里的 num_workers 设置为 4 或 8 以加速数据读取
    source_loader = DataLoader(
        ConcatDataset(source_datasets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 5. 优化器 (使用较小的 lr 保证微调稳定)
    optimizer = torch.optim.Adam(trainer.student.parameters(), lr=1e-4)

    # ==========================================
    # [关键] 启动自动调度器
    # ==========================================
    scheduler = RobustCurriculumScheduler(trainer.weights, total_epochs=args.epochs)

    # 6. 开始训练
    print("\n>>> Start Training with Robust Curriculum Schedule <<<")
    for epoch in range(args.epochs):
        # 这一步会自动修改 trainer.weights 中的数值
        phase_name = scheduler.step(epoch)

        print(f"\n=== Epoch {epoch + 1}/{args.epochs} | {phase_name} ===")
        # 打印当前权重给用户看，以此为证
        print(
            f"    [Weights] Seg={weights.seg:.1f} | Concept={weights.concept:.1f} | "
            f"Cls={weights.cls:.1f} | Reg={weights.reg:.1f}"
        )

        epoch_loss = 0.0
        steps = 0

        # 训练一个 Epoch
        for batch in source_loader:
            metrics = trainer.update(batch, optimizer, has_masks=True)
            epoch_loss += metrics["loss"]
            steps += 1

            if steps % 50 == 0:
                print(
                    f"    Step {steps:03d}: Loss={metrics['loss']:.4f} "
                    f"(Seg={metrics['seg']:.4f}, Concept={metrics['concept']:.4f})"
                )

        print(f"    >>> Epoch {epoch + 1} Avg Loss: {epoch_loss / steps:.4f}")

        # 保存模型 (建议只保存 Phase 3 的模型)
        if epoch >= scheduler.phase2_end:
            save_path = f"checkpoints/hc_gdrnet_epoch_{epoch+1}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(trainer.student.state_dict(), save_path)
            print(f"    [Save] Model saved to {save_path}")


if __name__ == "__main__":
    main()
