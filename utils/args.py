import argparse
from configs.defaults import _C as cfg_default

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--root", type=str, default="../DGDATA/", help="path to dataset")
    parser.add_argument("--root", type=str, default="./GDRBench", help="path to dataset")
    
    parser.add_argument("--algorithm", type=str, default='GDRNet', help='check in algorithms.py')
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--backbone1", type=str, default="resnet50")
    parser.add_argument("--backbone2", type=str, default="resnet50")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DGDR")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DGDR")
    parser.add_argument("--dg_mode", type=str, default='DG', help="DG or ESDG")
    
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--val_ep", type=int, default=10)
    parser.add_argument("--output", type=str, default='test')
    parser.add_argument("--override", action="store_true")
    
    ##### for Quat
    parser.add_argument("--q_steps", type=int, default=150)
    parser.add_argument("--quant", type=int, default=1)
    parser.add_argument("--swad", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42, help="Seed for everything else")

    parser.add_argument("--epochs", type=int, default=100, help="Max epoches for training")

    parser.add_argument("--model_path", type=str)


    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--learning_rate1", type=float, default=0.001)
    parser.add_argument("--learning_rate2", type=float, default=0.001)

    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--mask_ratio", type=float, default=0.7)
    parser.add_argument("--block_size", type=int, default=32)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fastmoco", type=float, default=0.0)
    parser.add_argument("--masked", type=float, default=0.5)
    parser.add_argument("--sup", type=float, default=1.0)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--positive", type=int, default=4)
    parser.add_argument("--p", type=float, default=0.2)
    parser.add_argument("--smooth", type=float, default=0.4)

    return parser.parse_args()

def setup_cfg(args):
    cfg = cfg_default.clone()
    cfg.RANDOM = args.random
    cfg.OUTPUT_PATH = args.output
    cfg.OVERRIDE = args.override
    cfg.DG_MODE = args.dg_mode
    
    cfg.ALGORITHM = args.algorithm
    cfg.BACKBONE = args.backbone
    cfg.BACKBONE1 = args.backbone1
    cfg.BACKBONE2 = args.backbone2

    
    cfg.DATASET.ROOT = args.root
    cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    cfg.DATASET.TARGET_DOMAINS = args.target_domains
    cfg.DATASET.NUM_CLASSES = args.num_classes
    cfg.MASK_RATIO = args.mask_ratio
    cfg.BLOCK = args.block_size



    cfg.VAL_EPOCH = args.val_ep
    
    if args.dg_mode == 'DG':
        cfg.merge_from_file("./configs/datasets/GDRBench.yaml")
    elif args.dg_mode == 'ESDG':
        cfg.merge_from_file("./configs/datasets/GDRBench_ESDG.yaml")
    else:
        raise ValueError('Wrong type')

    cfg.LEARNING_RATE = args.learning_rate
    cfg.LEARNING_RATE1 = args.learning_rate1
    cfg.LEARNING_RATE2 = args.learning_rate2
    cfg.FASTMOCO = args.fastmoco
    cfg.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed
    cfg.MASKED = args.masked
    cfg.EPOCHS = args.epochs
    cfg.SUP = args.sup
    cfg.KD = args.kd
    cfg.POSITIVE = args.positive
    cfg.P = args.p
    cfg.SMOOTH = args.smooth





    return cfg

