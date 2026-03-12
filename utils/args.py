import argparse
import os
from configs.defaults import _C as cfg_default


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument('--time-limit', type=int, default=0, help='time limit in seconds')
    parser.add_argument('--output', type=str, default=None, help='base output directory')
    parser.add_argument('--target-domain', type=str, default=None, help='Specify the single target domain (e.g., APTOS). Others will be sources.')
    args = parser.parse_args()
    return args


def setup_cfg(args):
    cfg = cfg_default.clone()
    cfg.defrost()

    if args.output is not None:
        cfg.OUT_DIR = args.output

    if args.target_domain is not None:
        ALL_DOMAINS = ["APTOS", "DDR", "DEEPDR", "FGADR", "IDRID", "MESSIDOR", "RLDR", "EYEPACS"]
        current_target = args.target_domain

        if current_target not in ALL_DOMAINS:
            raise ValueError(f"Target domain {current_target} not found in {ALL_DOMAINS}")

        cfg.DATASET.TARGET_DOMAINS = [current_target]
        cfg.DATASET.SOURCE_DOMAINS = [d for d in ALL_DOMAINS if d != current_target]

        cfg.OUT_DIR = os.path.join(cfg.OUT_DIR, current_target)

        print(f"================ [Auto Config] ================")
        print(f"Sources (Multi): {cfg.DATASET.SOURCE_DOMAINS}")
        print(f"Target (Single): {cfg.DATASET.TARGET_DOMAINS}")
        print(f"Output Dir: {cfg.OUT_DIR}")
        print(f"===============================================")

        cfg.OUTPUT_PATH = f"{cfg.ALGORITHM}_{cfg.DG_MODE}_to_{current_target}"
    else:
        sources_str = '_'.join(cfg.DATASET.SOURCE_DOMAINS)
        cfg.OUTPUT_PATH = f"{cfg.ALGORITHM}_{cfg.DG_MODE}_{sources_str}"

    cfg.freeze()
    return cfg
