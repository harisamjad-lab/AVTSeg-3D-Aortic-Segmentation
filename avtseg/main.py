from __future__ import annotations

import argparse
from pathlib import Path

from config import load_config
from dataio import make_kfold_splits, paired_cases, preprocess_case
from inference import infer_full
from stage1 import infer_stage1, train_stage1
from stage2 import build_stage2_records, train_stage2
from utils import load_json, maybe_mkdir, save_json, set_seed


def build_argparser():
    p = argparse.ArgumentParser(description="Local AVTSeg-style two-stage pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_dir", default="data")
    common.add_argument("--workdir", default="workdir")
    common.add_argument("--fold", type=int, default=0)
    common.add_argument("--num_workers", type=int, default=4)

    sub.add_parser("split", parents=[common])
    sub.add_parser("preprocess", parents=[common])
    sub.add_parser("train_stage1", parents=[common])
    sub.add_parser("infer_stage1", parents=[common])
    sub.add_parser("build_stage2_cache", parents=[common])
    sub.add_parser("train_stage2", parents=[common])
    sub.add_parser("infer_full", parents=[common])
    sub.add_parser("run_all", parents=[common])
    return p


def prepare_cfg(args):
    cfg = load_config()
    cfg.data.data_dir = args.data_dir
    cfg.data.workdir = args.workdir
    cfg.data.val_fold = args.fold
    cfg.data.num_workers = args.num_workers
    return cfg


def load_or_make_splits(cfg):
    split_path = Path(cfg.data.workdir) / "splits" / "splits.json"
    maybe_mkdir(split_path.parent)
    if split_path.exists():
        return load_json(split_path)
    return make_kfold_splits(cfg.data.data_dir, str(split_path), k=cfg.data.k_folds, seed=cfg.runtime.seed)


def preprocess_all(cfg):
    cases = paired_cases(cfg.data.data_dir)
    out_dir = Path(cfg.data.workdir) / "preprocessed"
    maybe_mkdir(out_dir)
    preprocessed_cases = [
        preprocess_case(
            case,
            str(out_dir),
            spacing=cfg.preprocess.spacing,
            intensity_shift_threshold=cfg.preprocess.intensity_shift_threshold,
        )
        for case in cases
    ]
    save_json(preprocessed_cases, out_dir / "preprocessed_cases.json")
    return preprocessed_cases


def load_preprocessed_cases(cfg):
    path = Path(cfg.data.workdir) / "preprocessed" / "preprocessed_cases.json"
    if not path.exists():
        return preprocess_all(cfg)
    return load_json(path)


def split_cases(preprocessed_cases, splits, fold):
    split = splits[f"fold_{fold}"]
    train_cases = [c for c in preprocessed_cases if c["id"] in split["train"]]
    val_cases = [c for c in preprocessed_cases if c["id"] in split["val"]]
    return train_cases, val_cases, split


def build_stage2_cache_files(preprocessed_cases, split, cfg):
    out_dir = Path(cfg.data.workdir) / "stage2"
    maybe_mkdir(out_dir)
    train_records = build_stage2_records(preprocessed_cases, split["train"], cfg)
    val_records = build_stage2_records(preprocessed_cases, split["val"], cfg)
    save_json(train_records, out_dir / f"train_records_fold{cfg.data.val_fold}.json")
    save_json(val_records, out_dir / f"val_records_fold{cfg.data.val_fold}.json")
    return train_records, val_records


def load_stage2_cache_files(cfg):
    out_dir = Path(cfg.data.workdir) / "stage2"
    train_records = load_json(out_dir / f"train_records_fold{cfg.data.val_fold}.json")
    val_records = load_json(out_dir / f"val_records_fold{cfg.data.val_fold}.json")
    return train_records, val_records


def main():
    args = build_argparser().parse_args()
    cfg = prepare_cfg(args)
    cfg.save(Path(cfg.data.workdir) / "config_snapshot.json")
    set_seed(cfg.runtime.seed)

    if args.cmd == "split":
        load_or_make_splits(cfg)
        return

    if args.cmd == "preprocess":
        preprocess_all(cfg)
        return

    splits = load_or_make_splits(cfg)
    preprocessed_cases = load_preprocessed_cases(cfg)
    train_cases, val_cases, split = split_cases(preprocessed_cases, splits, cfg.data.val_fold)

    if args.cmd == "train_stage1":
        train_stage1(train_cases, val_cases, cfg)
    elif args.cmd == "infer_stage1":
        infer_stage1(preprocessed_cases, cfg)
    elif args.cmd == "build_stage2_cache":
        build_stage2_cache_files(preprocessed_cases, split, cfg)
    elif args.cmd == "train_stage2":
        train_records, val_records = load_stage2_cache_files(cfg)
        train_stage2(train_records, val_records, cfg)
    elif args.cmd == "infer_full":
        infer_full(preprocessed_cases, cfg)
    elif args.cmd == "run_all":
        print("=== Stage 1: training coarse model ===")
        train_stage1(train_cases, val_cases, cfg)
        print("=== Stage 1: coarse inference ===")
        infer_stage1(preprocessed_cases, cfg)
        print("=== Stage 2: building cache ===")
        train_records, val_records = build_stage2_cache_files(preprocessed_cases, split, cfg)
        print("=== Stage 2: training refinement model ===")
        train_stage2(train_records, val_records, cfg)
        print("=== Final: full-resolution inference ===")
        infer_full(preprocessed_cases, cfg)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
