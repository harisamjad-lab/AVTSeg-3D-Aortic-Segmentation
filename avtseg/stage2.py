from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio import Stage2PatchDataset, load_volume_as_numpy
from models import build_model
from utils import farthest_point_subsample, pseudo_centerline_from_mask, save_json


def build_stage2_records(preprocessed_cases, split_case_ids, cfg):
    coarse_dir = Path(cfg.data.workdir) / "stage1" / "coarse_masks"
    records = []
    for case in tqdm([c for c in preprocessed_cases if c["id"] in split_case_ids], desc="build stage2 records"):
        coarse, _ = load_volume_as_numpy(str(coarse_dir / f"{case['id']}_coarse.mha"))
        centers = pseudo_centerline_from_mask(coarse)
        centers = farthest_point_subsample(
            centers,
            target_spacing=cfg.stage2.center_spacing_vox,
            max_points=cfg.stage2.max_centers_per_case,
        )
        for c in centers:
            records.append({
                "case_id": case["id"],
                "image": case["pre_image"],
                "label": case["pre_label"],
                "center": c.tolist(),
            })
    return records


def validate_stage2(model, loader, device, cfg):
    model.eval()
    dices = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="stage2 val", leave=False):
            images = batch["image"].float().to(device)
            labels = batch["label"].float().to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > cfg.stage2.threshold).float()
            inter = (preds * labels).sum().item() * 2.0
            denom = preds.sum().item() + labels.sum().item() + 1e-6
            dices.append(inter / denom)
    return float(np.mean(dices)) if dices else 0.0


def train_stage2(train_records, val_records, cfg):
    device = torch.device(cfg.runtime.device)
    train_ds = Stage2PatchDataset(
        train_records,
        patch_size=cfg.stage2.patch_size,
        intensity_min=cfg.preprocess.intensity_min,
        intensity_max=cfg.preprocess.intensity_max,
        training=True,
    )
    val_ds = Stage2PatchDataset(
        val_records,
        patch_size=cfg.stage2.patch_size,
        intensity_min=cfg.preprocess.intensity_min,
        intensity_max=cfg.preprocess.intensity_max,
        training=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.stage2.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.stage2.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.stage2.lr, weight_decay=cfg.stage2.weight_decay)
    loss_fn = DiceCELoss(sigmoid=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.runtime.amp and device.type == "cuda"))

    stage2_dir = Path(cfg.data.workdir) / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    best_path = stage2_dir / f"best_fold{cfg.data.val_fold}.pt"
    hist_path = stage2_dir / f"history_fold{cfg.data.val_fold}.json"

    best_dice = -1.0
    history = []
    for epoch in range(1, cfg.stage2.max_epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"stage2 epoch {epoch}/{cfg.stage2.max_epochs}")
        for batch in pbar:
            images = batch["image"].float().to(device)
            labels = batch["label"].float().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.runtime.amp and device.type == "cuda")):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses))
        print(f"[stage2] running validation for epoch {epoch}...")
        val_dice = validate_stage2(model, val_loader, device, cfg)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_dice": val_dice})
        print(f"[stage2] epoch={epoch} train_loss={np.mean(losses):.4f} val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice}, best_path)
            print(f"[stage2] saved best checkpoint -> {best_path}")
    save_json(history, hist_path)
    return str(best_path)
