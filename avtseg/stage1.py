from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio import Stage1Dataset, load_volume_as_numpy
from models import build_model
from utils import dice_np, largest_connected_component, normalize_ct, save_json


def save_overlay_jpeg(image_zyx, mask_zyx, out_path: Path, title: str) -> None:
    slice_scores = mask_zyx.reshape(mask_zyx.shape[0], -1).sum(axis=1)
    z = int(slice_scores.argmax()) if slice_scores.max() > 0 else image_zyx.shape[0] // 2

    plt.figure(figsize=(6, 6))
    plt.imshow(image_zyx[z], cmap="gray")
    plt.imshow((mask_zyx[z] > 0).astype(float), alpha=0.4, cmap="Reds")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def validate_stage1(model, cases_for_val, device, cfg):
    model.eval()
    dices = []
    with torch.no_grad():
        for case in tqdm(cases_for_val, desc="stage1 val", leave=False):
            img, _ = load_volume_as_numpy(case["pre_image"])
            lab, _ = load_volume_as_numpy(case["pre_label"])
            img = normalize_ct(img, cfg.preprocess.intensity_min, cfg.preprocess.intensity_max)
            inp = torch.from_numpy(img[None, None]).float().to(device)
            logits = sliding_window_inference(
                inp,
                roi_size=cfg.stage1.patch_size,
                sw_batch_size=cfg.stage1.sw_batch_size,
                predictor=model,
                overlap=cfg.stage1.overlap,
            )
            pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > cfg.stage1.threshold).astype(np.uint8)
            pred = largest_connected_component(pred)
            dices.append(dice_np(pred, lab > 0))
    return float(np.mean(dices)) if dices else 0.0


def train_stage1(train_cases, val_cases, cfg):
    device = torch.device(cfg.runtime.device)
    train_ds = Stage1Dataset(
        train_cases,
        patch_size=cfg.stage1.patch_size,
        intensity_min=cfg.preprocess.intensity_min,
        intensity_max=cfg.preprocess.intensity_max,
        fg_prob=cfg.stage1.train_fg_prob,
        training=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.stage1.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.stage1.lr, weight_decay=cfg.stage1.weight_decay)
    loss_fn = DiceCELoss(sigmoid=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.runtime.amp and device.type == "cuda"))

    stage1_dir = Path(cfg.data.workdir) / "stage1"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    best_path = stage1_dir / f"best_fold{cfg.data.val_fold}.pt"
    hist_path = stage1_dir / f"history_fold{cfg.data.val_fold}.json"

    best_dice = -1.0
    history = []
    for epoch in range(1, cfg.stage1.max_epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"stage1 epoch {epoch}/{cfg.stage1.max_epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.runtime.amp and device.type == "cuda")):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses))

        print(f"[stage1] running validation for epoch {epoch}...")
        val_dice = validate_stage1(model, val_cases, device, cfg)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_dice": val_dice})
        print(f"[stage1] epoch={epoch} train_loss={np.mean(losses):.4f} val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice}, best_path)
            print(f"[stage1] saved best checkpoint -> {best_path}")
    save_json(history, hist_path)
    return str(best_path)


def infer_stage1(preprocessed_cases, cfg):
    device = torch.device(cfg.runtime.device)
    model = build_model().to(device)
    ckpt_path = Path(cfg.data.workdir) / "stage1" / f"best_fold{cfg.data.val_fold}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(cfg.data.workdir) / "stage1" / "coarse_masks"
    jpeg_dir = Path(cfg.data.workdir) / "stage1" / "jpeg_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    jpeg_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for case in tqdm(preprocessed_cases, desc="stage1 infer"):
            img_sitk = sitk.ReadImage(case["pre_image"])
            img_raw = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
            img = normalize_ct(img_raw, cfg.preprocess.intensity_min, cfg.preprocess.intensity_max)
            inp = torch.from_numpy(img[None, None]).float().to(device)
            logits = sliding_window_inference(
                inp,
                roi_size=cfg.stage1.patch_size,
                sw_batch_size=cfg.stage1.sw_batch_size,
                predictor=model,
                overlap=cfg.stage1.overlap,
            )
            pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > cfg.stage1.threshold).astype(np.uint8)
            pred = largest_connected_component(pred)
            pred_sitk = sitk.GetImageFromArray(pred)
            pred_sitk.CopyInformation(img_sitk)
            sitk.WriteImage(pred_sitk, str(out_dir / f"{case['id']}_coarse.mha"))
            save_overlay_jpeg(img_raw, pred, jpeg_dir / f"{case['id']}_coarse.jpg", f"{case['id']} coarse stage-1")
    return str(out_dir)
