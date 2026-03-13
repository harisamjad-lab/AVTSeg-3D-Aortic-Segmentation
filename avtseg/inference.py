from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from dataio import load_volume_as_numpy
from models import build_model
from utils import (
    binary_postprocess,
    dice_np,
    farthest_point_subsample,
    normalize_ct,
    paste_patch_additive,
    pseudo_centerline_from_mask,
    crop_patch,
    save_json,
)


def infer_case_full_pipeline(case, stage2_model, cfg, device):
    img, img_sitk = load_volume_as_numpy(case["pre_image"])
    gt, _ = load_volume_as_numpy(case["pre_label"])
    coarse, _ = load_volume_as_numpy(str(Path(cfg.data.workdir) / "stage1" / "coarse_masks" / f"{case['id']}_coarse.mha"))

    centers = pseudo_centerline_from_mask(coarse)
    centers = farthest_point_subsample(
        centers,
        target_spacing=cfg.stage2.center_spacing_vox,
        max_points=cfg.stage2.max_centers_per_case,
    )

    accum = np.zeros_like(img, dtype=np.float32)
    weight = np.zeros_like(img, dtype=np.float32)
    stage2_model.eval()
    with torch.no_grad():
        for center in centers:
            patch = crop_patch(img, center, cfg.stage2.patch_size)
            patch = normalize_ct(patch, cfg.preprocess.intensity_min, cfg.preprocess.intensity_max)
            inp = torch.from_numpy(patch[None, None]).float().to(device)
            prob = torch.sigmoid(stage2_model(inp))[0, 0].cpu().numpy().astype(np.float32)
            paste_patch_additive(accum, weight, prob, center)

    if weight.max() == 0:
        fused = coarse.astype(np.float32)
    else:
        fused = accum / np.maximum(weight, 1e-6)
        fused[weight == 0] = coarse[weight == 0].astype(np.float32)

    pred = (fused > cfg.stage2.threshold).astype(np.uint8)
    pred = binary_postprocess(pred, min_component_size=cfg.stage2.min_component_size)
    return {
        "pred": pred,
        "dice": dice_np(pred, gt > 0),
        "pre_image_sitk": img_sitk,
    }


def export_prediction_to_original_geometry(pre_pred_path: str, original_image_path: str, out_path: str):
    pred_pre = sitk.ReadImage(str(pre_pred_path))
    ref_orig = sitk.ReadImage(str(original_image_path))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_orig)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    pred_orig = resampler.Execute(pred_pre)
    sitk.WriteImage(pred_orig, str(out_path))


def infer_full(preprocessed_cases, cfg):
    device = torch.device(cfg.runtime.device)
    model = build_model().to(device)
    ckpt_path = Path(cfg.data.workdir) / "stage2" / f"best_fold{cfg.data.val_fold}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    pred_pre_dir = Path(cfg.data.workdir) / "predictions" / "preprocessed"
    pred_orig_dir = Path(cfg.data.workdir) / "predictions" / "original_geometry"
    jpeg_dir = Path(cfg.data.workdir) / "predictions" / "jpeg_overlays"
    pred_pre_dir.mkdir(parents=True, exist_ok=True)
    pred_orig_dir.mkdir(parents=True, exist_ok=True)
    jpeg_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    for case in tqdm(preprocessed_cases, desc="full inference"):
        out = infer_case_full_pipeline(case, model, cfg, device)

        # Save prediction in preprocessed and original geometries
        pre_path = pred_pre_dir / f"{case['id']}_pred.mha"
        pred_sitk = sitk.GetImageFromArray(out["pred"].astype(np.uint8))
        pred_sitk.CopyInformation(out["pre_image_sitk"])
        sitk.WriteImage(pred_sitk, str(pre_path))
        export_prediction_to_original_geometry(pre_path, case["orig_image"], pred_orig_dir / f"{case['id']}_pred.mha")

        # Create a JPEG overlay for quick inspection on any PC.
        img_arr = sitk.GetArrayFromImage(out["pre_image_sitk"]).astype(np.float32)
        pred_arr = out["pred"].astype(np.uint8)
        if img_arr.shape[0] != pred_arr.shape[0]:
            raise ValueError(f"Image depth {img_arr.shape[0]} does not match prediction depth {pred_arr.shape[0]}")

        # Choose the slice with the largest predicted area (or middle slice if empty)
        slice_scores = pred_arr.reshape(pred_arr.shape[0], -1).sum(axis=1)
        if slice_scores.max() > 0:
            z = int(slice_scores.argmax())
        else:
            z = img_arr.shape[0] // 2

        img_slice = img_arr[z]
        pred_slice = (pred_arr[z] > 0).astype(float)

        plt.figure(figsize=(6, 6))
        plt.imshow(img_slice, cmap="gray")
        plt.imshow(pred_slice, alpha=0.4, cmap="Reds")
        plt.title(f"{case['id']} (slice {z}, dice={out['dice']:.4f})")
        plt.axis("off")
        jpeg_path = jpeg_dir / f"{case['id']}_overlay.jpg"
        plt.tight_layout()
        plt.savefig(jpeg_path, dpi=100, bbox_inches="tight")
        plt.close()

        metrics.append({"case_id": case["id"], "dice": out["dice"]})
        print(f"[final] {case['id']} dice={out['dice']:.4f}  -> saved {jpeg_path.name}")

    mean_dice = float(np.mean([m["dice"] for m in metrics])) if metrics else 0.0
    save_json({"metrics": metrics, "mean_dice": mean_dice}, Path(cfg.data.workdir) / "predictions" / f"metrics_fold{cfg.data.val_fold}.json")
    print(f"[final] mean dice = {mean_dice:.4f}")
    return str(pred_orig_dir)
