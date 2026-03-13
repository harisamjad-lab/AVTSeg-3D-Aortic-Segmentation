from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

# allow imports from avtseg_local/
THIS_FILE = Path(__file__).resolve()
PKG_ROOT = THIS_FILE.parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from dataio import load_volume_as_numpy
from utils import pseudo_centerline_from_mask, farthest_point_subsample


def resample_mask_to_reference(mask_path: str, reference_path: str) -> np.ndarray:
    ref = sitk.ReadImage(str(reference_path))
    mask = sitk.ReadImage(str(mask_path))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    aligned = resampler.Execute(mask)
    return sitk.GetArrayFromImage(aligned).astype(np.uint8)


def choose_best_slice(mask_zyx: np.ndarray, fallback_depth: int) -> int:
    scores = mask_zyx.reshape(mask_zyx.shape[0], -1).sum(axis=1)
    if scores.max() > 0:
        return int(scores.argmax())
    return fallback_depth // 2


def save_step1_input(cta: np.ndarray, z: int, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(cta[z], -200, 800), cmap="gray")
    plt.title(f"Step 1 — Input CTA slice (z={z})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_step2_coarse(cta: np.ndarray, coarse: np.ndarray, z: int, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(cta[z], -200, 800), cmap="gray")
    plt.imshow(np.ma.masked_where(coarse[z] == 0, coarse[z]), cmap="Reds", alpha=0.35)
    plt.title(f"Step 2 — Stage-1 coarse segmentation (z={z})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_step3_centerline(
    cta: np.ndarray,
    coarse: np.ndarray,
    centers: np.ndarray,
    out_path: Path,
) -> None:
    center_overlay = np.zeros_like(coarse, dtype=np.uint8)
    for c in centers:
        zz, yy, xx = map(int, c)
        if 0 <= zz < center_overlay.shape[0] and 0 <= yy < center_overlay.shape[1] and 0 <= xx < center_overlay.shape[2]:
            center_overlay[zz, yy, xx] = 1

    center_slices = np.where(center_overlay.sum(axis=(1, 2)) > 0)[0]
    z = int(center_slices[len(center_slices) // 2]) if len(center_slices) > 0 else choose_best_slice(coarse, cta.shape[0])

    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(cta[z], -200, 800), cmap="gray")
    plt.imshow(np.ma.masked_where(coarse[z] == 0, coarse[z]), cmap="Reds", alpha=0.12)
    ys, xs = np.where(center_overlay[z] > 0)
    plt.scatter(xs, ys, c="yellow", s=18)
    plt.title(f"Step 3 — Pseudo centerline points (z={z})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_step4_patch(
    cta: np.ndarray,
    coarse: np.ndarray,
    centers: np.ndarray,
    patch_size: tuple[int, int, int],
    out_path: Path,
) -> None:
    if len(centers) == 0:
        raise RuntimeError("No centerline points found. Cannot draw patch example.")

    center = centers[len(centers) // 2]
    cz, cy, cx = map(int, center)

    half_y = patch_size[1] // 2
    half_x = patch_size[2] // 2

    x1 = max(cx - half_x, 0)
    x2 = min(cx + half_x, cta.shape[2] - 1)
    y1 = max(cy - half_y, 0)
    y2 = min(cy + half_y, cta.shape[1] - 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(cta[cz], -200, 800), cmap="gray")
    plt.imshow(np.ma.masked_where(coarse[cz] == 0, coarse[cz]), cmap="Reds", alpha=0.12)
    plt.scatter([cx], [cy], c="yellow", s=28)
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color="cyan", linewidth=2)
    plt.title(f"Step 4 — Patch sampling example (z={cz})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_step5_final(cta: np.ndarray, pred: np.ndarray, z: int, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(cta[z], -200, 800), cmap="gray")
    plt.imshow(np.ma.masked_where(pred[z] == 0, pred[z]), cmap="Reds", alpha=0.4)
    plt.title(f"Step 5 — Final vessel segmentation (z={z})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Generate pipeline step visualizations for one case.")
    p.add_argument("--cta", required=True, help="Original CTA path, e.g. data/img43/img43.nrrd")
    p.add_argument("--coarse", required=True, help="Stage-1 coarse mask path")
    p.add_argument("--pred", required=True, help="Final prediction in original geometry")
    p.add_argument("--out_dir", required=True, help="Output folder, e.g. examples/case_img43")
    p.add_argument("--center_spacing", type=int, default=12)
    p.add_argument("--max_centers", type=int, default=256)
    p.add_argument("--patch_z", type=int, default=96)
    p.add_argument("--patch_y", type=int, default=96)
    p.add_argument("--patch_x", type=int, default=96)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cta, _ = load_volume_as_numpy(args.cta)
    coarse = resample_mask_to_reference(args.coarse, args.cta)
    pred = resample_mask_to_reference(args.pred, args.cta)

    centers = pseudo_centerline_from_mask(coarse)
    centers = farthest_point_subsample(
        centers,
        target_spacing=args.center_spacing,
        max_points=args.max_centers,
    )

    z_coarse = choose_best_slice(coarse, cta.shape[0])
    z_pred = choose_best_slice(pred, cta.shape[0])

    save_step1_input(cta, z_coarse, out_dir / "step1_input_cta_slice.png")
    save_step2_coarse(cta, coarse, z_coarse, out_dir / "step2_stage1_coarse_segmentation.png")
    save_step3_centerline(cta, coarse, centers, out_dir / "step3_centerline_points.png")
    save_step4_patch(
        cta,
        coarse,
        centers,
        patch_size=(args.patch_z, args.patch_y, args.patch_x),
        out_path=out_dir / "step4_patch_sampling.png",
    )
    save_step5_final(cta, pred, z_pred, out_dir / "step5_final_segmentation_overlay.png")

    print(f"Saved step visualizations to: {out_dir}")


if __name__ == "__main__":
    main()