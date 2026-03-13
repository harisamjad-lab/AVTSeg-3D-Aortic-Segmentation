from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from utils import crop_patch, maybe_shift_intensity, normalize_ct, save_json


def paired_cases(data_dir: str) -> List[Dict[str, str]]:
    """
    Expected structure:

    data/
      img1/
        img1.nrrd
        img1.seg.nrrd
      img2/
        img2.nrrd
        img2.seg.nrrd
      ...

    Each case folder must contain exactly:
    - one image file: *.nrrd
    - one mask file: *.seg.nrrd
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    cases = []

    # each immediate subfolder is treated as one subject/case
    case_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    for case_dir in case_dirs:
        all_nrrd = sorted(case_dir.glob("*.nrrd"))
        if not all_nrrd:
            continue

        image_files = [p for p in all_nrrd if not p.name.endswith(".seg.nrrd")]
        label_files = [p for p in all_nrrd if p.name.endswith(".seg.nrrd")]

        if len(image_files) != 1 or len(label_files) != 1:
            raise RuntimeError(
                f"Expected exactly 1 image (*.nrrd) and 1 label (*.seg.nrrd) in {case_dir}, "
                f"found {len(image_files)} image(s) and {len(label_files)} label(s)."
            )

        image_path = image_files[0]
        label_path = label_files[0]

        # optional consistency check:
        # img1.nrrd should match img1.seg.nrrd
        image_stem = image_path.name[:-5]   # remove .nrrd
        label_stem = label_path.name[:-9]   # remove .seg.nrrd
        if image_stem != label_stem:
            raise RuntimeError(
                f"Image/mask filename mismatch in {case_dir}: "
                f"{image_path.name} vs {label_path.name}"
            )

        cases.append(
            {
                "id": case_dir.name,
                "image": str(image_path),
                "label": str(label_path),
            }
        )

    if not cases:
        raise RuntimeError(f"No paired .nrrd / .seg.nrrd cases found under {root}")

    return cases


def make_kfold_splits(data_dir: str, out_json: str, k: int = 5, seed: int = 42):
    cases = [c["id"] for c in paired_cases(data_dir)]
    rng = random.Random(seed)
    rng.shuffle(cases)

    if len(cases) < k:
        k = len(cases)

    folds = [[] for _ in range(k)]
    for i, c in enumerate(cases):
        folds[i % k].append(c)

    splits = {}
    for i in range(k):
        val = folds[i]
        train = [c for j, fold_cases in enumerate(folds) if j != i for c in fold_cases]
        splits[f"fold_{i}"] = {"train": train, "val": val}

    save_json(splits, out_json)
    return splits


def load_volume_as_numpy(path: str):
    """
    Works for .nrrd, .mha, .nii.gz, etc. as long as SimpleITK supports it.
    Returns array in z,y,x order.
    """
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def write_like_reference(reference_path: str, arr_zyx: np.ndarray, out_path: str) -> None:
    ref = sitk.ReadImage(reference_path)
    out = sitk.GetImageFromArray(arr_zyx.astype(np.uint8))
    out.CopyInformation(ref)
    sitk.WriteImage(out, out_path)


def visualize_volume_slice(image_path: str, label_path: str | None = None, slice_index: int | None = None, cmap_img: str = "gray") -> None:
    """
    Quick visualization helper for a 3D volume (and optional label) stored in a file
    that SimpleITK can read (.nrrd, .mha, .nii.gz, etc.).
    Shows a single axial slice using matplotlib.
    """
    img, _ = load_volume_as_numpy(image_path)  # (z, y, x)

    if slice_index is None:
        slice_index = img.shape[0] // 2

    slice_index = int(np.clip(slice_index, 0, img.shape[0] - 1))
    img_slice = img[slice_index]

    if label_path is None:
        plt.figure(figsize=(5, 5))
        plt.imshow(img_slice, cmap=cmap_img)
        plt.title(f"Image slice {slice_index}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        return

    lab, _ = load_volume_as_numpy(label_path)
    if lab.shape[0] != img.shape[0]:
        raise ValueError(f"Label depth {lab.shape[0]} does not match image depth {img.shape[0]}")

    lab_slice = (lab[slice_index] > 0).astype(float)

    plt.figure(figsize=(10, 5))
    # left: image only
    plt.subplot(1, 2, 1)
    plt.imshow(img_slice, cmap=cmap_img)
    plt.title(f"Image slice {slice_index}")
    plt.axis("off")

    # right: image + label overlay
    plt.subplot(1, 2, 2)
    plt.imshow(img_slice, cmap=cmap_img)
    plt.imshow(lab_slice, alpha=0.4, cmap="Reds")
    plt.title(f"Image + label slice {slice_index}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def resample_sitk_image(image: sitk.Image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = np.array(image.GetSpacing(), dtype=np.float64)
    original_size = np.array(image.GetSize(), dtype=np.int32)
    out_spacing = np.array(out_spacing, dtype=np.float64)
    out_size = np.round(original_size * (original_spacing / out_spacing)).astype(np.int32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampler.SetOutputSpacing(tuple(out_spacing.tolist()))
    resampler.SetSize([int(x) for x in out_size.tolist()])
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def preprocess_case(case: Dict[str, str], preproc_dir: str, spacing, intensity_shift_threshold: float):
    case_dir = Path(preproc_dir) / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    image_sitk = sitk.ReadImage(case["image"])
    label_sitk = sitk.ReadImage(case["label"])

    image_np = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    image_np = maybe_shift_intensity(image_np, threshold=intensity_shift_threshold)

    shifted = sitk.GetImageFromArray(image_np)
    shifted.CopyInformation(image_sitk)

    image_res = resample_sitk_image(shifted, out_spacing=spacing, is_label=False)
    label_res = resample_sitk_image(label_sitk, out_spacing=spacing, is_label=True)

    pre_image = case_dir / "image_resampled.nrrd"
    pre_label = case_dir / "label_resampled.nrrd"
    sitk.WriteImage(image_res, str(pre_image), useCompression=True)
    sitk.WriteImage(label_res, str(pre_label), useCompression=True)

    return {
        "id": case["id"],
        "orig_image": case["image"],
        "orig_label": case["label"],
        "pre_image": str(pre_image),
        "pre_label": str(pre_label),
    }


class Stage1Dataset(Dataset):
    def __init__(self, cases, patch_size, intensity_min, intensity_max, fg_prob=0.7, training=True):
        self.cases = cases
        self.patch_size = tuple(patch_size)
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.fg_prob = fg_prob
        self.training = training

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        img, _ = load_volume_as_numpy(case["pre_image"])
        lab, _ = load_volume_as_numpy(case["pre_label"])

        img = normalize_ct(img, self.intensity_min, self.intensity_max)
        lab = (lab > 0).astype(np.uint8)

        if self.training:
            fg = np.argwhere(lab > 0)
            if len(fg) > 0 and random.random() < self.fg_prob:
                center = fg[random.randint(0, len(fg) - 1)]
            else:
                center = np.array([random.randint(0, s - 1) for s in img.shape])

            img = crop_patch(img, center, self.patch_size)
            lab = crop_patch(lab, center, self.patch_size)

            for axis in (0, 1, 2):
                if random.random() < 0.5:
                    img = np.flip(img, axis=axis).copy()
                    lab = np.flip(lab, axis=axis).copy()

        return {
            "image": torch.from_numpy(img[None]).float(),
            "label": torch.from_numpy(lab[None].astype(np.float32)).float(),
            "case_id": case["id"],
        }


class Stage2PatchDataset(Dataset):
    def __init__(self, records, patch_size, intensity_min, intensity_max, training=True):
        self.records = records
        self.patch_size = tuple(patch_size)
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img, _ = load_volume_as_numpy(rec["image"])
        lab, _ = load_volume_as_numpy(rec["label"])
        center = np.array(rec["center"], dtype=np.int32)

        img = crop_patch(img, center, self.patch_size)
        lab = crop_patch(lab, center, self.patch_size)

        img = normalize_ct(img, self.intensity_min, self.intensity_max)
        lab = (lab > 0).astype(np.uint8)

        if self.training:
            for axis in (0, 1, 2):
                if random.random() < 0.5:
                    img = np.flip(img, axis=axis).copy()
                    lab = np.flip(lab, axis=axis).copy()

        return {
            "image": torch.from_numpy(img[None]).float(),
            "label": torch.from_numpy(lab[None].astype(np.float32)).float(),
            "center": torch.tensor(center).int(),
            "case_id": rec["case_id"],
        }
