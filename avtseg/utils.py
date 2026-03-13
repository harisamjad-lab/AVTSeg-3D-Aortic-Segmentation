from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.measure import label as cc_label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_mkdir(path: os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_shift_intensity(arr: np.ndarray, threshold: float = 500.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    if float(np.median(arr)) > threshold:
        arr = arr - 1024.0
    return arr


def normalize_ct(arr: np.ndarray, intensity_min: float, intensity_max: float) -> np.ndarray:
    arr = np.clip(arr, intensity_min, intensity_max)
    arr = (arr - intensity_min) / (intensity_max - intensity_min)
    return arr.astype(np.float32)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labeled = cc_label(mask > 0, connectivity=1)
    if labeled.max() == 0:
        return mask.astype(np.uint8)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return (labeled == keep).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_size: int = 200) -> np.ndarray:
    labeled = cc_label(mask > 0, connectivity=1)
    if labeled.max() == 0:
        return mask.astype(np.uint8)
    counts = np.bincount(labeled.ravel())
    keep_ids = np.where(counts >= min_size)[0]
    keep_ids = keep_ids[keep_ids != 0]
    return np.isin(labeled, keep_ids).astype(np.uint8)


def dice_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float((2.0 * inter + eps) / (pred.sum() + gt.sum() + eps))


def pseudo_centerline_from_mask(mask_zyx: np.ndarray) -> np.ndarray:
    mask = (mask_zyx > 0).astype(np.uint8)
    coords = []
    for z in range(mask.shape[0]):
        ys, xs = np.where(mask[z] > 0)
        if len(ys) > 0:
            coords.append((z, int(np.mean(ys)), int(np.mean(xs))))
    for y in range(mask.shape[1]):
        zs, xs = np.where(mask[:, y, :] > 0)
        if len(zs) > 0:
            coords.append((int(np.mean(zs)), y, int(np.mean(xs))))
    for x in range(mask.shape[2]):
        zs, ys = np.where(mask[:, :, x] > 0)
        if len(zs) > 0:
            coords.append((int(np.mean(zs)), int(np.mean(ys)), x))
    if not coords:
        return np.zeros((0, 3), dtype=np.int32)
    return np.unique(np.array(coords, dtype=np.int32), axis=0)


def farthest_point_subsample(points: np.ndarray, target_spacing: float, max_points: int | None = None) -> np.ndarray:
    if len(points) == 0:
        return points
    selected = [points[0]]
    remaining = points[1:].copy()
    while len(remaining) > 0:
        dists = np.sqrt(((remaining[:, None, :] - np.array(selected)[None, :, :]) ** 2).sum(axis=2))
        min_d = dists.min(axis=1)
        idx = int(np.argmax(min_d))
        if float(min_d[idx]) < target_spacing:
            break
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
        if max_points is not None and len(selected) >= max_points:
            break
    return np.array(selected, dtype=np.int32)


def compute_valid_center(center: np.ndarray, shape_zyx: Sequence[int], patch_size_zyx: Sequence[int]) -> np.ndarray:
    c = center.astype(np.int32).copy()
    half = np.array(patch_size_zyx) // 2
    low = half
    high = np.array(shape_zyx) - (np.array(patch_size_zyx) - half)
    c = np.minimum(np.maximum(c, low), high)
    return c


def crop_patch(arr_zyx: np.ndarray, center_zyx: Sequence[int], patch_size_zyx: Sequence[int]) -> np.ndarray:
    c = compute_valid_center(np.array(center_zyx), arr_zyx.shape, patch_size_zyx)
    half = np.array(patch_size_zyx) // 2
    start = c - half
    end = start + np.array(patch_size_zyx)
    return arr_zyx[start[0]:end[0], start[1]:end[1], start[2]:end[2]]


def paste_patch_additive(accum: np.ndarray, weight: np.ndarray, patch: np.ndarray, center_zyx: Sequence[int]) -> None:
    patch_size = np.array(patch.shape)
    c = compute_valid_center(np.array(center_zyx), accum.shape, patch_size)
    half = patch_size // 2
    start = c - half
    end = start + patch_size
    accum[start[0]:end[0], start[1]:end[1], start[2]:end[2]] += patch
    weight[start[0]:end[0], start[1]:end[1], start[2]:end[2]] += 1.0


def binary_postprocess(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    pred = remove_small_components(mask.astype(np.uint8), min_size=min_component_size)
    pred = largest_connected_component(pred)
    pred = ndi.binary_closing(pred, iterations=1).astype(np.uint8)
    return pred
