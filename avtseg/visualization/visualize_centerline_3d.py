from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import SimpleITK as sitk
import plotly.graph_objects as go
from skimage.measure import marching_cubes

THIS_FILE = Path(__file__).resolve()
PKG_ROOT = THIS_FILE.parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from utils import pseudo_centerline_from_mask, farthest_point_subsample


def resample_mask_to_reference(mask_path: str, reference_path: str):
    ref = sitk.ReadImage(str(reference_path))
    mask = sitk.ReadImage(str(mask_path))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    aligned = resampler.Execute(mask)
    return ref, sitk.GetArrayFromImage(aligned).astype(np.uint8)


def mask_to_mesh(mask_zyx: np.ndarray, spacing_xyz):
    if mask_zyx.sum() == 0:
        return None, None

    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    verts, faces, _, _ = marching_cubes(mask_zyx.astype(np.float32), level=0.5, spacing=spacing_zyx)
    verts_xyz = np.stack([verts[:, 2], verts[:, 1], verts[:, 0]], axis=1)
    return verts_xyz, faces


def main():
    p = argparse.ArgumentParser(description="Generate 3D coarse-mask + pseudo-centerline visualization.")
    p.add_argument("--cta", required=True)
    p.add_argument("--coarse", required=True)
    p.add_argument("--out_html", required=True)
    p.add_argument("--center_spacing", type=int, default=12)
    p.add_argument("--max_centers", type=int, default=256)
    args = p.parse_args()

    ref_img, coarse = resample_mask_to_reference(args.coarse, args.cta)

    centers = pseudo_centerline_from_mask(coarse)
    centers = farthest_point_subsample(
        centers,
        target_spacing=args.center_spacing,
        max_points=args.max_centers,
    )

    spacing_xyz = ref_img.GetSpacing()
    verts_xyz, faces = mask_to_mesh(coarse, spacing_xyz)
    if verts_xyz is None:
        raise RuntimeError("Coarse mask is empty.")

    centers_xyz = np.stack([
        centers[:, 2] * spacing_xyz[0],
        centers[:, 1] * spacing_xyz[1],
        centers[:, 0] * spacing_xyz[2],
    ], axis=1)
    centers_xyz = centers_xyz[np.argsort(centers_xyz[:, 2])]

    mesh = go.Mesh3d(
        x=verts_xyz[:, 0],
        y=verts_xyz[:, 1],
        z=verts_xyz[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="lightgray",
        opacity=0.20,
        name="Coarse vessel mask",
        flatshading=True,
    )

    center_pts = go.Scatter3d(
        x=centers_xyz[:, 0],
        y=centers_xyz[:, 1],
        z=centers_xyz[:, 2],
        mode="markers",
        marker=dict(size=3, color="red"),
        name="Pseudo-centerline points",
    )

    center_line = go.Scatter3d(
        x=centers_xyz[:, 0],
        y=centers_xyz[:, 1],
        z=centers_xyz[:, 2],
        mode="lines",
        line=dict(color="red", width=4),
        name="Pseudo-centerline curve",
    )

    fig = go.Figure(data=[mesh, center_pts, center_line])
    fig.update_layout(
        title="3D Pseudo-Centerline Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        width=1000,
        height=800,
        legend=dict(x=0.01, y=0.98),
    )

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    print(f"Saved interactive HTML to: {out_html}")


if __name__ == "__main__":
    main()