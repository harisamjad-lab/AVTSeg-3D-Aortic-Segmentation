from __future__ import annotations

from monai.networks.nets import SegResNet


def build_model():
    return SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.0,
    )
