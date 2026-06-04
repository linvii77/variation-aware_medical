"""Variation-aware proxy learning with segmentation backbones."""

from .model import VAPLPIDNetM, VAPLSCDL3D, build_vapl_pidnet_m, build_vapl_scdl_3d

__all__ = [
    "VAPLPIDNetM",
    "VAPLSCDL3D",
    "build_vapl_pidnet_m",
    "build_vapl_scdl_3d",
]
