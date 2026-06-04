"""Variation-aware proxy learning with a PIDNet-M segmentation backbone."""

from .model import VAPLPIDNetM, build_vapl_pidnet_m

__all__ = ["VAPLPIDNetM", "build_vapl_pidnet_m"]
