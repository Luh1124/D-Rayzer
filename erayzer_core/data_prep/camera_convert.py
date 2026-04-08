"""DL3DV / Nerfstudio camera axes → OpenCV c2w (for E-RayZer gsplat stack)."""

from __future__ import annotations

import numpy as np


def nerf_gl_c2w_to_opencv_c2w(transform_matrix: np.ndarray) -> np.ndarray:
    """
    transform_matrix: (4,4) c2w in Nerfstudio / OpenGL camera axes (+Y up, +Z back).
    Returns (4,4) c2w with OpenCV camera axes (+Y down, +Z forward), same as
    c2w_gl @ diag(1,-1,-1,1).
    """
    c2w_gl = np.asarray(transform_matrix, dtype=np.float64)
    return c2w_gl @ np.diag([1.0, -1.0, -1.0, 1.0])


def intrinsics_fullres_to_pixels(
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    full_w: float,
    full_h: float,
    pixel_w: int,
    pixel_h: int,
) -> tuple[float, float, float, float]:
    """Scale COLMAP-style intrinsics from transforms.json full-res to actual image size."""
    sx = pixel_w / full_w
    sy = pixel_h / full_h
    return fl_x * sx, fl_y * sy, cx * sx, cy * sy


def intrinsics_after_resize(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    old_w: int,
    old_h: int,
    new_w: int,
    new_h: int,
) -> tuple[float, float, float, float]:
    """Pinhole intrinsics when resizing image from (old_w, old_h) to (new_w, new_h)."""
    sx = new_w / old_w
    sy = new_h / old_h
    return fx * sx, fy * sy, cx * sx, cy * sy


def intrinsics_after_center_square_crop(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    start_w: int,
    start_h: int,
) -> tuple[float, float, float, float]:
    """After crop: subtract top-left of crop window from principal point."""
    return fx, fy, cx - start_w, cy - start_h
