import torch


@torch.no_grad()
def build_stepback_c2ws(frame_c2ws: torch.Tensor, step_back_distance: float) -> torch.Tensor:
    """
    frame_c2ws: (..., 4, 4) camera-to-world (OpenCV-style) transforms
    step_back_distance: scalar distance to move along each camera's local -Z axis
    returns: stepback_c2ws with same shape as frame_c2ws
    """
    # Extract rotation (R) and translation (t) from c2w
    R = frame_c2ws[..., :3, :3]                  # (..., 3, 3)
    t = frame_c2ws[..., :3,  3]                  # (..., 3)

    # Local camera +Z is the 3rd column of R in world coords; step-back is along -Z
    z_world = R[..., :, 2]                       # (..., 3)
    t_new = t - step_back_distance * z_world     # move camera center backward

    c2w_step = frame_c2ws.clone()
    c2w_step[..., :3, 3] = t_new
    return c2w_step