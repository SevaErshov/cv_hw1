import cv2
import numpy as np


def warp_frame(frame, dx, dy, da, w, h):
    cos = np.cos(da)
    sin = np.sin(da)

    M = np.array([
        [cos, -sin, dx],
        [sin,  cos, dy]
    ], dtype=np.float32)

    prev_frame = frame

    flow = np.zeros((h, w, 2), np.float32)

    grid_x, grid_y = np.meshgrid(
        np.arange(w),
        np.arange(h)
    )

    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    warped = cv2.remap(
        prev_frame,
        map_x,
        map_y,
        cv2.INTER_LINEAR
    )

    warped = cv2.warpAffine(warped, M, (w, h))
    return warped

def crop_frame(frame, border, w, h):
    cropped = frame[
        border:h-border,
        border:w-border
    ]

    cropped = cv2.resize(cropped, (w, h))
    return cropped
