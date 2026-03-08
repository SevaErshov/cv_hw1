import cv2
import numpy as np


def calculate_transforms_trajectory(frames, w, h):
    transforms = []

    for i in range(len(frames)-1):

        prev_frame = frames[i]
        next_frame = frames[i+1]

        prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev, next, None,
            pyr_scale=0.5,
            levels=5,
            winsize=15,
            iterations=5,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        u = flow[..., 0]
        v = flow[..., 1]

        grid_x, grid_y = np.meshgrid(
            np.arange(w),
            np.arange(h)
        )

        pts_prev = np.stack(
            [grid_x.flatten(), grid_y.flatten()],
            axis=1
        )

        pts_next = np.stack(
            [
                (grid_x + u).flatten(),
                (grid_y + v).flatten()
            ],
            axis=1
        )

        M, _ = cv2.estimateAffine2D(
            pts_prev,
            pts_next,
            method=cv2.RANSAC
        )

        dx = M[0,2]
        dy = M[1,2]
        da = np.arctan2(M[1,0], M[0,0])

        transforms.append([dx, dy, da])

    transforms = np.array(transforms)
    trajectory = np.cumsum(transforms, axis=0)
    return transforms, trajectory
