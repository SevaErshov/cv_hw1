import numpy as np


def moving_average(curve, radius):
    window = 2 * radius + 1
    f = np.ones(window)/window
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    return curve_smoothed[radius:-radius]

def smooth_trajectory(trajectory):
    smoothed = np.copy(trajectory)

    for i in range(trajectory.shape[1]):
        smoothed[:, i] = moving_average(trajectory[:, i], radius=15)

    return smoothed
