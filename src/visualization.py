import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(trajectory, smoothed_trajectory, save_dir="visualizations"):
    labels = ["x", "y", "angle"]

    for i in range(3):

        plt.figure(figsize=(10,5))

        plt.plot(trajectory[:, i], label="original")
        plt.plot(smoothed_trajectory[:, i], label="smoothed")

        plt.title(f"Camera trajectory ({labels[i]})")
        plt.xlabel("frame")
        plt.ylabel(labels[i])

        plt.legend()

        path = f"{save_dir}/trajectory_{labels[i]}.png"
        plt.savefig(path)

        plt.close()
