import cv2
import hydra
from omegaconf import DictConfig, OmegaConf
from src.smoothing import smooth_trajectory
from src.transforms import calculate_transforms_trajectory
from src.warp_utils import warp_frame, crop_frame
from src.visualization import plot_trajectory, create_side_by_side_video


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    cap = cv2.VideoCapture(cfg.input)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    transforms, trajectory = calculate_transforms_trajectory(frames, w, h)

    smoothed_trajectory = smooth_trajectory(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    plot_trajectory(trajectory, smoothed_trajectory, save_dir=cfg.visualization_dir)

    out = cv2.VideoWriter(
        cfg.output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w,h)
    )

    for i in range(len(frames)-1):
        frame = frames[i]
        dx, dy, da = transforms_smooth[i]
        warped = warp_frame(frame, dx, dy, da, w, h)
        cropped = crop_frame(warped, cfg.border, w, h)

        out.write(cropped)

    out.release()

if __name__ == "__main__":
    main()
