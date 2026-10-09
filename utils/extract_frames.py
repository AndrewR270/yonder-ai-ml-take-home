import argparse
import cv2
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, stride: int = 10):
    video = cv2.VideoCapture(video_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % stride == 0:
            out_file = output_path / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(out_file), frame)
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"Extracted {saved_count} frames into '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts video frames.")
    parser.add_argument("--video", type=str, required=True, help="Video path.")
    parser.add_argument(
        "--output",
        type=str,
        default="frames/last",
        help="Output folder.",
    )
    parser.add_argument(
        "--stride", type=int, default=10, help="Extract every Nth frame."
    )
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.stride)
