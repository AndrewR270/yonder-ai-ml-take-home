import argparse
import os
from pathlib import Path
from ultralytics import YOLO

# Class mapping expected by Yonder Dynamics
# Index 0: bottle, Index 1: mallet
CLASS_NAMES = ["bottle", "mallet"]
CONF_THRESHOLD = 0.25  # Chosen confidence threshold for detection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run object inference and output YOLO-formatted labels."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input directory containing images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inferences/last",
        help="Path to output directory to save prediction .txt files.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to trained model weights (.pt).",
    )
    return parser.parse_args()


def run_inference(input_dir: str, output_dir: str, weights_path: str):
    # Ensure input path exists
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading weights from: {weights_path}")
    model = YOLO(weights_path)

    # Collect valid image paths
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in input_path.iterdir() if f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images in '{input_path}'. Starting...")

    for img_file in image_files:
        # Run model inference on a single image
        results = model.predict(
            source=str(img_file),
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        result = results[0]  # First result object for this image
        txt_out_path = output_path / f"{img_file.stem}.txt"

        lines = []
        if result.boxes is not None and len(result.boxes) > 0:
            # Extract normalized xywh coordinates and confidence/class metrics
            # boxes.xywhn: [x_center, y_center, width, height] normalized
            boxes_norm = result.boxes.xywhn.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes_norm, confs, cls_ids):
                x_center, y_center, width, height = box
                # Output format: <class_id> <x_center> <y_center> <width> <height> <confidence>  # noqa: E501
                line = f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}"  # noqa: E501
                lines.append(line)

        # Write predictions to corresponding .txt file
        with open(txt_out_path, "w") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

    print(f"Inference complete! Results saved to '{output_path}'.")


if __name__ == "__main__":
    args = parse_args()
    # Adjust weights path if your saved weights folder uses a specific run name
    weights = (
        args.weights if os.path.exists(args.weights) else "weights/best.pt"
    )  # noqa: E501
    run_inference(args.input, args.output, weights)
