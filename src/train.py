"""
Handles Yonder Dynamics rover target detection training.
It configures parameters and ranges for augmentations
on Ultralytics YOLOv8, runs training epochs,
and outputs evaluation metrics and model weight checkpoints.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_model(
    data_cfg_path: str,
    epochs: int = 40,
    imgsz: int = 512,
    batch: int = 16,
) -> None:
    """
    Executes model training and validation using transfer learning.
    Args:
        data_cfg_path: Path to data.yaml.
        epochs: Total passes across the full dataset (dflt: 40).
        imgsz: Square image resolution (px) to resize input images (dflt: 512).
        batch: Number of images processed per gradient update step (dflt: 16).
    """

    # Loads YOLOv8 Nano. We use feature extractors (edges, textures, shapes)
    # and replace the prediction head to learn target classes.
    model = YOLO("yolov8n.pt")

    # Hyperparameters tune spatial and color variations for outdoor rovers.
    # These mimic field conditions for lighting and orientation changes.
    results = model.train(
        data=data_cfg_path,    # Path to data.yaml
        epochs=epochs,          # Total full iterations
        imgsz=imgsz,            # Max dimensions = imgsz x imgsz
        batch=batch,            # Number of images processed in parallel
        name="yonder_mallet_bottle_exp",  # Output directory under runs/detect/
        hsv_h=0.015,  # Shifts hue channels randomly within [-0.015, +0.015]
        hsv_s=0.7,  # Adjusts color intensity randomly within [-70%, +70%].
        hsv_v=0.4,  # Adjusts pixel brightness randomly within [-40%, +40%].
        degrees=15.0,  # Rotates images & bounding box between [-15.0°, +15.0°]
        fliplr=0.5,  # 50% chance of mirroring the image horizontally.
    )

    # Post-training: Access results_dict to retrieve final performance metrics.
    # (mAP@50 measures bounding box IoU accuracy at a 0.5 overlap threshold).
    if hasattr(results, "results_dict"):
        map50 = results.results_dict.get("metrics/mAP50(B)", 0.0)
        print(f"Training completed. Final Validation mAP50: {map50:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO training model for mallet and bottle detection."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs (passes over dataset)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Target image dimension in pixels (square context)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Mini-batch size per gradient update step",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_yaml = (
        project_root / "Sampled-YD-Object-Detection-2" / "data.yaml"
    )

    # Guard clause: ensure data download script has executed prior to training
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found at {data_yaml}. "
            "Please run get_dataset.py first."
        )

    # Trigger model training pipeline
    train_model(
        data_cfg_path=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
