import argparse
from pathlib import Path
from ultralytics import YOLO


def train_model(
    data_cfg_path: str,
    epochs: int = 40,
    imgsz: int = 512,
    batch: int = 16,
) -> None:
    """Train YOLOv8n detector with custom outdoor augmentation parameters."""
    model = YOLO("yolov8n.pt")

    # Train model and access results dictionary/metrics for Flake8 compliance
    results = model.train(
        data=data_cfg_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="yonder_mallet_bottle_exp",
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        fliplr=0.5,
    )

    # Access results object to display final validation metrics
    if hasattr(results, "results_dict"):
        map50 = results.results_dict.get("metrics/mAP50(B)", 0.0)
        print(f"Training completed. Final Validation mAP50: {map50:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO model for Yonder Dynamics take-home."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Image size for training",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    args = parser.parse_args()

    # Determine repository root relative to src/train.py location
    project_root = Path(__file__).resolve().parent.parent
    data_yaml = (
        project_root / "Sampled-YD-Object-Detection-2" / "data.yaml"
    )

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found at {data_yaml}. "
            "Please run download_data.py first."
        )

    train_model(
        data_cfg_path=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
