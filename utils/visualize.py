import argparse
from pathlib import Path
import cv2


def draw_yolo_boxes(img_path: str, label_path: str, class_names: dict = None):
    """Draws bounding boxes and confidence labels on an image using
    YOLO format predictions."""

    # Default class mapping matching data.yaml (0: bottle, 1: mallet)
    if class_names is None:
        class_names = {0: "bottle", 1: "mallet"}

    # Load image from disk using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at '{img_path}'.")
        return

    # Extract pixel dimensions needed to de-normalize 0-1 coordinates
    h, w, _ = img.shape
    txt_path = Path(label_path)

    # Return early if no detection text file exists for this frame
    if not txt_path.exists():
        print(f"No detection label file found at '{label_path}'.")
        return

    # Parse detection annotations line by line
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # Ensure line contains class_id, x, y, w, h, and confidence
            if len(parts) < 6:
                continue

            cls_id = int(parts[0])
            x_c, y_c, box_w, box_h = map(float, parts[1:5])
            conf = float(parts[5])

            # De-normalize 0-1 relative coordinates back into pixel values:
            # x_min (x1) = (x_center - width/2) * image_width
            # y_min (y1) = (y_center - height/2) * image_height
            # x_max (x2) = (x_center + width/2) * image_width
            # y_max (y2) = (y_center + height/2) * image_height
            x1 = int((x_c - box_w / 2) * w)
            y1 = int((y_c - box_h / 2) * h)
            x2 = int((x_c + box_w / 2) * w)
            y2 = int((y_c + box_h / 2) * h)

            # Build box overlay text and select distinct box colors
            # (Green = Bottle, Blue = Mallet)
            label = f"{class_names.get(cls_id, cls_id)}: {conf:.2f}"
            color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)

            # Draw 2px bounding box around target
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Overlay class name and confidence score above top-left corner
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    # Render annotated image in an interactive pop-up window
    cv2.imshow("Detection Verification", img)
    cv2.waitKey(0)  # Wait indefinitely until any key is pressed
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Command-line interface setup
    parser = argparse.ArgumentParser(
        description="Visualize YOLO normalized bounding box detections."
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--box",
        required=True,
        help="Path to generated YOLO .txt label file.",
    )
    args = parser.parse_args()

    draw_yolo_boxes(args.image, args.box)
