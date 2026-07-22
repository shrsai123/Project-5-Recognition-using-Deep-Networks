# Shreyas Raman
# Project 5: crop and classify handwritten digits from one sheet photo

# import statements
import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mnist_cnn import MyNetwork


# useful functions with a comment for each function
def parse_args(argv):
    """Parses command line options for classifying a digit sheet image."""
    parser = argparse.ArgumentParser(description="Auto-crop and classify handwritten digits from one photo.")
    parser.add_argument("image_path", type=Path, help="photo containing handwritten digits")
    parser.add_argument("--model-path", type=Path, default=Path("mnist_cnn.pt"), help="saved MNIST model path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/custom_sheet"), help="output directory")
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="ccw", help="image rotation")
    parser.add_argument("--labels", default="", help="comma-separated correct labels in detected reading order")
    parser.add_argument("--min-area", type=float, default=800.0, help="minimum contour area for digit boxes")
    parser.add_argument("--padding", type=float, default=0.35, help="fractional padding around each digit crop")
    return parser.parse_args(argv[1:])


def get_device():
    """Selects the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_network(model_path, device):
    """Loads the trained MNIST CNN and sets it to evaluation mode."""
    model = MyNetwork().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_grayscale_image(image_path, rotate):
    """Loads the image as grayscale and applies the requested rotation."""
    image = Image.open(image_path).convert("L")
    grayscale = np.array(image)
    if rotate == "cw":
        grayscale = cv2.rotate(grayscale, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == "ccw":
        grayscale = cv2.rotate(grayscale, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate == "180":
        grayscale = cv2.rotate(grayscale, cv2.ROTATE_180)
    return grayscale


def detect_digit_boxes(grayscale, min_area):
    """Finds likely digit bounding boxes using inverted Otsu thresholding."""
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _threshold, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = grayscale.shape
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (
            area >= min_area
            and w > 20
            and h > 20
            and x > 5
            and y > 5
            and x + w < width - 5
            and y + h < height - 5
            and w < width * 0.25
            and h < height * 0.25
        ):
            boxes.append((x, y, w, h))

    return sort_boxes_by_rows(boxes), mask


def sort_boxes_by_rows(boxes):
    """Sorts digit boxes by row, then left to right within each row."""
    if not boxes:
        return []

    median_height = float(np.median([box[3] for box in boxes]))
    row_threshold = max(80.0, median_height * 1.2)
    sorted_by_y = sorted(boxes, key=lambda box: box[1] + box[3] / 2)
    rows = []

    for box in sorted_by_y:
        center_y = box[1] + box[3] / 2
        if not rows:
            rows.append([box])
            continue

        row_centers = [other[1] + other[3] / 2 for other in rows[-1]]
        if abs(center_y - float(np.mean(row_centers))) <= row_threshold:
            rows[-1].append(box)
        else:
            rows.append([box])

    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda box: box[0]))
    return ordered


def crop_to_mnist(mask, box, padding):
    """Converts one digit box to a centered 28x28 MNIST-like white-on-black image."""
    x, y, w, h = box
    pad = int(max(w, h) * padding)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(mask.shape[1], x + w + pad)
    y1 = min(mask.shape[0], y + h + pad)
    crop = mask[y0:y1, x0:x1]

    side = max(crop.shape)
    square = np.zeros((side, side), dtype=np.uint8)
    top = (side - crop.shape[0]) // 2
    left = (side - crop.shape[1]) // 2
    square[top : top + crop.shape[0], left : left + crop.shape[1]] = crop

    resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = resized
    return canvas


def tensor_from_mnist_image(image):
    """Converts a 28x28 uint8 image to a normalized MNIST tensor."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return transform(Image.fromarray(image))


def classify_crops(model, device, crops):
    """Runs all cropped digit images through the trained network."""
    rows = []
    with torch.no_grad():
        for index, crop in enumerate(crops):
            tensor = tensor_from_mnist_image(crop).unsqueeze(0).to(device)
            output = model(tensor).squeeze(0).cpu()
            prediction = output.argmax().item()
            rows.append({"index": index, "crop": crop, "outputs": output.tolist(), "prediction": prediction})
    return rows


def parse_labels(label_text):
    """Parses optional comma-separated labels."""
    if not label_text.strip():
        return []
    return [int(piece.strip()) for piece in label_text.split(",") if piece.strip()]


def save_debug_boxes(grayscale, boxes, output_path):
    """Saves the source image with detected boxes and crop indices."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    color = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    for index, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(color, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.putText(color, str(index), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imwrite(str(output_path), color)


def save_crops(crops, output_dir):
    """Saves individual 28x28 crops for visual inspection."""
    crop_dir = output_dir / "crops_28x28"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for index, crop in enumerate(crops):
        cv2.imwrite(str(crop_dir / f"digit_{index:02d}.png"), crop)


def format_outputs(values):
    """Formats the 10 network output values with two decimal places."""
    return " ".join(f"{value:6.2f}" for value in values)


def save_prediction_tables(rows, labels, output_dir):
    """Saves prediction results as text and CSV tables."""
    text_lines = ["Index | Outputs for digits 0-9                                           | Pred | Label | Correct"]
    text_lines.append("-" * len(text_lines[0]))
    csv_path = output_dir / "sheet_digit_predictions.csv"
    text_path = output_dir / "sheet_digit_predictions.txt"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["index", "prediction", "label", "correct"] + [f"output_{digit}" for digit in range(10)])

        for row in rows:
            label = labels[row["index"]] if row["index"] < len(labels) else ""
            correct = row["prediction"] == label if label != "" else ""
            output_text = format_outputs(row["outputs"])
            text_lines.append(f"{row['index']:5d} | {output_text} | {row['prediction']:4d} | {label!s:5s} | {correct}")
            writer.writerow([row["index"], row["prediction"], label, correct] + [f"{value:.6f}" for value in row["outputs"]])

    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print("\n".join(text_lines))


def plot_predictions(rows, labels, output_path):
    """Saves a grid of cropped digits with predictions and optional labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = len(rows)
    columns = min(5, count)
    grid_rows = math.ceil(count / columns)
    fig, axes = plt.subplots(grid_rows, columns, figsize=(1.9 * columns, 2.6 * grid_rows))
    axes = [axes] if count == 1 else list(np.array(axes).reshape(-1))

    for axis in axes:
        axis.axis("off")

    for row in rows:
        index = row["index"]
        label = labels[index] if index < len(labels) else None
        title = f"Pred: {row['prediction']}"
        if label is not None:
            title += f"\nLabel: {label}"
        axes[index].imshow(row["crop"], cmap="gray")
        axes[index].set_title(title)

    fig.subplots_adjust(hspace=0.55, wspace=0.15)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# main function
def main(argv):
    """Auto-crops a handwritten digit sheet and classifies the detected digits."""
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = load_network(args.model_path, device)
    grayscale = load_grayscale_image(args.image_path, args.rotate)
    boxes, mask = detect_digit_boxes(grayscale, args.min_area)
    labels = parse_labels(args.labels)

    crops = [crop_to_mnist(mask, box, args.padding) for box in boxes]
    rows = classify_crops(model, device, crops)

    save_debug_boxes(grayscale, boxes, output_dir / "sheet_detected_boxes.jpg")
    save_crops(crops, output_dir)
    save_prediction_tables(rows, labels, output_dir)
    plot_predictions(rows, labels, output_dir / "sheet_digit_predictions.png")

    if labels and len(labels) != len(rows):
        print(f"Warning: received {len(labels)} labels but detected {len(rows)} digit crops.")

    print(f"Detected {len(rows)} digit crops.")
    print(f"Saved debug boxes to {output_dir / 'sheet_detected_boxes.jpg'}")
    print(f"Saved 28x28 crops to {output_dir / 'crops_28x28'}")
    print(f"Saved prediction plot to {output_dir / 'sheet_digit_predictions.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
