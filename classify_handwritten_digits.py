# Shreyas Raman
# Project 5: run the saved MNIST network on custom handwritten digit images

# import statements
import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from mnist_cnn import MyNetwork


# useful functions with a comment for each function
def parse_args(argv):
    """Parses command line options for classifying custom digit images."""
    parser = argparse.ArgumentParser(description="Classify cropped handwritten digit images.")
    parser.add_argument("image_dir", type=Path, help="directory containing cropped digit images")
    parser.add_argument("--model-path", type=Path, default=Path("mnist_cnn.pt"), help="saved model path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="output directory")
    parser.add_argument("--threshold", type=int, default=20, help="foreground threshold after inversion")
    parser.add_argument(
        "--no-invert",
        action="store_false",
        dest="invert",
        help="do not invert intensities before classification",
    )
    parser.set_defaults(invert=True)
    return parser.parse_args(argv[1:])


def get_device():
    """Selects the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_network(model_path, device):
    """Reads saved model weights and puts the network in evaluation mode."""
    model = MyNetwork().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def find_image_files(image_dir):
    """Finds common image files in a directory in sorted order."""
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def preprocess_digit_image(image_path, invert, threshold):
    """Converts a custom digit image to MNIST-like white-on-black 28x28 input."""
    image = Image.open(image_path).convert("L")
    if invert:
        image = ImageOps.invert(image)

    mask = image.point(lambda pixel: 255 if pixel > threshold else 0)
    bbox = mask.getbbox()
    if bbox is not None:
        image = image.crop(bbox)

    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - image.width) // 2
    top = (28 - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def image_to_tensor(image):
    """Normalizes a PIL image with the same values used for MNIST training."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return transform(image)


def format_outputs(values):
    """Formats the 10 network output values with two decimal places."""
    return " ".join(f"{value:6.2f}" for value in values)


def classify_images(model, device, image_paths, invert, threshold):
    """Classifies each custom image and returns predictions plus processed images."""
    rows = []
    with torch.no_grad():
        for image_path in image_paths:
            processed = preprocess_digit_image(image_path, invert, threshold)
            tensor = image_to_tensor(processed).unsqueeze(0).to(device)
            output = model(tensor).squeeze(0).cpu()
            rows.append(
                {
                    "path": image_path,
                    "processed": processed,
                    "outputs": output.tolist(),
                    "prediction": output.argmax().item(),
                }
            )
    return rows


def print_prediction_table(rows):
    """Prints network outputs and predicted digit for each custom image."""
    header = "File | Network outputs for digits 0-9                                      | Pred"
    separator = "-" * len(header)
    print(header)
    print(separator)

    lines = [header, separator]
    for row in rows:
        output_text = format_outputs(row["outputs"])
        line = f"{row['path'].name} | {output_text} | {row['prediction']:4d}"
        print(line)
        lines.append(line)
    return lines


def save_prediction_table(lines, output_path):
    """Writes the custom image prediction table to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_custom_predictions(rows, output_path):
    """Saves a grid of preprocessed custom digit images with predictions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = len(rows)
    columns = min(5, count)
    rows_needed = math.ceil(count / columns)
    fig, axes = plt.subplots(rows_needed, columns, figsize=(1.8 * columns, 2.1 * rows_needed))

    if count == 1:
        axes = [axes]
    else:
        axes = list(getattr(axes, "flat", axes))

    for axis in axes:
        axis.axis("off")

    for index, row in enumerate(rows):
        axes[index].imshow(row["processed"], cmap="gray")
        axes[index].set_title(f"{row['path'].stem}: {row['prediction']}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# main function
def main(argv):
    """Loads custom digit images, classifies them, and saves report outputs."""
    args = parse_args(argv)
    device = get_device()
    image_paths = find_image_files(args.image_dir)

    if not image_paths:
        print(f"No image files found in {args.image_dir}")
        return 1

    model = load_network(args.model_path, device)
    rows = classify_images(model, device, image_paths, args.invert, args.threshold)
    table_lines = print_prediction_table(rows)

    save_prediction_table(table_lines, args.output_dir / "custom_digit_predictions.txt")
    plot_custom_predictions(rows, args.output_dir / "custom_digit_predictions.png")

    print(f"Saved custom prediction table to {args.output_dir / 'custom_digit_predictions.txt'}")
    print(f"Saved custom prediction plot to {args.output_dir / 'custom_digit_predictions.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
