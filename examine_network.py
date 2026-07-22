# Shreyas Raman
# Project 5: examine the trained MNIST network and first-layer filters

# import statements
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from mnist_cnn import MyNetwork


# useful functions with a comment for each function
def parse_args(argv):
    """Parses command line options for examining the saved network."""
    parser = argparse.ArgumentParser(description="Analyze the first layer of the saved MNIST CNN.")
    parser.add_argument("--model-path", type=Path, default=Path("mnist_cnn.pt"), help="saved model path")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="output directory")
    parser.add_argument("--download", action="store_true", help="download MNIST if it is not present")
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


def load_first_training_example(data_dir, download):
    """Loads the first unshuffled MNIST training image."""
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    image, label = train_dataset[0]
    return image.squeeze(0).numpy(), label


def get_conv1_weights(model):
    """Returns the first convolution layer weights without gradient tracking."""
    with torch.no_grad():
        return model.conv1.weight.detach().cpu().clone()


def print_model_and_filters(model, weights):
    """Prints the model structure, weight shape, and all 10 conv1 filters."""
    lines = []

    def record(text=""):
        """Prints one line and stores it for the analysis log."""
        print(text)
        lines.append(str(text))

    record("Model structure:")
    record(model)
    record()
    record(f"conv1 weight tensor shape: {tuple(weights.shape)}")
    record()

    for index in range(weights.shape[0]):
        record(f"Filter {index} weights, shape {tuple(weights[index, 0].shape)}:")
        record(weights[index, 0].numpy())
        record()

    return lines


def save_text_log(lines, output_path):
    """Writes the printed model and filter values to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_conv1_filters(weights, output_path):
    """Saves a 3x4 grid visualization of the 10 conv1 filters."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))

    for index in range(weights.shape[0]):
        axis = fig.add_subplot(3, 4, index + 1)
        axis.imshow(weights[index, 0].numpy(), cmap="viridis")
        axis.set_title(f"Filter {index}")
        axis.set_xticks([])
        axis.set_yticks([])

    for empty_index in range(weights.shape[0] + 1, 13):
        axis = fig.add_subplot(3, 4, empty_index)
        axis.axis("off")

    fig.suptitle("First Layer Convolution Filters")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def apply_filters_to_image(weights, image):
    """Applies each conv1 filter to the first training image with OpenCV."""
    filtered_images = []
    with torch.no_grad():
        for index in range(weights.shape[0]):
            kernel = weights[index, 0].numpy()
            filtered = cv2.filter2D(image, ddepth=-1, kernel=kernel)
            filtered_images.append(filtered)
    return filtered_images


def plot_filtered_images(filtered_images, label, output_path):
    """Saves a 3x4 grid of the first training image after each conv1 filter."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))

    for index, filtered in enumerate(filtered_images):
        axis = fig.add_subplot(3, 4, index + 1)
        axis.imshow(filtered, cmap="gray")
        axis.set_title(f"Filter {index}")
        axis.set_xticks([])
        axis.set_yticks([])

    for empty_index in range(len(filtered_images) + 1, 13):
        axis = fig.add_subplot(3, 4, empty_index)
        axis.axis("off")

    fig.suptitle(f"First Training Image Filter Responses, Label {label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# main function
def main(argv):
    """Loads the trained network, prints conv1 weights, and saves filter plots."""
    args = parse_args(argv)
    device = get_device()
    model = load_network(args.model_path, device)
    weights = get_conv1_weights(model)
    first_image, first_label = load_first_training_example(args.data_dir, args.download)

    lines = print_model_and_filters(model, weights)
    save_text_log(lines, args.output_dir / "network_analysis.txt")
    plot_conv1_filters(weights, args.output_dir / "conv1_filters.png")
    filtered_images = apply_filters_to_image(weights, first_image)
    plot_filtered_images(filtered_images, first_label, args.output_dir / "conv1_filter_effects.png")

    print(f"Saved model/filter text to {args.output_dir / 'network_analysis.txt'}")
    print(f"Saved conv1 filter plot to {args.output_dir / 'conv1_filters.png'}")
    print(f"Saved filter effect plot to {args.output_dir / 'conv1_filter_effects.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
