# Shreyas Raman
# Project 5 extension: examine early filters in a pretrained PyTorch network

# import statements
import argparse
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
from torchvision import datasets, transforms


# useful functions with a comment for each function
def parse_args(argv):
    """Parses command line options for pretrained ResNet analysis."""
    parser = argparse.ArgumentParser(description="Examine early convolution layers in pretrained ResNet-18.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pretrained_resnet"), help="output directory")
    parser.add_argument("--download", action="store_true", help="download MNIST if it is not present")
    parser.add_argument("--num-filters", type=int, default=16, help="number of filters to visualize from each layer")
    return parser.parse_args(argv[1:])


def load_pretrained_model():
    """Loads a pretrained ResNet-18 model from torchvision and sets eval mode."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    return model


def load_first_training_example(data_dir, download):
    """Loads the first MNIST training image as grayscale and RGB arrays."""
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    image_tensor, label = train_dataset[0]
    grayscale = image_tensor.squeeze(0).numpy()
    rgb = np.repeat(grayscale[:, :, None], 3, axis=2)
    return grayscale, rgb, label


def normalize_for_display(image):
    """Normalizes one image to the 0-1 range for pyplot display."""
    image = image.astype(np.float32)
    minimum = image.min()
    maximum = image.max()
    if maximum - minimum < 1e-8:
        return np.zeros_like(image)
    return (image - minimum) / (maximum - minimum)


def first_conv_response(weights, rgb_image, output_index):
    """Applies one RGB first-layer filter to an RGB image using OpenCV."""
    response = np.zeros(rgb_image.shape[:2], dtype=np.float32)
    for channel in range(3):
        kernel = weights[output_index, channel].numpy()
        response += cv2.filter2D(rgb_image[:, :, channel], ddepth=cv2.CV_32F, kernel=kernel)
    return response


def feature_conv_response(weights, feature_maps, output_index):
    """Applies one deeper convolution filter to feature maps using OpenCV."""
    response = np.zeros(feature_maps.shape[1:], dtype=np.float32)
    input_channels = min(weights.shape[1], feature_maps.shape[0])
    for channel in range(input_channels):
        kernel = weights[output_index, channel].numpy()
        response += cv2.filter2D(feature_maps[channel], ddepth=cv2.CV_32F, kernel=kernel)
    return response


def compute_first_feature_maps(model, rgb_image):
    """Computes ResNet feature maps before layer1.0.conv1."""
    tensor = torch.from_numpy(rgb_image.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        features = model.conv1(tensor)
        features = model.bn1(features)
        features = model.relu(features)
        features = model.maxpool(features)
    return features.squeeze(0).cpu().numpy()


def plot_rgb_filters(weights, output_path, title, num_filters):
    """Saves a grid visualization of RGB convolution filters."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(num_filters, weights.shape[0])
    columns = 4
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.2 * rows))
    axes = list(np.array(axes).reshape(-1))

    for axis in axes:
        axis.axis("off")

    for index in range(count):
        filter_image = weights[index].permute(1, 2, 0).numpy()
        axes[index].imshow(normalize_for_display(filter_image))
        axes[index].set_title(f"Filter {index}")
        axes[index].set_xticks([])
        axes[index].set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_channel_averaged_filters(weights, output_path, title, num_filters):
    """Saves deeper filters as channel-averaged 2D kernels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(num_filters, weights.shape[0])
    columns = 4
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.2 * rows))
    axes = list(np.array(axes).reshape(-1))

    for axis in axes:
        axis.axis("off")

    for index in range(count):
        filter_image = weights[index].mean(dim=0).numpy()
        axes[index].imshow(filter_image, cmap="viridis")
        axes[index].set_title(f"Filter {index}")
        axes[index].set_xticks([])
        axes[index].set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_responses(responses, output_path, title, num_filters):
    """Saves a grid of filter responses."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(num_filters, len(responses))
    columns = 4
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.2 * rows))
    axes = list(np.array(axes).reshape(-1))

    for axis in axes:
        axis.axis("off")

    for index in range(count):
        axes[index].imshow(responses[index], cmap="gray")
        axes[index].set_title(f"Filter {index}")
        axes[index].set_xticks([])
        axes[index].set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_and_save_analysis(model, conv1_weights, layer1_weights, output_path):
    """Prints and saves model structure plus selected convolution weights."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    def record(text=""):
        """Prints one line and stores it for the analysis log."""
        print(text)
        lines.append(str(text))

    record("Pretrained model: torchvision.models.resnet18")
    record()
    record("Model structure:")
    record(model)
    record()
    record(f"conv1 weight shape: {tuple(conv1_weights.shape)}")
    record("conv1 first filter, red channel weights:")
    record(conv1_weights[0, 0].numpy())
    record()
    record(f"layer1.0.conv1 weight shape: {tuple(layer1_weights.shape)}")
    record("layer1.0.conv1 first output filter, first input-channel weights:")
    record(layer1_weights[0, 0].numpy())

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# main function
def main(argv):
    """Loads pretrained ResNet-18 and analyzes its first two convolutional layers."""
    args = parse_args(argv)
    model = load_pretrained_model()
    grayscale, rgb_image, label = load_first_training_example(args.data_dir, args.download)

    with torch.no_grad():
        conv1_weights = model.conv1.weight.detach().cpu().clone()
        layer1_weights = model.layer1[0].conv1.weight.detach().cpu().clone()
        first_features = compute_first_feature_maps(model, rgb_image)
        conv1_responses = [
            first_conv_response(conv1_weights, rgb_image, index)
            for index in range(min(args.num_filters, conv1_weights.shape[0]))
        ]
        layer1_responses = [
            feature_conv_response(layer1_weights, first_features, index)
            for index in range(min(args.num_filters, layer1_weights.shape[0]))
        ]

    print_and_save_analysis(
        model,
        conv1_weights,
        layer1_weights,
        args.output_dir / "pretrained_resnet_analysis.txt",
    )

    plot_rgb_filters(
        conv1_weights,
        args.output_dir / "resnet_conv1_filters.png",
        "ResNet-18 conv1 RGB Filters",
        args.num_filters,
    )
    plot_channel_averaged_filters(
        layer1_weights,
        args.output_dir / "resnet_layer1_0_conv1_filters.png",
        "ResNet-18 layer1.0.conv1 Channel-Averaged Filters",
        args.num_filters,
    )
    plot_responses(
        conv1_responses,
        args.output_dir / "resnet_conv1_filter_effects.png",
        f"ResNet-18 conv1 Responses on First MNIST Training Image, Label {label}",
        args.num_filters,
    )
    plot_responses(
        layer1_responses,
        args.output_dir / "resnet_layer1_0_conv1_filter_effects.png",
        "ResNet-18 layer1.0.conv1 Responses After Initial ResNet Stem",
        args.num_filters,
    )

    fig, axis = plt.subplots(figsize=(3, 3))
    axis.imshow(grayscale, cmap="gray")
    axis.set_title(f"Input MNIST Image, Label {label}")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(args.output_dir / "resnet_input_mnist_image.png", dpi=150)
    plt.close(fig)

    print(f"Saved analysis text to {args.output_dir / 'pretrained_resnet_analysis.txt'}")
    print(f"Saved conv1 filter plot to {args.output_dir / 'resnet_conv1_filters.png'}")
    print(f"Saved layer1 filter plot to {args.output_dir / 'resnet_layer1_0_conv1_filters.png'}")
    print(f"Saved conv1 response plot to {args.output_dir / 'resnet_conv1_filter_effects.png'}")
    print(f"Saved layer1 response plot to {args.output_dir / 'resnet_layer1_0_conv1_filter_effects.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
