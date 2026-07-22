# Shreyas Raman
# Project 5 extension: fixed Gabor filter bank as the first MNIST layer

# import statements
import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mnist_cnn import build_data_loaders, get_device, plot_metric, save_model, train_network


# class definitions
class FixedGaborNetwork(nn.Module):
    """MNIST CNN with a frozen first layer of hand-designed Gabor filters."""

    def __init__(self, num_filters=10, kernel_size=7, dropout=0.5):
        """Initializes a fixed Gabor conv1 layer and trainable later layers."""
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(num_filters, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(20 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 10)
        self.initialize_gabor_filters()

    # initializes the fixed first layer with Gabor filters
    def initialize_gabor_filters(self):
        """Builds a bank of normalized Gabor filters and freezes conv1."""
        filters = make_gabor_filter_bank(self.num_filters, self.kernel_size)
        with torch.no_grad():
            self.conv1.weight.copy_(filters)
        for param in self.conv1.parameters():
            param.requires_grad = False

    # computes a forward pass for the network
    def forward(self, x):
        """Runs one forward pass and returns log probabilities for 10 digits."""
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), kernel_size=2))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# useful functions with a comment for each function
def parse_args(argv):
    """Parses command line options for fixed-Gabor MNIST training."""
    parser = argparse.ArgumentParser(description="Train MNIST with a frozen Gabor filter first layer.")
    parser.add_argument("--batch-size", type=int, default=256, help="training batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000, help="testing batch size")
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gabor_fixed"), help="output directory")
    parser.add_argument("--model-path", type=Path, default=Path("mnist_gabor_fixed.pt"), help="saved model path")
    parser.add_argument("--no-download", action="store_true", help="use an existing local MNIST dataset")
    parser.add_argument("--num-filters", type=int, default=10, help="number of fixed Gabor filters")
    parser.add_argument("--kernel-size", type=int, default=7, help="Gabor filter size")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    return parser.parse_args(argv[1:])


def make_gabor_filter_bank(num_filters, kernel_size):
    """Creates a [num_filters, 1, kernel_size, kernel_size] Gabor filter bank."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    radius = kernel_size // 2
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    filters = []

    for index in range(num_filters):
        theta = math.pi * index / num_filters
        wavelength = 3.0 if index < num_filters // 2 else 5.0
        sigma = 2.0
        gamma = 0.6
        phase = 0.0 if index % 2 == 0 else math.pi / 2

        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
        envelope = torch.exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2))
        carrier = torch.cos((2 * math.pi * x_theta / wavelength) + phase)
        kernel = envelope * carrier
        kernel = kernel - kernel.mean()
        kernel = kernel / (kernel.norm() + 1e-8)
        filters.append(kernel)

    return torch.stack(filters).unsqueeze(1)


def plot_gabor_filters(model, output_path):
    """Saves a grid visualization of the fixed Gabor filter bank."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights = model.conv1.weight.detach().cpu()
    columns = 5
    rows = math.ceil(weights.shape[0] / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.0 * columns, 2.0 * rows))
    axes = list(torch.tensor([], dtype=torch.float32).numpy().reshape(0)) if False else list(axes.flat)

    for axis in axes:
        axis.axis("off")

    for index in range(weights.shape[0]):
        axes[index].imshow(weights[index, 0].numpy(), cmap="viridis")
        axes[index].set_title(f"Gabor {index}")
        axes[index].set_xticks([])
        axes[index].set_yticks([])

    fig.suptitle("Frozen Gabor Filter Bank")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_gabor_log(model, history, output_path):
    """Writes the fixed-filter model printout and training history."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    lines = [
        "Fixed Gabor MNIST model:",
        str(model),
        "",
        f"Frozen parameters: {frozen}",
        f"Trainable parameters: {trainable}",
        f"conv1 requires_grad: {model.conv1.weight.requires_grad}",
        "",
        "epoch,train_loss,test_loss,train_accuracy,test_accuracy,train_error,test_error",
    ]

    for index, epoch in enumerate(history["epoch"]):
        lines.append(
            f"{epoch},"
            f"{history['train_loss'][index]:.6f},"
            f"{history['test_loss'][index]:.6f},"
            f"{history['train_accuracy'][index]:.6f},"
            f"{history['test_accuracy'][index]:.6f},"
            f"{history['train_error'][index]:.6f},"
            f"{history['test_error'][index]:.6f}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# main function
def main(argv):
    """Trains the MNIST network while holding the Gabor first layer constant."""
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, _test_dataset = build_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        download=not args.no_download,
    )

    model = FixedGaborNetwork(
        num_filters=args.num_filters,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)
    print("Fixed Gabor model:")
    print(model)
    print(f"conv1 requires_grad: {model.conv1.weight.requires_grad}")

    optimizer = optim.SGD((param for param in model.parameters() if param.requires_grad), lr=args.lr, momentum=args.momentum)
    history = train_network(model, device, train_loader, test_loader, optimizer, args.epochs)

    plot_gabor_filters(model, args.output_dir / "gabor_filter_bank.png")
    plot_metric(
        history,
        train_key="train_error",
        test_key="test_error",
        ylabel="Error Rate",
        title="Fixed Gabor CNN Training and Testing Error",
        output_path=args.output_dir / "gabor_training_testing_error.png",
    )
    plot_metric(
        history,
        train_key="train_accuracy",
        test_key="test_accuracy",
        ylabel="Accuracy",
        title="Fixed Gabor CNN Training and Testing Accuracy",
        output_path=args.output_dir / "gabor_training_testing_accuracy.png",
    )
    write_gabor_log(model, history, args.output_dir / "gabor_training_log.txt")
    save_model(model, args.model_path)

    print(f"Saved Gabor filter plot to {args.output_dir / 'gabor_filter_bank.png'}")
    print(f"Saved accuracy plot to {args.output_dir / 'gabor_training_testing_accuracy.png'}")
    print(f"Saved error plot to {args.output_dir / 'gabor_training_testing_error.png'}")
    print(f"Saved training log to {args.output_dir / 'gabor_training_log.txt'}")
    print(f"Saved trained model to {args.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
