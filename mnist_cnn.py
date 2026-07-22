# Shreyas Raman
# Project 5: MNIST digit recognition with a convolutional neural network

# import statements
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# class definitions
class MyNetwork(nn.Module):
    """Convolutional neural network for recognizing MNIST digits."""

    def __init__(self):
        """Initializes the convolution, dropout, and fully connected layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

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
    """Parses command line options for training and output paths."""
    parser = argparse.ArgumentParser(description="Train a CNN on the MNIST digit dataset.")
    parser.add_argument("--batch-size", type=int, default=64, help="training batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000, help="testing batch size")
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="figure output directory")
    parser.add_argument("--model-path", type=Path, default=Path("mnist_cnn.pt"), help="saved model path")
    parser.add_argument("--no-download", action="store_true", help="use an existing local MNIST dataset")
    return parser.parse_args(argv[1:])


def get_device():
    """Selects the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_data_loaders(data_dir, batch_size, test_batch_size, download):
    """Creates DataLoader objects for the MNIST training and test sets."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader, test_dataset


def plot_first_six_digits(test_dataset, output_path):
    """Saves a subplot figure containing the first six MNIST test images."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))

    for index, axis in enumerate(axes.flat):
        image, label = test_dataset[index]
        axis.imshow(image.squeeze(0), cmap="gray")
        axis.set_title(f"Label: {label}")
        axis.axis("off")

    fig.suptitle("First six MNIST test examples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_network_diagram(output_path):
    """Saves a simple architecture diagram of the required CNN."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = [
        ("Input", "1 x 28 x 28"),
        ("Conv 5x5", "10 maps"),
        ("MaxPool\n+ ReLU", "2 x 2"),
        ("Conv 5x5", "20 maps"),
        ("Dropout", "p = 0.5"),
        ("MaxPool\n+ ReLU", "2 x 2"),
        ("Flatten", "320 values"),
        ("Linear\n+ ReLU", "50 nodes"),
        ("Linear\n+ LogSoftmax", "10 digits"),
    ]

    fig, axis = plt.subplots(figsize=(13, 3.8))
    axis.set_xlim(0, len(layers))
    axis.set_ylim(0, 1)
    axis.axis("off")

    box_width = 0.78
    colors = ["#d9ead3", "#cfe2f3", "#fff2cc", "#cfe2f3", "#eadcf8"]

    for index, (name, detail) in enumerate(layers):
        x_position = index + 0.08
        color = colors[index % len(colors)]
        rectangle = plt.Rectangle(
            (x_position, 0.28),
            box_width,
            0.44,
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.2,
        )
        axis.add_patch(rectangle)
        axis.text(
            x_position + box_width / 2,
            0.56,
            name,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
        axis.text(
            x_position + box_width / 2,
            0.43,
            detail,
            ha="center",
            va="center",
            fontsize=8,
        )

        if index < len(layers) - 1:
            axis.annotate(
                "",
                xy=(index + 1.04, 0.5),
                xytext=(index + 0.91, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.2},
            )

    axis.set_title("MNIST CNN Architecture", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_one_epoch(model, device, train_loader, optimizer):
    """Trains the model for one complete pass through the training data."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        prediction = output.argmax(dim=1)
        correct += prediction.eq(target).sum().item()
        total += target.size(0)

    average_loss = total_loss / total
    accuracy = correct / total
    error = 1.0 - accuracy
    return average_loss, accuracy, error


def evaluate_network(model, device, data_loader):
    """Evaluates loss, accuracy, and error without updating model weights."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            total_loss += F.nll_loss(output, target, reduction="sum").item()
            prediction = output.argmax(dim=1)
            correct += prediction.eq(target).sum().item()
            total += target.size(0)

    average_loss = total_loss / total
    accuracy = correct / total
    error = 1.0 - accuracy
    return average_loss, accuracy, error


def train_network(model, device, train_loader, test_loader, optimizer, epochs):
    """Trains for the requested epochs and records train/test metrics."""
    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "train_error": [],
        "test_error": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy, train_error = train_one_epoch(
            model,
            device,
            train_loader,
            optimizer,
        )
        test_loss, test_accuracy, test_error = evaluate_network(model, device, test_loader)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_accuracy"].append(test_accuracy)
        history["train_error"].append(train_error)
        history["test_error"].append(test_error)

        print(
            f"Epoch {epoch}: "
            f"train accuracy={train_accuracy:.4f}, test accuracy={test_accuracy:.4f}, "
            f"train error={train_error:.4f}, test error={test_error:.4f}"
        )

    return history


def plot_metric(history, train_key, test_key, ylabel, title, output_path):
    """Saves a train-versus-test metric plot for the report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4.5))

    axis.plot(history["epoch"], history[train_key], marker="o", color="tab:blue", label="Training")
    axis.plot(history["epoch"], history[test_key], marker="s", color="tab:orange", label="Testing")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xticks(history["epoch"])
    axis.grid(True, alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_model(model, model_path):
    """Saves the trained model parameters to a file."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)


# main function (yes, it needs a comment too)
def main(argv):
    """Builds data loaders, trains the network, saves figures, and writes the model."""
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, test_dataset = build_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        download=not args.no_download,
    )

    plot_first_six_digits(test_dataset, args.output_dir / "first_six_test_digits.png")
    plot_network_diagram(args.output_dir / "network_diagram.png")

    model = MyNetwork().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    history = train_network(model, device, train_loader, test_loader, optimizer, args.epochs)

    plot_metric(
        history,
        train_key="train_error",
        test_key="test_error",
        ylabel="Error Rate",
        title="MNIST Training and Testing Error",
        output_path=args.output_dir / "training_testing_error.png",
    )
    plot_metric(
        history,
        train_key="train_accuracy",
        test_key="test_accuracy",
        ylabel="Accuracy",
        title="MNIST Training and Testing Accuracy",
        output_path=args.output_dir / "training_testing_accuracy.png",
    )
    save_model(model, args.model_path)

    print(f"Saved first-six digit plot to {args.output_dir / 'first_six_test_digits.png'}")
    print(f"Saved network diagram to {args.output_dir / 'network_diagram.png'}")
    print(f"Saved error plot to {args.output_dir / 'training_testing_error.png'}")
    print(f"Saved accuracy plot to {args.output_dir / 'training_testing_accuracy.png'}")
    print(f"Saved trained network to {args.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
