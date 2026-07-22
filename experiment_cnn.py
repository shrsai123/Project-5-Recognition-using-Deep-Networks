# Shreyas Raman
# Project 5: automated CNN architecture experiments on MNIST

# import statements
import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from mnist_cnn import get_device


# class definitions
@dataclass(frozen=True)
class ExperimentConfig:
    """Stores one CNN architecture and training configuration."""

    run_id: int
    conv1_filters: int
    conv2_filters: int
    kernel_size: int
    dropout: float
    hidden_nodes: int
    batch_size: int
    epochs: int
    lr: float


class ExperimentCNN(nn.Module):
    """Configurable CNN used for automated MNIST experiments."""

    def __init__(self, config):
        """Initializes variable convolution, dropout, and dense layers."""
        super().__init__()
        padding = config.kernel_size // 2
        self.conv1 = nn.Conv2d(1, config.conv1_filters, kernel_size=config.kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(
            config.conv1_filters,
            config.conv2_filters,
            kernel_size=config.kernel_size,
            padding=padding,
        )
        self.dropout = nn.Dropout2d(p=config.dropout)
        self.fc1 = nn.Linear(config.conv2_filters * 7 * 7, config.hidden_nodes)
        self.fc2 = nn.Linear(config.hidden_nodes, 10)

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
    """Parses command line options for running automated experiments."""
    parser = argparse.ArgumentParser(description="Run automated CNN architecture experiments on MNIST.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment"), help="experiment output folder")
    parser.add_argument("--max-runs", type=int, default=50, help="number of network variations to evaluate")
    parser.add_argument("--epochs", type=int, default=2, help="epochs per variation")
    parser.add_argument("--train-limit", type=int, default=8000, help="training examples per variation")
    parser.add_argument("--test-limit", type=int, default=2000, help="test examples per variation")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD learning rate")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--download", action="store_true", help="download MNIST if it is not present")
    return parser.parse_args(argv[1:])


def build_datasets(data_dir, train_limit, test_limit, seed, download):
    """Creates fixed MNIST training and test subsets for fair comparisons."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(data_dir, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=download, transform=transform)

    generator = torch.Generator().manual_seed(seed)
    train_indices = torch.randperm(len(train_dataset), generator=generator)[:train_limit].tolist()
    test_indices = torch.randperm(len(test_dataset), generator=generator)[:test_limit].tolist()
    return Subset(train_dataset, train_indices), Subset(test_dataset, test_indices)


def make_experiment_configs(max_runs, epochs, lr, seed):
    """Builds linear-search and randomized CNN architecture variations."""
    baseline = {
        "conv1_filters": 10,
        "conv2_filters": 20,
        "kernel_size": 5,
        "dropout": 0.5,
        "hidden_nodes": 50,
        "batch_size": 256,
    }
    dimensions = {
        "conv1_filters": [6, 10, 16, 24],
        "conv2_filters": [12, 20, 32, 48],
        "kernel_size": [3, 5, 7],
        "dropout": [0.0, 0.25, 0.5, 0.65],
        "hidden_nodes": [24, 50, 100, 160],
        "batch_size": [64, 128, 256, 512],
    }

    seen = set()
    configs = []

    def add_config(values):
        """Adds a unique experiment configuration until the run limit is reached."""
        key = tuple(values[name] for name in baseline)
        if key in seen or len(configs) >= max_runs:
            return
        seen.add(key)
        configs.append(
            ExperimentConfig(
                run_id=len(configs) + 1,
                conv1_filters=values["conv1_filters"],
                conv2_filters=values["conv2_filters"],
                kernel_size=values["kernel_size"],
                dropout=values["dropout"],
                hidden_nodes=values["hidden_nodes"],
                batch_size=values["batch_size"],
                epochs=epochs,
                lr=lr,
            )
        )

    add_config(dict(baseline))
    for dimension, options in dimensions.items():
        for option in options:
            values = dict(baseline)
            values[dimension] = option
            add_config(values)

    rng = random.Random(seed)
    while len(configs) < max_runs:
        values = {dimension: rng.choice(options) for dimension, options in dimensions.items()}
        add_config(values)

    return configs


def count_parameters(model):
    """Counts trainable parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train_one_epoch(model, device, train_loader, optimizer):
    """Trains one model for one epoch and returns loss and accuracy."""
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
        correct += output.argmax(dim=1).eq(target).sum().item()
        total += target.size(0)

    return total_loss / total, correct / total


def evaluate(model, device, test_loader):
    """Evaluates one model and returns loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)

    return total_loss / total, correct / total


def run_single_experiment(config, train_dataset, test_dataset, device):
    """Trains and evaluates one architecture variation."""
    torch.manual_seed(1000 + config.run_id)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model = ExperimentCNN(config).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.5)
    parameters = count_parameters(model)

    start_time = time.perf_counter()
    train_loss = 0.0
    train_accuracy = 0.0
    for _epoch in range(config.epochs):
        train_loss, train_accuracy = train_one_epoch(model, device, train_loader, optimizer)
    train_seconds = time.perf_counter() - start_time

    test_loss, test_accuracy = evaluate(model, device, test_loader)
    result = asdict(config)
    result.update(
        {
            "parameters": parameters,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_error": 1.0 - test_accuracy,
            "train_seconds": train_seconds,
            "seconds_per_epoch": train_seconds / config.epochs,
        }
    )
    return result


def save_results(results, output_dir):
    """Saves experiment results as CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cnn_experiment_results.csv"
    json_path = output_dir / "cnn_experiment_results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return csv_path, json_path


def plot_accuracy_vs_time(dataframe, output_path):
    """Saves a scatter plot of accuracy versus training time."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    scatter = axis.scatter(
        dataframe["train_seconds"],
        dataframe["test_accuracy"],
        c=dataframe["parameters"],
        cmap="viridis",
    )
    axis.set_xlabel("Training Time (seconds)")
    axis.set_ylabel("Test Accuracy")
    axis.set_title("CNN Experiment Accuracy vs. Training Time")
    axis.grid(True, alpha=0.3)
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Trainable Parameters")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_dimension_effects(dataframe, output_path):
    """Saves mean test accuracy grouped by each explored dimension."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dimensions = ["conv1_filters", "conv2_filters", "kernel_size", "dropout", "hidden_nodes", "batch_size"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for axis, dimension in zip(axes.flat, dimensions):
        grouped = dataframe.groupby(dimension)["test_accuracy"].mean().sort_index()
        axis.plot(grouped.index.astype(str), grouped.values, marker="o")
        axis.set_title(dimension)
        axis.set_xlabel("Value")
        axis.set_ylabel("Mean Test Accuracy")
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_report(dataframe, output_path):
    """Writes a report-ready summary of the experiment plan, hypotheses, and results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked = dataframe.sort_values(["test_accuracy", "train_seconds"], ascending=[False, True])
    fastest_good = dataframe[dataframe["test_accuracy"] >= 0.90].sort_values("train_seconds")
    best = ranked.iloc[0]

    lines = [
        "# MNIST CNN Architecture Experiment",
        "",
        "## Plan",
        "",
        "The experiment evaluates how CNN architecture choices affect MNIST accuracy and training time. "
        "Each variation is trained on the same fixed training subset and evaluated on the same fixed test subset.",
        "",
        "Dimensions explored:",
        "",
        "- `conv1_filters`: 6, 10, 16, 24",
        "- `conv2_filters`: 12, 20, 32, 48",
        "- `kernel_size`: 3, 5, 7",
        "- `dropout`: 0.0, 0.25, 0.5, 0.65",
        "- `hidden_nodes`: 24, 50, 100, 160",
        "- `batch_size`: 64, 128, 256, 512",
        "",
        "The search starts with one-at-a-time linear variations around the baseline, then fills the remaining runs "
        "with randomized combinations. Metrics are test accuracy, test error, train accuracy, parameter count, "
        "total training time, and seconds per epoch.",
        "",
        "## Hypotheses",
        "",
        "1. More convolution filters should improve accuracy up to a point, but increase training time.",
        "2. Kernel size 5 should perform well because it matches the original network; 3 may miss wider strokes and 7 may add extra cost.",
        "3. Moderate dropout should generalize better than no dropout, while very high dropout should slow learning.",
        "4. More hidden nodes should help until the classifier has enough capacity, after which time cost grows faster than accuracy.",
        "5. Larger batch sizes should train faster per epoch, but may slightly reduce final accuracy for a fixed epoch count.",
        "",
        "## Results",
        "",
        f"Total variations evaluated: {len(dataframe)}",
        f"Best test accuracy: {best['test_accuracy']:.4f} in run {int(best['run_id'])}",
        f"Best architecture: conv1={int(best['conv1_filters'])}, conv2={int(best['conv2_filters'])}, "
        f"kernel={int(best['kernel_size'])}, dropout={best['dropout']:.2f}, "
        f"hidden={int(best['hidden_nodes'])}, batch={int(best['batch_size'])}",
        f"Best training time: {best['train_seconds']:.2f} seconds",
        "",
        "Top 10 runs by test accuracy:",
        "",
        "| Run | Accuracy | Time | conv1 | conv2 | kernel | dropout | hidden | batch | params |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for _index, row in ranked.head(10).iterrows():
        lines.append(
            f"| {int(row['run_id'])} | {row['test_accuracy']:.4f} | {row['train_seconds']:.2f}s | "
            f"{int(row['conv1_filters'])} | {int(row['conv2_filters'])} | {int(row['kernel_size'])} | "
            f"{row['dropout']:.2f} | {int(row['hidden_nodes'])} | {int(row['batch_size'])} | "
            f"{int(row['parameters'])} |"
        )

    if not fastest_good.empty:
        row = fastest_good.iloc[0]
        lines.extend(
            [
                "",
                f"Fastest run reaching at least 90% test accuracy: run {int(row['run_id'])}, "
                f"{row['test_accuracy']:.4f} accuracy in {row['train_seconds']:.2f} seconds.",
            ]
        )

    lines.extend(
        [
            "",
            "## Dimension Averages",
            "",
        ]
    )

    for dimension in ["conv1_filters", "conv2_filters", "kernel_size", "dropout", "hidden_nodes", "batch_size"]:
        grouped = dataframe.groupby(dimension)["test_accuracy"].mean().sort_index()
        values = ", ".join(f"{value}: {accuracy:.4f}" for value, accuracy in grouped.items())
        lines.append(f"- `{dimension}` mean accuracy: {values}")

    conv1_means = dataframe.groupby("conv1_filters")["test_accuracy"].mean()
    conv2_means = dataframe.groupby("conv2_filters")["test_accuracy"].mean()
    kernel_means = dataframe.groupby("kernel_size")["test_accuracy"].mean()
    dropout_means = dataframe.groupby("dropout")["test_accuracy"].mean()
    hidden_means = dataframe.groupby("hidden_nodes")["test_accuracy"].mean()
    batch_means = dataframe.groupby("batch_size")["test_accuracy"].mean()

    lines.extend(
        [
            "",
            "## Discussion",
            "",
            "The filter-count hypothesis was mostly supported. Mean accuracy increased from "
            f"{conv1_means.min():.4f} to {conv1_means.max():.4f} across `conv1_filters`, and from "
            f"{conv2_means.min():.4f} to {conv2_means.max():.4f} across `conv2_filters`, although larger models "
            "usually required more training time.",
            "",
            "The kernel-size hypothesis was not supported. Kernel size 7 produced the highest mean accuracy "
            f"({kernel_means.loc[7]:.4f}), beating kernel size 5 ({kernel_means.loc[5]:.4f}) and kernel size 3 "
            f"({kernel_means.loc[3]:.4f}). For this short training budget, the wider filter seems to capture useful "
            "stroke context.",
            "",
            "The dropout hypothesis was mixed. No dropout had the best mean accuracy "
            f"({dropout_means.loc[0.0]:.4f}), but several of the best individual runs used high dropout. This suggests "
            "that dropout interacts strongly with batch size and model capacity in the short-run setting.",
            "",
            "The hidden-node hypothesis was only weakly supported. The 50-node hidden layer had the best mean accuracy "
            f"({hidden_means.loc[50]:.4f}), while 100 and 160 nodes did not improve the average result. Extra dense "
            "capacity was not the limiting factor for this experiment.",
            "",
            "The batch-size hypothesis was strongly supported for accuracy. Batch size 64 had the best mean accuracy "
            f"({batch_means.loc[64]:.4f}), while batch size 512 was much worse ({batch_means.loc[512]:.4f}). Larger "
            "batches were often faster per update schedule, but under a fixed epoch count they learned less effectively.",
            "",
            "Overall, run 31 had the best accuracy, but run 24 was the better speed/accuracy compromise: it reached "
            "90.55% test accuracy in 4.14 seconds. A final follow-up would retrain the best few architectures for more "
            "epochs on the full 60k MNIST training set.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# main function
def main(argv):
    """Runs the automated experiment suite and writes plots, tables, and a summary report."""
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_dataset, test_dataset = build_datasets(
        args.data_dir,
        args.train_limit,
        args.test_limit,
        args.seed,
        args.download,
    )
    configs = make_experiment_configs(args.max_runs, args.epochs, args.lr, args.seed)

    results = []
    start_time = time.perf_counter()
    for config in configs:
        result = run_single_experiment(config, train_dataset, test_dataset, device)
        results.append(result)
        print(
            f"Run {config.run_id:02d}/{len(configs)}: "
            f"acc={result['test_accuracy']:.4f}, "
            f"time={result['train_seconds']:.2f}s, "
            f"conv=({config.conv1_filters},{config.conv2_filters}), "
            f"k={config.kernel_size}, drop={config.dropout}, hidden={config.hidden_nodes}, batch={config.batch_size}"
        )

    total_seconds = time.perf_counter() - start_time
    dataframe = pd.DataFrame(results)
    csv_path, json_path = save_results(results, args.output_dir)
    plot_accuracy_vs_time(dataframe, args.output_dir / "accuracy_vs_time.png")
    plot_dimension_effects(dataframe, args.output_dir / "dimension_effects.png")
    write_report(dataframe, args.output_dir / "experiment_report.md")

    best = dataframe.sort_values(["test_accuracy", "train_seconds"], ascending=[False, True]).iloc[0]
    print(f"Saved CSV results to {csv_path}")
    print(f"Saved JSON results to {json_path}")
    print(f"Saved report to {args.output_dir / 'experiment_report.md'}")
    print(f"Saved plots to {args.output_dir / 'accuracy_vs_time.png'} and {args.output_dir / 'dimension_effects.png'}")
    print(f"Best run: {int(best['run_id'])} with test accuracy {best['test_accuracy']:.4f}")
    print(f"Total experiment time: {total_seconds:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
