# Shreyas Raman
# Project 5 extension: evaluate transfer-learning dimensions on Greek letters

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

from mnist_cnn import MyNetwork
from transfer_greek import build_greek_loader, evaluate_network, get_device


# class definitions
@dataclass(frozen=True)
class GreekExperimentConfig:
    """Stores one Greek transfer-learning experiment configuration."""

    run_id: int
    trainable_scope: str
    optimizer_name: str
    learning_rate: float
    batch_size: int
    dropout: float
    epochs: int
    target_accuracy: float


# useful functions with a comment for each function
def parse_args(argv):
    """Parses command line options for Greek transfer experiments."""
    parser = argparse.ArgumentParser(description="Evaluate Greek-letter transfer-learning dimensions.")
    parser.add_argument("--training-set-path", type=Path, default=Path("greek_train"), help="Greek ImageFolder path")
    parser.add_argument("--mnist-model-path", type=Path, default=Path("mnist_cnn.pt"), help="saved MNIST model path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/greek_experiment"), help="output directory")
    parser.add_argument("--max-runs", type=int, default=60, help="number of transfer variants to evaluate")
    parser.add_argument("--epochs", type=int, default=40, help="maximum epochs for each variant")
    parser.add_argument("--target-accuracy", type=float, default=1.0, help="early-stop accuracy target")
    parser.add_argument("--seed", type=int, default=11, help="random seed")
    return parser.parse_args(argv[1:])


def make_experiment_configs(max_runs, epochs, target_accuracy, seed):
    """Creates linear and randomized Greek transfer-learning variations."""
    baseline = {
        "trainable_scope": "fc2_only",
        "optimizer_name": "sgd",
        "learning_rate": 0.01,
        "batch_size": 5,
        "dropout": 0.5,
    }
    dimensions = {
        "trainable_scope": ["fc2_only", "fc1_fc2", "conv2_fc"],
        "optimizer_name": ["sgd", "adam"],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "batch_size": [3, 5, 9, 27],
        "dropout": [0.0, 0.25, 0.5, 0.65],
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
            GreekExperimentConfig(
                run_id=len(configs) + 1,
                trainable_scope=values["trainable_scope"],
                optimizer_name=values["optimizer_name"],
                learning_rate=values["learning_rate"],
                batch_size=values["batch_size"],
                dropout=values["dropout"],
                epochs=epochs,
                target_accuracy=target_accuracy,
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


def set_trainable_scope(network, scope):
    """Freezes all parameters, then unfreezes the selected transfer-learning layers."""
    for param in network.parameters():
        param.requires_grad = False

    for param in network.fc2.parameters():
        param.requires_grad = True

    if scope in {"fc1_fc2", "conv2_fc"}:
        for param in network.fc1.parameters():
            param.requires_grad = True

    if scope == "conv2_fc":
        for param in network.conv2.parameters():
            param.requires_grad = True


def build_transfer_model(config, mnist_model_path, device):
    """Loads MNIST weights, replaces the final layer, and applies an unfreeze strategy."""
    network = MyNetwork().to(device)
    state_dict = torch.load(mnist_model_path, map_location=device, weights_only=True)
    network.load_state_dict(state_dict)
    network.dropout.p = config.dropout
    network.fc2 = nn.Linear(50, 3).to(device)
    set_trainable_scope(network, config.trainable_scope)
    return network


def build_optimizer(config, network):
    """Creates the configured optimizer over trainable parameters only."""
    trainable_parameters = [param for param in network.parameters() if param.requires_grad]
    if config.optimizer_name == "adam":
        return optim.Adam(trainable_parameters, lr=config.learning_rate)
    return optim.SGD(trainable_parameters, lr=config.learning_rate, momentum=0.5)


def count_trainable_parameters(network):
    """Counts parameters updated during transfer learning."""
    return sum(param.numel() for param in network.parameters() if param.requires_grad)


def train_one_epoch(network, device, greek_train, optimizer):
    """Trains one Greek transfer model for one epoch."""
    network.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in greek_train:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        correct += output.argmax(dim=1).eq(target).sum().item()
        total += target.size(0)

    return total_loss / total, correct / total


def run_single_experiment(config, training_set_path, mnist_model_path, device):
    """Runs one Greek transfer-learning variation and returns metrics."""
    torch.manual_seed(2000 + config.run_id)
    greek_train, greek_dataset = build_greek_loader(training_set_path, config.batch_size)
    network = build_transfer_model(config, mnist_model_path, device)
    optimizer = build_optimizer(config, network)
    trainable_parameters = count_trainable_parameters(network)

    best_accuracy = 0.0
    final_loss = 0.0
    final_train_accuracy = 0.0
    reached_epoch = None
    start_time = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(network, device, greek_train, optimizer)
        eval_loss, eval_accuracy, _prediction, _target = evaluate_network(network, device, greek_dataset)
        best_accuracy = max(best_accuracy, eval_accuracy)
        final_loss = eval_loss
        final_train_accuracy = train_accuracy

        if eval_accuracy >= config.target_accuracy:
            reached_epoch = epoch
            break

    train_seconds = time.perf_counter() - start_time
    result = asdict(config)
    result.update(
        {
            "epochs_run": reached_epoch if reached_epoch is not None else config.epochs,
            "reached_target": reached_epoch is not None,
            "final_eval_loss": final_loss,
            "final_train_accuracy": final_train_accuracy,
            "best_eval_accuracy": best_accuracy,
            "trainable_parameters": trainable_parameters,
            "train_seconds": train_seconds,
            "seconds_per_epoch": train_seconds / (reached_epoch if reached_epoch is not None else config.epochs),
        }
    )
    return result


def save_results(results, output_dir):
    """Saves Greek transfer experiment results as CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "greek_transfer_experiment_results.csv"
    json_path = output_dir / "greek_transfer_experiment_results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return csv_path, json_path


def plot_epochs_to_target(dataframe, output_path):
    """Plots epochs to perfect training recognition for successful runs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    successful = dataframe[dataframe["reached_target"]].copy()
    fig, axis = plt.subplots(figsize=(7, 4.5))

    if successful.empty:
        axis.text(0.5, 0.5, "No runs reached target accuracy", ha="center", va="center")
        axis.axis("off")
    else:
        grouped = successful.groupby("trainable_scope")["epochs_run"].mean().sort_values()
        axis.bar(grouped.index, grouped.values, color="tab:blue")
        axis.set_xlabel("Trainable Scope")
        axis.set_ylabel("Mean Epochs To 100%")
        axis.set_title("Greek Transfer: Epochs To Perfect Recognition")
        axis.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_dimension_success(dataframe, output_path):
    """Plots target-reaching rate grouped by each explored dimension."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dimensions = ["trainable_scope", "optimizer_name", "learning_rate", "batch_size", "dropout"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for axis, dimension in zip(axes.flat, dimensions):
        grouped = dataframe.groupby(dimension)["reached_target"].mean().sort_index()
        axis.plot(grouped.index.astype(str), grouped.values, marker="o")
        axis.set_title(dimension)
        axis.set_xlabel("Value")
        axis.set_ylabel("Success Rate")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True, alpha=0.3)

    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_report(dataframe, output_path):
    """Writes a report-ready summary of the Greek transfer extension."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked = dataframe.sort_values(
        ["reached_target", "epochs_run", "train_seconds", "best_eval_accuracy"],
        ascending=[False, True, True, False],
    )
    successful = dataframe[dataframe["reached_target"]]
    best = ranked.iloc[0]

    lines = [
        "# Greek Transfer Learning Extension",
        "",
        "## Plan",
        "",
        "This extension evaluates which transfer-learning choices make the pretrained MNIST network learn "
        "alpha, beta, and gamma fastest. Each run starts from the saved MNIST network, replaces `fc2` with "
        "a 3-output linear layer, and trains on the same 27 Greek examples.",
        "",
        "Dimensions explored:",
        "",
        "- `trainable_scope`: `fc2_only`, `fc1_fc2`, `conv2_fc`",
        "- `optimizer_name`: `sgd`, `adam`",
        "- `learning_rate`: 0.001, 0.01, 0.05, 0.1",
        "- `batch_size`: 3, 5, 9, 27",
        "- `dropout`: 0.0, 0.25, 0.5, 0.65",
        "",
        "Metrics:",
        "",
        "- Whether the model reaches 100% accuracy on the 27 training examples",
        "- Epochs needed to reach 100%",
        "- Training time",
        "- Trainable parameter count",
        "- Best evaluation accuracy on all 27 examples",
        "",
        "## Hypotheses",
        "",
        "1. Training only `fc2` should usually be fastest and least likely to overfit, because the Greek dataset is tiny.",
        "2. Adam should reach perfect recognition in fewer epochs than SGD, especially at smaller learning rates.",
        "3. Very large learning rates should be unstable for the small final layer.",
        "4. Smaller batch sizes should learn faster because they create more weight updates per epoch.",
        "5. Dropout 0.5 should be a reasonable default, but lower dropout may fit the 27 examples faster.",
        "",
        "## Results",
        "",
        f"Total variations evaluated: {len(dataframe)}",
        f"Runs reaching 100% accuracy: {len(successful)}",
        f"Best run: {int(best['run_id'])}",
        f"Best configuration: scope={best['trainable_scope']}, optimizer={best['optimizer_name']}, "
        f"lr={best['learning_rate']}, batch={int(best['batch_size'])}, dropout={best['dropout']}",
        f"Best run epochs to target: {int(best['epochs_run'])}",
        f"Best run time: {best['train_seconds']:.3f} seconds",
        "",
        "Top 10 runs:",
        "",
        "| Run | Target? | Epochs | Accuracy | Time | Scope | Optimizer | LR | Batch | Dropout | Params |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for _index, row in ranked.head(10).iterrows():
        lines.append(
            f"| {int(row['run_id'])} | {bool(row['reached_target'])} | {int(row['epochs_run'])} | "
            f"{row['best_eval_accuracy']:.4f} | {row['train_seconds']:.3f}s | {row['trainable_scope']} | "
            f"{row['optimizer_name']} | {row['learning_rate']} | {int(row['batch_size'])} | "
            f"{row['dropout']:.2f} | {int(row['trainable_parameters'])} |"
        )

    lines.extend(["", "## Dimension Averages", ""])
    for dimension in ["trainable_scope", "optimizer_name", "learning_rate", "batch_size", "dropout"]:
        success = dataframe.groupby(dimension)["reached_target"].mean().sort_index()
        epochs = dataframe[dataframe["reached_target"]].groupby(dimension)["epochs_run"].mean().sort_index()
        success_values = ", ".join(f"{value}: {rate:.2f}" for value, rate in success.items())
        epoch_values = ", ".join(f"{value}: {value_epochs:.2f}" for value, value_epochs in epochs.items())
        lines.append(f"- `{dimension}` success rate: {success_values}")
        lines.append(f"- `{dimension}` mean epochs among successful runs: {epoch_values if epoch_values else 'none'}")

    lines.extend(["", "## Discussion", ""])

    if not successful.empty:
        fastest_scope = successful.groupby("trainable_scope")["epochs_run"].mean().sort_values().index[0]
        fastest_optimizer = successful.groupby("optimizer_name")["epochs_run"].mean().sort_values().index[0]
        fastest_batch = successful.groupby("batch_size")["epochs_run"].mean().sort_values().index[0]
        fastest_dropout = successful.groupby("dropout")["epochs_run"].mean().sort_values().index[0]
        lines.extend(
            [
                f"The fastest average trainable scope among successful runs was `{fastest_scope}`. This directly tests "
                "whether the original frozen-feature transfer strategy is enough or whether unfreezing deeper layers helps.",
                "",
                f"The fastest average optimizer among successful runs was `{fastest_optimizer}`. This shows which optimizer "
                "adapted the replacement classifier most efficiently on the tiny 27-image dataset.",
                "",
                f"The best average successful batch size was `{fastest_batch}`. Smaller batches usually perform more updates "
                "per epoch, while larger batches make each epoch more stable but can need more epochs.",
                "",
                f"The fastest average successful dropout value was `{fastest_dropout}`. This helps decide whether the original "
                "MNIST dropout setting is still appropriate when only a few Greek examples are available.",
                "",
                "Because the dataset has only 27 examples, these results measure memorization and transfer efficiency rather "
                "than true generalization. The useful extension is that it compares how quickly each transfer strategy can "
                "adapt the pretrained network before testing on custom handwritten Greek symbols.",
            ]
        )
    else:
        lines.append(
            "No run reached 100% accuracy within the epoch budget, so the next step would be to raise the epoch limit "
            "or use less aggressive freezing."
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# main function
def main(argv):
    """Runs the Greek transfer extension sweep and saves tables, plots, and a report."""
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}", flush=True)

    configs = make_experiment_configs(args.max_runs, args.epochs, args.target_accuracy, args.seed)
    results = []
    start_time = time.perf_counter()

    for config in configs:
        result = run_single_experiment(config, args.training_set_path, args.mnist_model_path, device)
        results.append(result)
        print(
            f"Run {config.run_id:02d}/{len(configs)}: "
            f"target={result['reached_target']}, epochs={result['epochs_run']}, "
            f"acc={result['best_eval_accuracy']:.4f}, time={result['train_seconds']:.3f}s, "
            f"scope={config.trainable_scope}, opt={config.optimizer_name}, "
            f"lr={config.learning_rate}, batch={config.batch_size}, dropout={config.dropout}",
            flush=True,
        )

    total_seconds = time.perf_counter() - start_time
    dataframe = pd.DataFrame(results)
    csv_path, json_path = save_results(results, args.output_dir)
    plot_epochs_to_target(dataframe, args.output_dir / "epochs_to_target.png")
    plot_dimension_success(dataframe, args.output_dir / "dimension_success.png")
    write_report(dataframe, args.output_dir / "greek_transfer_experiment_report.md")

    ranked = dataframe.sort_values(
        ["reached_target", "epochs_run", "train_seconds", "best_eval_accuracy"],
        ascending=[False, True, True, False],
    )
    best = ranked.iloc[0]
    print(f"Saved CSV results to {csv_path}")
    print(f"Saved JSON results to {json_path}")
    print(f"Saved report to {args.output_dir / 'greek_transfer_experiment_report.md'}")
    print(f"Saved plots to {args.output_dir / 'epochs_to_target.png'} and {args.output_dir / 'dimension_success.png'}")
    print(f"Best run: {int(best['run_id'])} reached target={bool(best['reached_target'])} in {int(best['epochs_run'])} epochs")
    print(f"Total experiment time: {total_seconds:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
