# Shreyas Raman
# Project 5: run the saved MNIST network on test examples

# import statements
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from mnist_cnn import MyNetwork


# useful functions with a comment for each function
def parse_args(argv):
    """Parses command line options for running the saved network."""
    parser = argparse.ArgumentParser(description="Run a saved MNIST CNN on test examples.")
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


def load_test_dataset(data_dir, download):
    """Loads the MNIST test dataset without shuffling the examples."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)


def load_network(model_path, device):
    """Reads saved model weights and puts the network in evaluation mode."""
    model = MyNetwork().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def format_outputs(values):
    """Formats the 10 network output values with two decimal places."""
    return " ".join(f"{value:6.2f}" for value in values)


def run_first_ten(model, device, test_dataset):
    """Runs the first 10 MNIST test images through the network."""
    rows = []
    with torch.no_grad():
        for index in range(10):
            image, label = test_dataset[index]
            output = model(image.unsqueeze(0).to(device)).squeeze(0).cpu()
            prediction = output.argmax().item()
            rows.append(
                {
                    "index": index,
                    "outputs": output.tolist(),
                    "prediction": prediction,
                    "label": label,
                }
            )
    return rows


def print_prediction_table(rows):
    """Prints network outputs, predicted digit, and true label for each example."""
    header = "Example | Network outputs for digits 0-9                                      | Pred | Label"
    separator = "-" * len(header)
    print(header)
    print(separator)

    lines = [header, separator]
    for row in rows:
        output_text = format_outputs(row["outputs"])
        line = f"{row['index']:7d} | {output_text} | {row['prediction']:4d} | {row['label']:5d}"
        print(line)
        lines.append(line)
    return lines


def save_prediction_table(lines, output_path):
    """Writes the printed prediction table to a text file for the report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_first_nine_predictions(rows, test_dataset, output_path):
    """Saves a 3x3 plot of the first nine test images with predictions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))

    for plot_index, axis in enumerate(axes.flat):
        image, _label = test_dataset[plot_index]
        prediction = rows[plot_index]["prediction"]
        axis.imshow(image.squeeze(0), cmap="gray")
        axis.set_title(f"Prediction: {prediction}")
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# main function
def main(argv):
    """Loads the saved network, evaluates 10 test examples, and saves report outputs."""
    args = parse_args(argv)
    device = get_device()

    test_dataset = load_test_dataset(args.data_dir, args.download)
    model = load_network(args.model_path, device)
    rows = run_first_ten(model, device, test_dataset)

    table_lines = print_prediction_table(rows)
    save_prediction_table(table_lines, args.output_dir / "test_set_predictions.txt")
    plot_first_nine_predictions(rows, test_dataset, args.output_dir / "plot_predictions.png")

    print(f"Saved printed table to {args.output_dir / 'test_set_predictions.txt'}")
    print(f"Saved prediction plot to {args.output_dir / 'plot_predictions.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
