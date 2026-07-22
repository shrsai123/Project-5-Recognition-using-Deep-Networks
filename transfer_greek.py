# Shreyas Raman
# Project 5: transfer learning from MNIST digits to Greek letters

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
import torchvision
from PIL import Image

from mnist_cnn import MyNetwork


# class definitions
class GreekTransform:
    """Transforms Greek letter images to match MNIST-style network inputs."""

    def __init__(self):
        """Initializes the transform object."""
        pass

    def __call__(self, x):
        """Converts RGB images to grayscale, scales, crops, and inverts intensities."""
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# useful functions with a comment for each function
def parse_args(argv):
    """Parses command line options for Greek-letter transfer learning."""
    parser = argparse.ArgumentParser(description="Transfer MNIST CNN features to Greek letters.")
    parser.add_argument("--training-set-path", type=Path, default=Path("greek_train"), help="Greek ImageFolder path")
    parser.add_argument("--mnist-model-path", type=Path, default=Path("mnist_cnn.pt"), help="saved MNIST model path")
    parser.add_argument("--greek-model-path", type=Path, default=Path("greek_cnn.pt"), help="saved Greek model path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="output directory")
    parser.add_argument("--epochs", type=int, default=50, help="maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=5, help="Greek training batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for the replacement final layer")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--target-accuracy", type=float, default=0.99, help="early stop accuracy")
    parser.add_argument("--custom-path", type=Path, default=None, help="optional folder of custom Greek images")
    return parser.parse_args(argv[1:])


def get_device():
    """Selects the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_greek_transform():
    """Creates the transform used for the Greek ImageFolder dataset."""
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            GreekTransform(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def validate_training_path(training_set_path):
    """Checks that the Greek training folder has alpha, beta, and gamma subfolders."""
    expected = {"alpha", "beta", "gamma"}
    present = {path.name for path in training_set_path.iterdir() if path.is_dir()} if training_set_path.exists() else set()
    missing = sorted(expected - present)
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"{training_set_path} must contain alpha, beta, and gamma folders; missing {missing_text}"
        )


def build_greek_loader(training_set_path, batch_size):
    """Creates a shuffled DataLoader for the Greek training set."""
    validate_training_path(training_set_path)
    greek_dataset = torchvision.datasets.ImageFolder(training_set_path, transform=build_greek_transform())
    greek_train = torch.utils.data.DataLoader(greek_dataset, batch_size=batch_size, shuffle=True)
    return greek_train, greek_dataset


def build_transfer_model(mnist_model_path, device):
    """Loads MNIST weights, freezes them, and replaces the final layer with 3 outputs."""
    network = MyNetwork().to(device)
    state_dict = torch.load(mnist_model_path, map_location=device, weights_only=True)
    network.load_state_dict(state_dict)

    # freezes the parameters for the whole network
    for param in network.parameters():
        param.requires_grad = False

    network.fc2 = nn.Linear(50, 3).to(device)
    return network


def train_one_epoch(network, device, greek_train, optimizer):
    """Trains the replacement final layer for one epoch."""
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
        prediction = output.argmax(dim=1)
        correct += prediction.eq(target).sum().item()
        total += target.size(0)

    return total_loss / total, correct / total


def evaluate_network(network, device, greek_dataset):
    """Evaluates the Greek model on all training examples in fixed order."""
    loader = torch.utils.data.DataLoader(greek_dataset, batch_size=len(greek_dataset), shuffle=False)
    network.eval()

    with torch.no_grad():
        data, target = next(iter(loader))
        data = data.to(device)
        target = target.to(device)
        output = network(data)
        loss = F.nll_loss(output, target).item()
        prediction = output.argmax(dim=1)
        accuracy = prediction.eq(target).sum().item() / target.size(0)

    return loss, accuracy, prediction.cpu().tolist(), target.cpu().tolist()


def train_greek_network(network, device, greek_train, greek_dataset, epochs, target_accuracy, lr, momentum):
    """Trains the new Greek classifier and records accuracy after each epoch."""
    optimizer = optim.SGD((param for param in network.parameters() if param.requires_grad), lr=lr, momentum=momentum)
    history = {"epoch": [], "train_loss": [], "train_accuracy": [], "eval_loss": [], "eval_accuracy": []}
    reached_epoch = None

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(network, device, greek_train, optimizer)
        eval_loss, eval_accuracy, _prediction, _target = evaluate_network(network, device, greek_dataset)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["eval_loss"].append(eval_loss)
        history["eval_accuracy"].append(eval_accuracy)

        print(
            f"Epoch {epoch}: "
            f"train accuracy={train_accuracy:.4f}, eval accuracy={eval_accuracy:.4f}, "
            f"train loss={train_loss:.4f}, eval loss={eval_loss:.4f}"
        )

        if eval_accuracy >= target_accuracy:
            reached_epoch = epoch
            break

    return history, reached_epoch


def plot_accuracy(history, output_path):
    """Saves a plot of Greek-letter training and fixed-order evaluation accuracy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(history["epoch"], history["train_accuracy"], marker="o", label="Training batches")
    axis.plot(history["epoch"], history["eval_accuracy"], marker="s", label="All 27 examples")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_title("Greek Letter Transfer Learning Accuracy")
    axis.set_xticks(history["epoch"])
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_error(history, output_path):
    """Saves a plot of Greek-letter training and fixed-order evaluation error."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_error = [1.0 - accuracy for accuracy in history["train_accuracy"]]
    eval_error = [1.0 - accuracy for accuracy in history["eval_accuracy"]]

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(history["epoch"], train_error, marker="o", label="Training batches")
    axis.plot(history["epoch"], eval_error, marker="s", label="All 27 examples")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Error")
    axis.set_title("Greek Letter Transfer Learning Error")
    axis.set_xticks(history["epoch"])
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_predictions(network, device, greek_dataset, output_path):
    """Saves a grid of Greek training examples with predicted and true labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_names = greek_dataset.classes
    loader = torch.utils.data.DataLoader(greek_dataset, batch_size=len(greek_dataset), shuffle=False)
    images, targets = next(iter(loader))

    network.eval()
    with torch.no_grad():
        outputs = network(images.to(device))
        predictions = outputs.argmax(dim=1).cpu()

    count = len(greek_dataset)
    columns = 9
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(1.5 * columns, 1.8 * rows))
    axes = list(axes.flat)

    for axis in axes:
        axis.axis("off")

    for index in range(count):
        image = images[index].squeeze(0).numpy()
        predicted_name = class_names[predictions[index].item()]
        actual_name = class_names[targets[index].item()]
        axes[index].imshow(image, cmap="gray")
        axes[index].set_title(f"{predicted_name}\n({actual_name})", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_log(output_path, network, greek_dataset, history, reached_epoch):
    """Writes the model printout and training summary to a report text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if reached_epoch is None:
        summary = f"Target accuracy was not reached in {history['epoch'][-1]} epochs."
    else:
        summary = f"Reached target accuracy at epoch {reached_epoch}."

    lines = [
        "Greek transfer learning model:",
        str(network),
        "",
        f"Class names: {greek_dataset.classes}",
        f"Class to index: {greek_dataset.class_to_idx}",
        f"Number of examples: {len(greek_dataset)}",
        summary,
        "",
        "epoch,train_loss,train_accuracy,eval_loss,eval_accuracy",
    ]

    for index, epoch in enumerate(history["epoch"]):
        lines.append(
            f"{epoch},"
            f"{history['train_loss'][index]:.6f},"
            f"{history['train_accuracy'][index]:.6f},"
            f"{history['eval_loss'][index]:.6f},"
            f"{history['eval_accuracy'][index]:.6f}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_model(network, output_path):
    """Saves the trained Greek classifier weights."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), output_path)


def find_custom_images(custom_path):
    """Finds custom Greek-letter images for optional classification."""
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    class_names = {"alpha", "beta", "gamma"}
    image_paths = []

    for path in custom_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue

        relative_parts = {part.lower() for part in path.relative_to(custom_path).parts}
        if relative_parts & class_names:
            image_paths.append(path)

    return sorted(image_paths)


def infer_expected_label(image_path, custom_path, class_names):
    """Infers the expected class from a parent folder name or filename."""
    relative_parts = [part.lower() for part in image_path.relative_to(custom_path).parts]
    stem = image_path.stem.lower()

    for class_name in class_names:
        if class_name in relative_parts or class_name in stem:
            return class_name

    return ""


def classify_custom_images(network, device, custom_path, output_dir):
    """Classifies optional custom Greek-letter image files and saves a plot and table."""
    image_paths = find_custom_images(custom_path)
    if not image_paths:
        print(f"No custom images found in {custom_path}")
        return

    transform = build_greek_transform()
    class_names = ["alpha", "beta", "gamma"]
    rows = ["file,prediction,expected,correct"]
    processed_images = []
    predictions = []
    expected_labels = []
    correct_flags = []

    network.eval()
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image)
            output = network(tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
            prediction = output.argmax().item()
            prediction_name = class_names[prediction]
            expected_name = infer_expected_label(image_path, custom_path, class_names)
            correct = "" if not expected_name else str(prediction_name == expected_name)

            rows.append(f"{image_path.name},{prediction_name},{expected_name},{correct}")
            processed_images.append(tensor.squeeze(0).numpy())
            predictions.append(prediction_name)
            expected_labels.append(expected_name)
            correct_flags.append(correct)

            if expected_name:
                print(f"{image_path.name}: predicted={prediction_name}, expected={expected_name}, correct={correct}")
            else:
                print(f"{image_path.name}: predicted={prediction_name}")

    table_path = output_dir / "custom_greek_predictions.txt"
    table_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    columns = min(5, len(processed_images))
    grid_rows = math.ceil(len(processed_images) / columns)
    fig, axes = plt.subplots(grid_rows, columns, figsize=(1.8 * columns, 2.0 * grid_rows))
    if len(processed_images) == 1:
        axes = [axes]
    else:
        axes = list(axes.flat)

    for axis in axes:
        axis.axis("off")

    for index, image in enumerate(processed_images):
        axes[index].imshow(image, cmap="gray")
        if expected_labels[index]:
            axes[index].set_title(f"{predictions[index]}\n({expected_labels[index]})", fontsize=9)
        else:
            axes[index].set_title(predictions[index], fontsize=9)

    plot_path = output_dir / "custom_greek_predictions.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    labeled = [flag for flag in correct_flags if flag]
    if labeled:
        correct_count = sum(flag == "True" for flag in labeled)
        print(f"Custom Greek accuracy: {correct_count}/{len(labeled)}")

    print(f"Saved custom Greek table to {table_path}")
    print(f"Saved custom Greek plot to {plot_path}")


# main function
def main(argv):
    """Runs Greek-letter transfer learning from the saved MNIST CNN."""
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    greek_train, greek_dataset = build_greek_loader(args.training_set_path, args.batch_size)
    network = build_transfer_model(args.mnist_model_path, device)

    print("Modified Greek transfer model:")
    print(network)
    print(f"Class names: {greek_dataset.classes}")

    history, reached_epoch = train_greek_network(
        network,
        device,
        greek_train,
        greek_dataset,
        args.epochs,
        args.target_accuracy,
        args.lr,
        args.momentum,
    )

    if reached_epoch is None:
        print(f"Target accuracy was not reached in {history['epoch'][-1]} epochs.")
    else:
        print(f"Reached target accuracy at epoch {reached_epoch}.")

    save_model(network, args.greek_model_path)
    write_log(args.output_dir / "greek_transfer_log.txt", network, greek_dataset, history, reached_epoch)
    plot_accuracy(history, args.output_dir / "greek_training_accuracy.png")
    plot_error(history, args.output_dir / "greek_training_error.png")
    plot_predictions(network, device, greek_dataset, args.output_dir / "greek_predictions.png")

    print(f"Saved Greek model to {args.greek_model_path}")
    print(f"Saved transfer log to {args.output_dir / 'greek_transfer_log.txt'}")
    print(f"Saved accuracy plot to {args.output_dir / 'greek_training_accuracy.png'}")
    print(f"Saved error plot to {args.output_dir / 'greek_training_error.png'}")
    print(f"Saved prediction plot to {args.output_dir / 'greek_predictions.png'}")

    if args.custom_path is not None:
        classify_custom_images(network, device, args.custom_path, args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
