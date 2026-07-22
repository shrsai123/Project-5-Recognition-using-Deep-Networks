# Shreyas Raman
# Project 5 extension: improved MNIST transformer patch embedding and classifier

# import statements
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mnist_cnn import build_data_loaders, get_device, plot_metric, save_model, train_network


# class definitions
class AdvancedNetTransformer(nn.Module):
    """MNIST transformer with convolutional patch embedding and a fused classifier."""

    def __init__(
        self,
        image_size=28,
        patch_size=7,
        patch_stride=7,
        embed_dim=48,
        num_heads=4,
        num_layers=2,
        mlp_dim=96,
        classifier_dim=128,
        dropout=0.1,
        num_classes=10,
    ):
        """Initializes overlapping patch embedding, transformer encoder, and MLP classifier."""
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim
        self.num_patches_per_side = ((image_size - patch_size) // patch_stride) + 1
        self.num_patches = self.num_patches_per_side**2

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_stride),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.token_norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, classifier_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )

    # converts an image batch into a sequence of overlapping patch tokens
    def image_to_tokens(self, x):
        """Creates overlapping patch tokens with a small convolutional patch embedder."""
        patches = self.patch_embedding(x)
        tokens = patches.flatten(2).transpose(1, 2)
        return self.token_norm(tokens)

    # computes a forward pass for the network
    def forward(self, x):
        """Runs one forward pass and returns log probabilities for 10 digits."""
        tokens = self.image_to_tokens(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.position_embedding
        encoded_tokens = self.transformer_encoder(tokens)

        cls_output = encoded_tokens[:, 0]
        mean_output = encoded_tokens[:, 1:].mean(dim=1)
        fused = torch.cat((cls_output, mean_output), dim=1)
        logits = self.classifier(fused)
        return F.log_softmax(logits, dim=1)


# useful functions with a comment for each function
def parse_args(argv):
    """Parses command line options for advanced transformer training."""
    parser = argparse.ArgumentParser(description="Train an advanced MNIST transformer.")
    parser.add_argument("--batch-size", type=int, default=256, help="training batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000, help="testing batch size")
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/advanced_transformer"), help="output directory")
    parser.add_argument("--model-path", type=Path, default=Path("mnist_transformer_advanced.pt"), help="saved model path")
    parser.add_argument("--no-download", action="store_true", help="use an existing local MNIST dataset")
    parser.add_argument("--patch-size", type=int, default=7, help="patch embedding kernel size")
    parser.add_argument("--patch-stride", type=int, default=7, help="patch embedding stride")
    parser.add_argument("--embed-dim", type=int, default=48, help="token embedding dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="transformer attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="transformer encoder layers")
    parser.add_argument("--mlp-dim", type=int, default=96, help="transformer feedforward dimension")
    parser.add_argument("--classifier-dim", type=int, default=128, help="classifier hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    return parser.parse_args(argv[1:])


def plot_advanced_transformer_diagram(model, output_path):
    """Saves a diagram of the advanced transformer architecture."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = [
        ("Input", "1 x 28 x 28"),
        ("Conv Patch\nEmbedding", f"{model.num_patches} overlapping\npatch tokens"),
        ("LayerNorm\n+ Position", f"{model.num_patches + 1} tokens"),
        ("Transformer", "GELU encoder\nnorm-first"),
        ("CLS + Mean", "fused token"),
        ("LayerNorm\n+ MLP", "classifier"),
        ("LogSoftmax", "10 digits"),
    ]

    fig, axis = plt.subplots(figsize=(12, 3.8))
    axis.set_xlim(0, len(layers))
    axis.set_ylim(0, 1)
    axis.axis("off")
    box_width = 0.78
    colors = ["#d9ead3", "#cfe2f3", "#fff2cc", "#eadcf8"]

    for index, (name, detail) in enumerate(layers):
        x_position = index + 0.08
        rectangle = plt.Rectangle(
            (x_position, 0.28),
            box_width,
            0.44,
            facecolor=colors[index % len(colors)],
            edgecolor="#333333",
            linewidth=1.2,
        )
        axis.add_patch(rectangle)
        axis.text(x_position + box_width / 2, 0.57, name, ha="center", va="center", fontsize=8.5, fontweight="bold")
        axis.text(x_position + box_width / 2, 0.42, detail, ha="center", va="center", fontsize=8)
        if index < len(layers) - 1:
            axis.annotate(
                "",
                xy=(index + 1.04, 0.5),
                xytext=(index + 0.91, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.2},
            )

    axis.set_title("Advanced MNIST Transformer Architecture", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_advanced_transformer_log(model, history, output_path):
    """Writes the advanced transformer printout and training history."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    lines = [
        "Advanced MNIST transformer model:",
        str(model),
        "",
        f"Trainable parameters: {trainable}",
        f"Patch size: {model.patch_size}",
        f"Patch stride: {model.patch_stride}",
        f"Number of patch tokens: {model.num_patches}",
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
    """Trains, plots, and saves the advanced transformer MNIST model."""
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

    model = AdvancedNetTransformer(
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_dim=args.mlp_dim,
        classifier_dim=args.classifier_dim,
        dropout=args.dropout,
    ).to(device)
    print("Advanced transformer model:")
    print(model)

    plot_advanced_transformer_diagram(model, args.output_dir / "advanced_transformer_diagram.png")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    history = train_network(model, device, train_loader, test_loader, optimizer, args.epochs)

    plot_metric(
        history,
        train_key="train_error",
        test_key="test_error",
        ylabel="Error Rate",
        title="Advanced Transformer Training and Testing Error",
        output_path=args.output_dir / "advanced_transformer_training_testing_error.png",
    )
    plot_metric(
        history,
        train_key="train_accuracy",
        test_key="test_accuracy",
        ylabel="Accuracy",
        title="Advanced Transformer Training and Testing Accuracy",
        output_path=args.output_dir / "advanced_transformer_training_testing_accuracy.png",
    )
    write_advanced_transformer_log(model, history, args.output_dir / "advanced_transformer_training_log.txt")
    save_model(model, args.model_path)

    print(f"Saved diagram to {args.output_dir / 'advanced_transformer_diagram.png'}")
    print(f"Saved accuracy plot to {args.output_dir / 'advanced_transformer_training_testing_accuracy.png'}")
    print(f"Saved error plot to {args.output_dir / 'advanced_transformer_training_testing_error.png'}")
    print(f"Saved training log to {args.output_dir / 'advanced_transformer_training_log.txt'}")
    print(f"Saved trained model to {args.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
