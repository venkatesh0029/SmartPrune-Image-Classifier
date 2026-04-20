"""
Train self-pruning models on CIFAR-10 with L = L_cls + lambda * L_sparse.
High-accuracy path uses a prunable Wide-ResNet-style CNN + strong augmentation.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torchvision
import torchvision.transforms as T

from model import SelfPruningMLP, SelfPruningResNet
from train import evaluate, plot_gate_histogram, set_seed, train_epoch
from utils import sparsity_percent


def get_dataloaders(batch_size: int, data_dir: str, *, strong_aug: bool):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    if strong_aug:
        train_tf = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
                T.Normalize(mean, std),
            ]
        )
    else:
        train_tf = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf
    )
    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )
    return train_loader, test_loader


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def run_one(
    *,
    model_name: str,
    lambda_sparse: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    data_dir: str,
    out_dir: str,
    gate_threshold: float,
    save_plot: bool,
    mixup_alpha: float,
    label_smoothing: float,
    warmup_epochs: int,
    eta_min: float,
    width: int,
    max_grad_norm: float,
) -> tuple[float, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    strong_aug = model_name == "cnn"
    train_loader, test_loader = get_dataloaders(batch_size, data_dir, strong_aug=strong_aug)

    if model_name == "mlp":
        model = SelfPruningMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        model = SelfPruningResNet(num_classes=10, width=width, blocks_per_stage=(4, 4, 4)).to(
            device
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )

    log_every = max(1, epochs // 12)

    for epoch in range(1, epochs + 1):
        if epoch <= warmup_epochs:
            wlr = lr * float(epoch) / float(max(1, warmup_epochs))
        else:
            t = (epoch - warmup_epochs - 1) / float(max(1, epochs - warmup_epochs))
            wlr = eta_min + (lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * t))
        _set_lr(optimizer, wlr)

        stats = train_epoch(
            model,
            train_loader,
            optimizer,
            lambda_sparse,
            device,
            mixup_alpha=mixup_alpha if model_name == "cnn" else 0.0,
            label_smoothing=label_smoothing if model_name == "cnn" else 0.0,
            max_grad_norm=max_grad_norm,
            use_amp=use_amp,
        )
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"  epoch {epoch:4d}/{epochs}  lr={wlr:.5f}  "
                f"loss={stats.loss:.4f}  cls={stats.cls_loss:.4f}  "
                f"sparse={stats.sparse_loss:.2f}  train_acc={stats.acc:.2f}%"
            )

    test_acc = evaluate(model, test_loader, device, use_amp=use_amp)
    sparsity = sparsity_percent(model, threshold=gate_threshold)
    print(
        f"  -> test_acc={test_acc:.2f}%  sparsity={sparsity:.1f}% "
        f"(sigma(G)<{gate_threshold})"
    )

    if save_plot:
        os.makedirs(out_dir, exist_ok=True)
        safe_lambda = f"{lambda_sparse:.8f}".replace(".", "p")
        tag = f"{model_name}_w{width}" if model_name == "cnn" else model_name
        plot_path = os.path.join(out_dir, f"gate_hist_{tag}_lambda_{safe_lambda}.png")
        plot_gate_histogram(model, plot_path)
        print(f"  saved gate histogram: {plot_path}")

    return test_acc, sparsity


def main() -> None:
    p = argparse.ArgumentParser(description="Self-pruning network on CIFAR-10")
    p.add_argument(
        "--model",
        choices=("mlp", "cnn"),
        default="cnn",
        help="mlp: baseline MLP; cnn: prunable Wide-ResNet-style CNN (recommended for accuracy)",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--eta-min", type=float, default=1e-5, help="Cosine LR floor after warmup")
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./outputs")
    p.add_argument(
        "--lambda-sparse",
        type=float,
        default=1e-6,
        help="Sparsity strength; use smaller values when maximizing accuracy",
    )
    p.add_argument(
        "--lambda-sweep",
        type=float,
        nargs="*",
        default=None,
        help="If set, run multiple lambda values and print a tradeoff table",
    )
    p.add_argument("--gate-threshold", type=float, default=0.01)
    p.add_argument("--no-plot", action="store_true", help="Skip saving gate histograms")
    p.add_argument("--mixup", type=float, default=0.2, help="Mixup alpha (0 disables); CNN only")
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Cross-entropy label smoothing; CNN only",
    )
    p.add_argument("--width", type=int, default=10, help="Wide-ResNet width multiplier (CNN only)")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir
    out_dir = args.out_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(script_dir, data_dir)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(script_dir, out_dir)

    lambdas = args.lambda_sweep if args.lambda_sweep is not None else [args.lambda_sparse]

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"model={args.model}  epochs={args.epochs}  lambda values: {lambdas}")
    if args.model == "cnn":
        print(
            f"CNN width={args.width}  mixup={args.mixup}  "
            f"label_smoothing={args.label_smoothing}"
        )
    print()

    rows = []
    for lam in lambdas:
        print(f"=== Training with lambda = {lam} ===")
        acc, sp = run_one(
            model_name=args.model,
            lambda_sparse=lam,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr if args.model == "cnn" else 1e-3,
            weight_decay=args.weight_decay if args.model == "cnn" else 0.0,
            seed=args.seed,
            data_dir=data_dir,
            out_dir=out_dir,
            gate_threshold=args.gate_threshold,
            save_plot=not args.no_plot,
            mixup_alpha=args.mixup,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            eta_min=args.eta_min,
            width=args.width,
            max_grad_norm=args.max_grad_norm,
        )
        rows.append((lam, acc, sp))

    if len(rows) > 1:
        print()
        print("Lambda tradeoff (accuracy vs sparsity %):")
        print(f"{'lambda':<14} {'Test Acc %':<12} {'Sparsity %':<12}")
        for lam, acc, sp in rows:
            print(f"{lam:<14.8g} {acc:<12.2f} {sp:<12.1f}")


if __name__ == "__main__":
    main()
    sys.exit(0)
