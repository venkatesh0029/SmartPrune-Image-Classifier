"""Training and evaluation for self-pruning models on CIFAR-10."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import collect_all_gate_values, sparsity_loss_from_model, sparsity_percent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class EpochStats:
    loss: float
    cls_loss: float
    sparse_loss: float
    acc: float


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_sparse: float,
    device: torch.device,
    *,
    mixup_alpha: float = 0.0,
    label_smoothing: float = 0.0,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
) -> EpochStats:
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_loss = 0.0
    total_cls = 0.0
    total_sparse = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        lam = 1.0
        if mixup_alpha > 0.0:
            lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item())
            idx = torch.randperm(x.size(0), device=device)
            x = lam * x + (1.0 - lam) * x[idx]
            y_a, y_b = y, y[idx]
        else:
            y_a, y_b = y, y

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            if mixup_alpha > 0.0:
                l_cls = lam * ce(logits, y_a) + (1.0 - lam) * ce(logits, y_b)
            else:
                l_cls = ce(logits, y_a)
            l_sp = sparsity_loss_from_model(model)
            loss = l_cls + lambda_sparse * l_sp

        if use_amp:
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_cls += l_cls.item() * x.size(0)
        total_sparse += l_sp.item() * x.size(0)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            if mixup_alpha > 0.0:
                correct += (pred == y_a).sum().item()
            else:
                correct += (pred == y).sum().item()
        n += x.size(0)

    return EpochStats(
        loss=total_loss / n,
        cls_loss=total_cls / n,
        sparse_loss=total_sparse / n,
        acc=100.0 * correct / n,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    model.eval()
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)
    return 100.0 * correct / n


def plot_gate_histogram(
    model: nn.Module,
    out_path: str,
    bins: int = 80,
) -> None:
    import matplotlib.pyplot as plt

    gates = collect_all_gate_values(model).cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.hist(gates, bins=bins, color="#2c5282", edgecolor="white", alpha=0.9)
    plt.xlabel("Gate value sigma(G)")
    plt.ylabel("Count")
    plt.title("Distribution of gate values after training")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
