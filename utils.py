"""Sparsity metrics and gate aggregation for evaluation and plots."""

from __future__ import annotations

import torch

from layers import PrunableConv2d, PrunableLinear


def _prunable_modules(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, (PrunableLinear, PrunableConv2d)):
            yield m


def sparsity_loss_from_model(model: torch.nn.Module) -> torch.Tensor:
    """L_sparse = sum of sigma(G) over all prunable gates."""
    total = None
    for m in _prunable_modules(model):
        g = torch.sigmoid(m.gate_scores)
        s = g.sum()
        total = s if total is None else total + s
    if total is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return total


def collect_all_gate_values(model: torch.nn.Module) -> torch.Tensor:
    """1D tensor of all sigma(G) values across prunable layers."""
    parts = []
    for m in _prunable_modules(model):
        parts.append(m.gate_values().reshape(-1))
    if not parts:
        return torch.tensor([])
    return torch.cat(parts, dim=0)


def sparsity_percent(
    model: torch.nn.Module,
    threshold: float = 0.01,
) -> float:
    """Fraction of gates with sigma(G) < threshold, as percentage."""
    gates = collect_all_gate_values(model)
    if gates.numel() == 0:
        return 0.0
    pruned = (gates < threshold).sum().item()
    return 100.0 * pruned / gates.numel()
