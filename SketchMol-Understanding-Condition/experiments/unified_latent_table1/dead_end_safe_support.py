#!/usr/bin/env python3
"""Keep sampling alive when B40 dynamic support hits a dead particle."""

from __future__ import annotations

import torch


def batch_slice(source: dict, index: int) -> dict:
    size = int(source["atomic_number"].shape[0])
    return {
        key: (
            value[index : index + 1]
            if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == size
            else value
        )
        for key, value in source.items()
    }


def dead_end_diagnostics(count: int, legal: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    ones = torch.ones(count, dtype=torch.long, device=device)
    return {
        "base_legal": ones,
        "constrained_legal": legal.sum(dim=1).long(),
        "stop_masked": ~legal[:, 0],
    }


class DeadEndSafeSupport:
    """Force STOP on particles with no legal event so the rest of the batch continues."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.dead_end_calls = 0
        self.dead_end_particles = 0

    def __call__(
        self,
        field,
        source,
        node_actions: torch.Tensor,
        edge_actions: torch.Tensor,
        working: torch.Tensor,
        support,
        support_tensors,
    ):
        try:
            legal, diagnostics = self.inner(
                field,
                source,
                node_actions,
                edge_actions,
                working,
                support,
                support_tensors,
            )
        except RuntimeError as exc:
            if "dead end" not in str(exc):
                raise
            legal, diagnostics = self._recover(
                field,
                source,
                node_actions,
                edge_actions,
                working,
                support,
                support_tensors,
            )
        dead = ~legal.any(dim=1)
        if bool(dead.any()):
            legal = legal.clone()
            legal[dead, 0] = True
            self.dead_end_calls += 1
            self.dead_end_particles += int(dead.sum().cpu())
            diagnostics = dict(diagnostics)
            diagnostics["stop_masked"] = ~legal[:, 0]
        return legal, diagnostics

    def _recover(
        self,
        field,
        source,
        node_actions,
        edge_actions,
        working,
        support,
        support_tensors,
    ):
        count = int(node_actions.shape[0])
        n_events = int(field.layout.total_events)
        legal = torch.zeros(count, n_events, dtype=torch.bool, device=node_actions.device)
        self.dead_end_calls += 1
        for index in range(count):
            try:
                one, _ = self.inner(
                    field,
                    batch_slice(source, index),
                    node_actions[index : index + 1],
                    edge_actions[index : index + 1],
                    working[index : index + 1],
                    support,
                    support_tensors,
                )
                legal[index] = one[0]
            except RuntimeError as exc:
                if "dead end" not in str(exc):
                    raise
                legal[index, 0] = True
                self.dead_end_particles += 1
        return legal, dead_end_diagnostics(count, legal, node_actions.device)

    def manifest(self) -> dict[str, object]:
        inner = self.inner.manifest() if hasattr(self.inner, "manifest") else {}
        return {
            **inner,
            "dead_end_calls": self.dead_end_calls,
            "dead_end_particles": self.dead_end_particles,
        }
