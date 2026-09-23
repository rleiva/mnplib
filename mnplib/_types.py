"""Shared type aliases for metric and model configuration."""

from typing import Literal

BinSpec = int | Literal["auto", "adaptive"]

XType = Literal["auto", "numeric", "categorical"]
YType = XType

Aggregation = Literal[
    "euclidean",
    "arithmetic",
    "geometric",
    "harmonic",
    "maximum",
    "addition",
    "product",
]

ResolvedTask = Literal["classification", "regression"]
Task = Literal["auto", ResolvedTask]
