"""Shared type aliases for metric and model configuration."""

from typing import Literal

BinSpec = int | Literal["auto", "adaptive"]

XType = Literal["auto", "numeric", "categorical"]
YType = XType

ResolvedTask = Literal["classification", "regression"]
Task = Literal["auto", ResolvedTask]
