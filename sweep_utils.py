from __future__ import annotations

import json
import logging
import os
import random
from typing import Iterable, Sequence

import torch

logger = logging.getLogger(__name__)


def _to_float_list(value, name: str) -> list[float]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return [float(x) for x in value.detach().cpu().flatten().tolist()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.isdigit():
            count = int(text)
            if count <= 0:
                return []
            return [float(i + 1) for i in range(count)]
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name} is not valid JSON list: {exc}") from exc
            if not isinstance(parsed, list):
                raise ValueError(f"{name} JSON must be a list of numbers.")
            return [float(x) for x in parsed]
        # fallback: split by comma/space
        parts = [p for p in text.replace(",", " ").split() if p]
        return [float(p) for p in parts]
    raise ValueError(f"{name} must be a list of floats, tensor, or string, got {type(value).__name__}.")


def to_float_list(value, name: str) -> list[float]:
    return _to_float_list(value, name)


def build_clocked_sweep(clock: Sequence[float], values: Sequence[float]) -> list[float]:
    n = len(clock)
    m = len(values)
    if n <= 0:
        raise ValueError("clock must have at least one entry.")
    if m <= 0:
        raise ValueError("values must have at least one entry.")
    if m > n:
        raise ValueError("values length cannot exceed clock length.")

    base = n // m
    remainder = n % m
    output: list[float] = []
    for idx, val in enumerate(values):
        count = base + (1 if idx < remainder else 0)
        output.extend([float(val)] * count)
    if len(output) != n:
        logger.warning("Clocked sweep length mismatch: expected %d, got %d", n, len(output))
    return output


def parse_clock_and_values(clock_input, values_input) -> tuple[list[float], list[float]]:
    clock = _to_float_list(clock_input, "clock")
    values = _to_float_list(values_input, "values")
    if not clock and values:
        clock = [float(i + 1) for i in range(len(values))]
    return clock, values


def build_combinations(values_lists: Sequence[Sequence[float]]) -> list[list[float]]:
    lengths = [len(v) for v in values_lists]
    if not lengths:
        raise ValueError("At least one values list is required.")
    if any(l <= 0 for l in lengths):
        raise ValueError("All provided values lists must be non-empty.")
    total = 1
    for l in lengths:
        total *= l
    outputs: list[list[float]] = []
    for values, length in zip(values_lists, lengths):
        outputs.append([float(values[i % length]) for i in range(total)])
    return outputs


def generate_seed_batch(
    seed: int,
    count: int,
    *,
    min_value: int = 0,
    max_value: int = 0xFFFFFFFFFFFFFFFF,
) -> list[int]:
    n = int(count)
    if n <= 0:
        raise ValueError("count must be > 0.")
    lo = int(min_value)
    hi = int(max_value)
    if lo < 0:
        raise ValueError("min_value must be >= 0.")
    if hi < lo:
        raise ValueError("max_value must be >= min_value.")

    rng = random.Random(int(seed))
    span = hi - lo + 1
    return [lo + rng.randrange(span) for _ in range(n)]


def normalize_comet_experiment_key(
    value: str | None,
    *,
    min_len: int = 30,
    max_len: int = 50,
) -> str:
    if min_len <= 0:
        raise ValueError("min_len must be > 0.")
    if max_len < min_len:
        raise ValueError("max_len must be >= min_len.")

    text = "" if value is None else str(value)
    key = "".join(ch for ch in text if ch.isalnum())
    if not key:
        key = f"exp{random.getrandbits(128):032x}"
    if len(key) < min_len:
        key = key + ("X" * (min_len - len(key)))
    if len(key) > max_len:
        key = key[:max_len]
    return key


def load_prompt_list_from_json(json_path: str) -> list[str]:
    path = str(json_path or "").strip()
    if not path:
        raise ValueError("json_path is required.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array of strings in {path}.")

    prompts: list[str] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, str):
            raise ValueError(f"Expected string at index {idx} in {path}, got {type(item).__name__}.")
        prompts.append(item)
    return prompts
