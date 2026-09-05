#!/usr/bin/env python3
"""Verify actual argparse/config resolution for every global-64 candidate."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main import get_args_parser
from ufo.utils.config import merge_config_and_args


CANDIDATES = ((8, 1), (4, 2), (2, 4), (1, 8))


def resolve(config, batch, accumulation):
    return merge_config_and_args(
        get_args_parser(),
        config_path=str(config),
        cli_args=[
            "--config",
            str(config),
            "--batch_size",
            str(batch),
            "--gradient_accumulation_steps",
            str(accumulation),
        ],
    )


def main():
    benchmark = Path("configs/h200/ufo_r9_h200_benchmark.json")
    full = Path("configs/h200/ufo_r9_waymo_full_global64.json")
    for batch, accumulation in CANDIDATES:
        args = resolve(benchmark, batch, accumulation)
        if (args.batch_size, args.gradient_accumulation_steps) != (
            batch,
            accumulation,
        ):
            raise RuntimeError(
                f"benchmark b{batch}a{accumulation} resolved to "
                f"b{args.batch_size}a{args.gradient_accumulation_steps}"
            )
        if 8 * args.batch_size * args.gradient_accumulation_steps != 64:
            raise RuntimeError("benchmark candidate is not global batch 64")
        print(
            f"BENCHMARK_MERGE b{batch}a{accumulation} -> "
            f"batch={args.batch_size} accumulation={args.gradient_accumulation_steps} "
            "global=64"
        )

    full_b8 = resolve(full, 8, 1)
    if (full_b8.batch_size, full_b8.gradient_accumulation_steps) != (8, 1):
        raise RuntimeError(
            "full config/recommended b8a1 did not resolve to batch=8 accumulation=1"
        )
    print("FULL_MERGE b8a1 -> batch=8 accumulation=1 global=64")
    print("R9_CONFIG_MERGE_PASS")


if __name__ == "__main__":
    main()
