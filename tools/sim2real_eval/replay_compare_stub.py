#!/usr/bin/env python3
"""Milestone 0 placeholder for sim-vs-real replay comparison."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder replay comparison entrypoint")
    parser.add_argument("--sim-log", type=str, required=True)
    parser.add_argument("--real-log", type=str, required=True)
    args = parser.parse_args()
    print(
        "Milestone 0 stub: comparison logic deferred to v0.19.1+. "
        f"sim-log={args.sim_log} real-log={args.real_log}"
    )


if __name__ == "__main__":
    main()
