#!/usr/bin/env python3
"""Milestone 0 placeholder entrypoint for SysID data capture."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder SysID capture entrypoint")
    parser.add_argument("--output-dir", type=str, default="runtime/logs/sysid")
    args = parser.parse_args()
    print(
        "Milestone 0 stub: no SysID capture pipeline yet. "
        f"Planned output dir: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
