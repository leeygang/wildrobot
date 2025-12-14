#!/usr/bin/env python3
"""Scan the workspace for `mjx`/`mujoco` usage to assist deprecation."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = [re.compile(r"\bmujoco\.mjx\b"), re.compile(r"\bmjx\b"), re.compile(r"\bmujoco\b")]

def scan(root: Path):
    hits = []
    for p in root.rglob("*.py"):
        try:
            txt = p.read_text()
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), start=1):
            for pat in PATTERNS:
                if pat.search(line):
                    hits.append((str(p.relative_to(root)), i, line.strip()))
    return hits

def main():
    hits = scan(ROOT)
    if not hits:
        print("No mjx/mujoco usages found.")
        return
    print("Found mjx/mujoco usages (file, line, snippet):")
    for f, ln, l in hits:
        print(f"{f}:{ln}: {l}")

if __name__ == '__main__':
    main()
