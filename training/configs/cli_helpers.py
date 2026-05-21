"""Small CLI helpers shared by training/eval/scripts entrypoints.

Single responsibility: present a uniform fail-fast experience when a
default config path doesn't exist on disk, instead of letting each
CLI throw a raw ``FileNotFoundError`` traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path


def fail_if_config_missing(
    config_path: str | Path,
    *,
    user_passed_explicit: bool = False,
    extra_hint: str | None = None,
) -> None:
    """Exit non-zero with a clear message if ``config_path`` doesn't exist.

    ``user_passed_explicit=True`` is accepted for callers that want to
    distinguish "user typed the path" from "we fell back to a default",
    but otherwise has no effect on the message.
    """
    del user_passed_explicit  # accepted for compatibility, currently unused
    p = Path(config_path)
    if p.exists():
        return
    print(f"ERROR: Config file not found: {p}", file=sys.stderr)
    if extra_hint:
        print(f"  {extra_hint}", file=sys.stderr)
    sys.exit(2)
