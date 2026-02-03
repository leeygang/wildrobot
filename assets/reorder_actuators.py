#!/usr/bin/env python3
"""Reorder MJCF <actuator> entries to a canonical order.

Usage:
  python3 assets/reorder_actuators.py --xml assets/v2/wildrobot.xml --order assets/v2/actuator_order.txt
  python3 assets/reorder_actuators.py --xml assets/v2/wildrobot.xml --order assets/v2/actuator_order.txt --check
  python3 assets/reorder_actuators.py --xml assets/v2/wildrobot.xml --order assets/v2/actuator_order.txt --out /tmp/wildrobot.xml
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Sequence


def _read_order(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    order: List[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        order.append(stripped)
    if not order:
        raise ValueError(f"Actuator order file is empty: {path}")
    return order


def _actuator_children(actuator_elem: ET.Element) -> List[ET.Element]:
    return [child for child in list(actuator_elem) if child.tag is not None]


def _index_by_name(children: Iterable[ET.Element]) -> dict[str, ET.Element]:
    mapping: dict[str, ET.Element] = {}
    for child in children:
        name = child.get("name")
        if not name:
            raise ValueError(f"Actuator element missing name attribute: {ET.tostring(child, encoding='unicode')}")
        if name in mapping:
            raise ValueError(f"Duplicate actuator name in MJCF: {name}")
        mapping[name] = child
    return mapping


def _format_mismatch(missing: Sequence[str], extra: Sequence[str]) -> str:
    parts = []
    if missing:
        parts.append("missing: " + ", ".join(missing))
    if extra:
        parts.append("extra: " + ", ".join(extra))
    return "; ".join(parts) if parts else ""


def reorder_actuators(
    *,
    xml_path: Path,
    order_path: Path,
    out_path: Path | None = None,
    check_only: bool = False,
) -> None:
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)
    if not order_path.exists():
        raise FileNotFoundError(order_path)

    canonical_order = _read_order(order_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    actuator_elem = root.find("actuator")
    if actuator_elem is None:
        raise ValueError(f"<actuator> element not found in {xml_path}")

    children = _actuator_children(actuator_elem)
    mapping = _index_by_name(children)

    missing = [name for name in canonical_order if name not in mapping]
    extra = [name for name in mapping.keys() if name not in canonical_order]
    if missing or extra:
        raise ValueError(
            f"Actuator list mismatch for {xml_path} vs {order_path}: "
            + _format_mismatch(missing, extra)
        )

    current_order = [child.get("name") or "" for child in children]
    if current_order == canonical_order:
        if check_only:
            print("Actuator order OK")
        return

    if check_only:
        raise SystemExit(
            "Actuator order mismatch:\n"
            f"  current:   {', '.join(current_order)}\n"
            f"  canonical: {', '.join(canonical_order)}"
        )

    for child in children:
        actuator_elem.remove(child)

    for name in canonical_order:
        actuator_elem.append(mapping[name])

    ET.indent(tree, space="  ", level=0)
    out_target = out_path if out_path is not None else xml_path
    tree.write(out_target, encoding="utf-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reorder MJCF actuators to a canonical order.")
    parser.add_argument("--xml", required=True, type=Path, help="Path to MJCF XML file")
    parser.add_argument("--order", required=True, type=Path, help="Path to actuator_order.txt")
    parser.add_argument("--out", type=Path, default=None, help="Optional output path (defaults to in-place)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate ordering and exit nonzero if mismatch",
    )
    args = parser.parse_args()

    reorder_actuators(
        xml_path=args.xml,
        order_path=args.order,
        out_path=args.out,
        check_only=bool(args.check),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
