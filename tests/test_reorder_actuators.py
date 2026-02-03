from __future__ import annotations

from pathlib import Path

import pytest

from assets.reorder_actuators import reorder_actuators


def _write_xml(path: Path) -> None:
    content = """
<mujoco>
  <actuator>
    <position name="b" joint="b"/>
    <position name="a" joint="a"/>
    <position name="c" joint="c"/>
  </actuator>
</mujoco>
""".strip()
    path.write_text(content)


def test_reorder_actuators_in_place(tmp_path: Path) -> None:
    xml_path = tmp_path / "model.xml"
    order_path = tmp_path / "order.txt"
    _write_xml(xml_path)
    order_path.write_text("a\nb\nc\n")

    reorder_actuators(xml_path=xml_path, order_path=order_path)

    text = xml_path.read_text()
    assert text.index('name="a"') < text.index('name="b"') < text.index('name="c"')


def test_reorder_actuators_check_mode(tmp_path: Path) -> None:
    xml_path = tmp_path / "model.xml"
    order_path = tmp_path / "order.txt"
    _write_xml(xml_path)
    order_path.write_text("a\nb\nc\n")

    with pytest.raises(SystemExit):
        reorder_actuators(xml_path=xml_path, order_path=order_path, check_only=True)


def test_reorder_actuators_mismatch(tmp_path: Path) -> None:
    xml_path = tmp_path / "model.xml"
    order_path = tmp_path / "order.txt"
    _write_xml(xml_path)
    order_path.write_text("a\nb\n")

    with pytest.raises(ValueError):
        reorder_actuators(xml_path=xml_path, order_path=order_path)
