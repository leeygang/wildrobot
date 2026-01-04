from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FootSwitchSample:
    # Order: left_toe, left_heel, right_toe, right_heel
    switches: List[bool]


class FootSwitches:
    """4x foot contact switches using Raspberry GPIO via Blinka.

    Config expects Blinka `board` pin names (e.g. "D5").
    """

    ORDER = ("left_toe", "left_heel", "right_toe", "right_heel")

    def __init__(self, pins: Dict[str, str]):
        import board
        import digitalio

        self._dio = {}
        for name in self.ORDER:
            if name not in pins:
                raise ValueError(f"Missing foot switch pin for '{name}'")
            pin_name = pins[name]
            pin = getattr(board, pin_name)
            dio = digitalio.DigitalInOut(pin)
            dio.direction = digitalio.Direction.INPUT
            dio.pull = digitalio.Pull.UP
            self._dio[name] = dio

    def read(self) -> FootSwitchSample:
        # Pull-up input: False means pressed/closed.
        vals = [not self._dio[n].value for n in self.ORDER]
        return FootSwitchSample(switches=vals)

    def close(self) -> None:
        for dio in self._dio.values():
            try:
                dio.deinit()
            except Exception:
                pass
