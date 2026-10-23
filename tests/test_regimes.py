"""The disruption regime's geometry and its guards.

The band size is the number the report quotes (101 cells) and the value the
station-access probe ships, so this test fails if either drifts.

Run from the repository root::

    python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import disruptions  # noqa: E402
import regimes  # noqa: E402


class CompactBand(unittest.TestCase):
    def test_band_size_matches_the_report(self) -> None:
        self.assertEqual(len(regimes.compact_fire_band()), 101)

    def test_band_never_covers_a_platform(self) -> None:
        platforms, _commit = regimes.compact_geometry()
        self.assertTrue(platforms)
        self.assertFalse(regimes.compact_fire_band() & platforms)

    def test_band_is_inside_the_grid(self) -> None:
        for row, col in regimes.compact_fire_band():
            self.assertTrue(0 <= row < 8, (row, col))
            self.assertTrue(0 <= col < 87, (row, col))

    def test_scenario_detection(self) -> None:
        self.assertTrue(regimes.is_compact("network-simple-compact"))
        self.assertTrue(regimes.is_s2("granville-harris-park"))
        self.assertFalse(regimes.is_compact("granville-harris-park"))


class Guards(unittest.TestCase):
    def test_unknown_scenario_raises(self) -> None:
        with self.assertRaises(ValueError):
            regimes.install(object(), "no-such-map", 3)

    def test_install_requires_a_zone(self) -> None:
        with self.assertRaises(ValueError):
            disruptions.install(object())


if __name__ == "__main__":
    unittest.main()
