"""The disruption regime's geometry and its guards.

The band size is the value the shipped regime and the station-access probe
produce (101 cells), so this test fails if either drifts.

Run from the repository root::

    python -m unittest discover -s tests
"""

from __future__ import annotations

import os
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


class VariantSelection(unittest.TestCase):
    """The reported regime must be what a bare call selects."""

    def setUp(self) -> None:
        self._saved = os.environ.get("RL_A3_REGIME")
        os.environ.pop("RL_A3_REGIME", None)

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop("RL_A3_REGIME", None)
        else:
            os.environ["RL_A3_REGIME"] = self._saved

    def test_default_is_the_reported_block_regime(self) -> None:
        self.assertEqual(regimes.selected_variant(), "block")

    def test_env_var_overrides_the_default(self) -> None:
        os.environ["RL_A3_REGIME"] = "band"
        self.assertEqual(regimes.selected_variant(), "band")

    def test_explicit_argument_beats_the_env_var(self) -> None:
        os.environ["RL_A3_REGIME"] = "band"
        self.assertEqual(regimes.selected_variant("crossing"), "crossing")

    def test_clean_variant_installs_nothing(self) -> None:
        for name in ("none", "clean"):
            self.assertEqual(regimes.install(None, "network-simple-compact", 2, variant=name), {})


class Guards(unittest.TestCase):
    def test_unknown_scenario_raises(self) -> None:
        with self.assertRaises(ValueError):
            regimes.install(object(), "no-such-map", 3)

    def test_install_requires_a_zone(self) -> None:
        with self.assertRaises(ValueError):
            disruptions.install(object())


if __name__ == "__main__":
    unittest.main()
