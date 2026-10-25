"""Statistics used by the reported tables.

Run from the repository root::

    python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_pair import sign_test_p  # noqa: E402


class SignTest(unittest.TestCase):
    def test_reported_corridor_p_value(self) -> None:
        """The block-regime corridor row is 30/30 at p < 1e-4 (2 * 2**-30)."""
        self.assertAlmostEqual(sign_test_p(30, 0), 2 * 2 ** -30, places=12)

    def test_single_loss_is_not_significant(self) -> None:
        """One non-tied seed cannot reach significance (the retired band-regime A2C row)."""
        self.assertAlmostEqual(sign_test_p(0, 1), 1.0)

    def test_unanimous_result_is_bounded(self) -> None:
        """The junction is 30/30, so p = 2 * 2**-30 and never zero."""
        self.assertAlmostEqual(sign_test_p(30, 0), 2 * 2 ** -30, places=12)

    def test_no_observations(self) -> None:
        self.assertEqual(sign_test_p(0, 0), 1.0)

    def test_partial_win_is_not_significant(self) -> None:
        """23 of 30 would be p = 0.0052 - the retired band-regime result, kept as a boundary."""
        self.assertAlmostEqual(sign_test_p(23, 7), 0.0052, places=4)

    def test_symmetric_in_wins_and_losses(self) -> None:
        self.assertAlmostEqual(sign_test_p(18, 12), sign_test_p(12, 18))


if __name__ == "__main__":
    unittest.main()
