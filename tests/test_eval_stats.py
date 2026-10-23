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
        """23 wins and 7 losses is the p quoted in Table 3 (0.0052)."""
        self.assertAlmostEqual(sign_test_p(23, 7), 0.0052, places=4)

    def test_single_loss_is_not_significant(self) -> None:
        """A2C's corridor row is 0 wins, 29 ties, 1 loss -> p = 1.0."""
        self.assertAlmostEqual(sign_test_p(0, 1), 1.0)

    def test_unanimous_result_is_bounded(self) -> None:
        """The junction is 30/30, so p = 2 * 2**-30 and never zero."""
        self.assertAlmostEqual(sign_test_p(30, 0), 2 * 2 ** -30, places=12)

    def test_no_observations(self) -> None:
        self.assertEqual(sign_test_p(0, 0), 1.0)

    def test_symmetric_in_wins_and_losses(self) -> None:
        self.assertAlmostEqual(sign_test_p(18, 12), sign_test_p(12, 18))


if __name__ == "__main__":
    unittest.main()
