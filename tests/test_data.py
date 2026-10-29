import unittest
from pathlib import Path

import pandas as pd

from src.data import COLUMNS, clean_grants, load_grants


class DataTests(unittest.TestCase):
    def test_cleaning_preserves_uncertainty_and_source_rows(self):
        rows = [{column: "" for column in COLUMNS} for _ in range(4)]
        rows[0].update({"Program": "  Test\nGrant  ", "Applied?": " YES ",
                        "Deadline": "2026-11-01", "Priority / Deadline Order": 1})
        rows[2].update({"Program": "Rolling grant", "Applied?": "Unknown",
                        "Deadline": " Rolling / call-specific "})
        rows[3].update({"Program": "Partial date", "Applied?": "No",
                        "Deadline": "2026-10 (exact deadline not published)"})
        raw = pd.DataFrame(rows)
        original = raw.copy(deep=True)
        data = clean_grants(raw)
        self.assertEqual(data.program.tolist(), ["Test Grant", "Rolling grant", "Partial date"])
        self.assertEqual(data.source_row.tolist(), [2, 4, 5])
        self.assertTrue(data.loc[0, "applied_bool"])
        self.assertTrue(pd.isna(data.loc[1, "applied_bool"]))
        self.assertFalse(data.loc[2, "applied_bool"])
        self.assertEqual(data.loc[0, "deadline_date"], pd.Timestamp("2026-11-01"))
        self.assertTrue(data.loc[1:, "deadline_date"].isna().all())
        self.assertEqual(data.loc[1, "deadline"], "Rolling / call-specific")
        pd.testing.assert_frame_equal(raw, original)

    def test_missing_columns_are_reported(self):
        with self.assertRaisesRegex(ValueError, "Missing tracker columns"):
            clean_grants(pd.DataFrame({"Program": ["Grant"]}))

    def test_supplied_workbook(self):
        data = load_grants(Path(__file__).resolve().parents[1] / "Ardook_data.xlsx")
        self.assertEqual(len(data), 37)
        self.assertEqual(data.applied_bool.sum(), 4)
        self.assertTrue(data.program.notna().all())
        self.assertEqual(data.deadline_date.notna().sum(), 24)
        self.assertEqual(data.loc[data.official_source.isna(), "source_row"].tolist(), [30, 31, 34])
        self.assertEqual(data.loc[data.applied_bool.isna(), "applied"].tolist(),
                         ["Not interested", "Not interested"])
        self.assertEqual(data.source_row.tolist(), list(range(2, 39)))


if __name__ == "__main__":
    unittest.main()
