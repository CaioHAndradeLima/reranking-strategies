import json
import unittest
from pathlib import Path

from reranking_strategies.judge_calibration import calibrate_judge


ROOT = Path(__file__).resolve().parents[1]


class JudgeCalibrationTests(unittest.TestCase):
    def test_synthetic_demo_is_explicit_and_uses_a_held_out_split(self):
        payload = json.loads((ROOT / "data/judge_calibration_demo.json").read_text())
        report = calibrate_judge(payload)

        self.assertEqual(report["label_source"], "synthetic_demo")
        self.assertIn("not evidence", report["interpretation"])
        for dimension in report["dimensions"].values():
            raw = dimension["validation_raw"]
            calibrated = dimension["validation_calibrated"]
            self.assertEqual(raw["sample_count"], 10)
            self.assertLess(calibrated["mae"], raw["mae"])
            mapping = list(dimension["score_mapping_0_to_4"].values())
            self.assertEqual(mapping, sorted(mapping))

    def test_human_source_is_reported_as_human_validation(self):
        payload = json.loads((ROOT / "data/judge_calibration_demo.json").read_text())
        payload["label_source"] = "human"
        report = calibrate_judge(payload)

        self.assertEqual(report["label_source"], "human")
        self.assertIn("supplied held-out human ratings", report["interpretation"])

    def test_rejects_invalid_scores_and_missing_validation(self):
        payload = json.loads((ROOT / "data/judge_calibration_demo.json").read_text())
        payload["records"][0]["human_scores"] = [5]
        with self.assertRaisesRegex(ValueError, "human_scores"):
            calibrate_judge(payload)

        payload = json.loads((ROOT / "data/judge_calibration_demo.json").read_text())
        payload["records"] = [
            row for row in payload["records"]
            if not (row["dimension"] == "correctness" and row["split"] == "validation")
        ]
        with self.assertRaisesRegex(ValueError, "at least 5 calibration and 2 validation"):
            calibrate_judge(payload)

    def test_prevents_the_same_answer_from_crossing_splits(self):
        payload = json.loads((ROOT / "data/judge_calibration_demo.json").read_text())
        validation = next(row for row in payload["records"] if row["split"] == "validation")
        paired_dimension = next(
            row for row in payload["records"]
            if row["sample_id"] != validation["sample_id"]
            and row["dimension"] != validation["dimension"]
            and row["split"] == "calibration"
        )
        paired_dimension["sample_id"] = validation["sample_id"]
        with self.assertRaisesRegex(ValueError, "appears in both"):
            calibrate_judge(payload)


if __name__ == "__main__":
    unittest.main()
