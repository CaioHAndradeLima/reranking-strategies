"""Create clearly synthetic paired scores for exercising judge calibration."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/judge_calibration_demo.json"


def bounded_score(value: int) -> int:
    return min(4, max(0, value))


def main() -> None:
    latent_by_dimension = {
        dimension: [score for score in range(5) for _ in range(6)]
        for dimension in ("correctness", "groundedness", "assertiveness_calibration")
    }
    judge_bias = {"correctness": 1, "groundedness": 1, "assertiveness_calibration": -1}
    records = []
    for dimension, latent_scores in latent_by_dimension.items():
        for index, latent in enumerate(latent_scores):
            # The fictional judge has a known one-point bias in each dimension.
            judge_score = bounded_score(latent + judge_bias[dimension])
            # Fictional raters differ in severity but center on the latent score.
            human_scores = [bounded_score(latent + severity) for severity in (-1, 0, 1)]
            records.append({
                "sample_id": f"demo-{dimension}-{index + 1:02}",
                "dimension": dimension,
                "split": "calibration" if index % 6 < 4 else "validation",
                "judge_score": judge_score,
                "human_scores": human_scores,
            })

    OUTPUT.write_text(json.dumps({
        "label_source": "synthetic_demo",
        "simulation": {
            "description": (
                "Deterministic fictional scores use predefined latent quality values, "
                "a known judge bias, and three rater-severity profiles. "
                "No answers were reviewed by people; do not interpret the metrics "
                "as evidence of real judge-human agreement."
            ),
        },
        "records": records,
    }, indent=2) + "\n")
    print(f"Wrote {len(records)} synthetic records to {OUTPUT}")


if __name__ == "__main__":
    main()
