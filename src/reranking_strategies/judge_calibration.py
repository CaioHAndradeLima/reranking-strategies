"""Compare ordinal LLM-judge scores with reviewer ratings and calibrate score bias."""

from collections import defaultdict
from statistics import mean
from typing import Any


DIMENSIONS = ("correctness", "groundedness", "assertiveness_calibration")
SPLITS = ("calibration", "validation")


def _validate_records(payload: Any) -> tuple[str, list[dict]]:
    if not isinstance(payload, dict):
        raise ValueError("Ratings file must be an object with label_source and records")
    label_source = payload.get("label_source")
    if label_source not in {"human", "synthetic_demo"}:
        raise ValueError("label_source must be 'human' or 'synthetic_demo'")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty array")

    seen = set()
    sample_splits = {}
    for index, row in enumerate(records, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} must be an object")
        sample_id = row.get("sample_id")
        dimension = row.get("dimension")
        split = row.get("split")
        judge_score = row.get("judge_score")
        human_scores = row.get("human_scores")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"Record {index} requires a non-empty sample_id")
        if dimension not in DIMENSIONS:
            raise ValueError(f"Record {index} dimension must be one of {', '.join(DIMENSIONS)}")
        if split not in SPLITS:
            raise ValueError(f"Record {index} split must be 'calibration' or 'validation'")
        if sample_id in sample_splits and sample_splits[sample_id] != split:
            raise ValueError(f"Sample '{sample_id}' appears in both calibration and validation")
        sample_splits[sample_id] = split
        if type(judge_score) is not int or not 0 <= judge_score <= 4:
            raise ValueError(f"Record {index} judge_score must be an integer from 0 to 4")
        if (not isinstance(human_scores, list) or not human_scores
                or any(type(score) is not int or not 0 <= score <= 4 for score in human_scores)):
            raise ValueError(f"Record {index} human_scores must be non-empty integer scores from 0 to 4")
        key = (sample_id, dimension)
        if key in seen:
            raise ValueError(f"Duplicate sample_id and dimension in record {index}")
        seen.add(key)

    return label_source, records


def _fit_monotone_mapping(records: list[dict]) -> dict[int, float]:
    """Fit a 0..4 isotonic lookup using pool-adjacent-violators on reviewer means."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in records:
        buckets[row["judge_score"]].append(mean(row["human_scores"]))
    observed = sorted(buckets)
    if len(observed) < 2:
        raise ValueError("Each dimension needs at least two distinct judge scores in calibration")

    # Each block stores [first score, last score, total human score, row count].
    blocks: list[list[float]] = []
    for score in observed:
        values = buckets[score]
        blocks.append([score, score, sum(values), len(values)])
        while (len(blocks) >= 2
               and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]):
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])

    observed_values: dict[int, float] = {}
    for start, end, total, count in blocks:
        for score in range(int(start), int(end) + 1):
            observed_values[score] = min(4.0, max(0.0, total / count))

    # Fill unseen judge-score buckets by linear interpolation between observed buckets.
    mapping: dict[int, float] = {}
    keys = sorted(observed_values)
    for score in range(5):
        if score in observed_values:
            mapping[score] = observed_values[score]
            continue
        lower = [key for key in keys if key < score]
        upper = [key for key in keys if key > score]
        if not lower:
            mapping[score] = observed_values[upper[0]]
        elif not upper:
            mapping[score] = observed_values[lower[-1]]
        else:
            lo, hi = lower[-1], upper[0]
            fraction = (score - lo) / (hi - lo)
            mapping[score] = observed_values[lo] * (1 - fraction) + observed_values[hi] * fraction
    return mapping


def _score_metrics(rows: list[dict], mapping: dict[int, float] | None = None) -> dict:
    errors = []
    within_one = []
    exact = []
    biases = []
    for row in rows:
        human = mean(row["human_scores"])
        predicted = float(row["judge_score"]) if mapping is None else mapping[row["judge_score"]]
        error = predicted - human
        errors.append(abs(error))
        within_one.append(abs(error) <= 1.0)
        exact.append(abs(error) < 1e-9)
        biases.append(error)
    return {
        "sample_count": len(rows),
        "mae": mean(errors),
        "exact_agreement": mean(exact),
        "within_one_point": mean(within_one),
        "mean_bias_judge_minus_human": mean(biases),
    }


def _human_consistency(rows: list[dict]) -> dict:
    per_sample = []
    pairwise_within_one = []
    for row in rows:
        scores = row["human_scores"]
        if len(scores) < 2:
            continue
        per_sample.append(mean(abs(left - right) for i, left in enumerate(scores)
                               for right in scores[i + 1:]))
        pairwise_within_one.append(mean(abs(left - right) <= 1 for i, left in enumerate(scores)
                                        for right in scores[i + 1:]))
    if not per_sample:
        return {"annotated_sample_count": 0, "pairwise_mae": None, "within_one_point": None}
    return {
        "annotated_sample_count": len(per_sample),
        "pairwise_mae": mean(per_sample),
        "within_one_point": mean(pairwise_within_one),
    }


def calibrate_judge(payload: dict) -> dict:
    """Fit calibration-only mappings and evaluate raw/calibrated scores on held-out labels."""
    label_source, rows = _validate_records(payload)
    results = {}
    for dimension in DIMENSIONS:
        dimension_rows = [row for row in rows if row["dimension"] == dimension]
        calibration = [row for row in dimension_rows if row["split"] == "calibration"]
        validation = [row for row in dimension_rows if row["split"] == "validation"]
        if len(calibration) < 5 or len(validation) < 2:
            raise ValueError(
                f"Dimension '{dimension}' requires at least 5 calibration and 2 validation records"
            )
        mapping = _fit_monotone_mapping(calibration)
        results[dimension] = {
            "calibration_sample_count": len(calibration),
            "validation_raw": _score_metrics(validation),
            "validation_calibrated": _score_metrics(validation, mapping),
            "human_rater_consistency_validation": _human_consistency(validation),
            "score_mapping_0_to_4": {str(score): round(mapping[score], 3) for score in range(5)},
        }
    return {
        "label_source": label_source,
        "interpretation": (
            "Synthetic demonstration only; this is not evidence of human agreement."
            if label_source == "synthetic_demo"
            else "Calibration is evaluated against the supplied held-out human ratings."
        ),
        "method": "Per-dimension isotonic regression fitted on calibration rows; evaluated on validation rows.",
        "dimensions": results,
    }
