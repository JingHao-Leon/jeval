"""Exported routing rules must name the scale the cost sweep actually used."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from jeval import presets
from jeval.cli import app, deployed_threshold
from jeval.costs import CostAction, evaluate_point, sweep, write_thresholds_yaml
from jeval.ingest import ingest_preset_files
from jeval.schema import normalize_record
from jeval.store import read_records, write_records

COMPARISON = {
    "field": "stated_probability",
    "operator": ">=",
    "by_question_type": {
        "choice": "prediction_probability",
        "noul": "max_p_yes_no",
        "score": "reported_confidence",
    },
}
STAMP = "2026-10-13T00:00:00Z"


@pytest.mark.parametrize(
    ("question_type", "payload", "expected"),
    [
        ("choice", {"prediction": "a", "probabilities": {"a": 0.6, "b": 0.4}}, 0.6),
        # The supplied prediction need not be the argmax. Name P(prediction), not max(P).
        ("choice", {"prediction": "b", "probabilities": {"a": 0.6, "b": 0.4}}, 0.4),
        ("noul", {"probability_positive": 0.9}, 0.9),
        ("noul", {"probability_positive": 0.1}, 0.9),
        (
            "score",
            {"prediction": "2", "confidence": 0.4, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
            0.4,
        ),
    ],
)
def test_export_comparison_matches_each_record_type(
    tmp_path: Path, question_type: str, payload: dict[str, Any], expected: float
) -> None:
    record = normalize_record(
        {"model": "m", "question_key": "q", "question_type": question_type, **payload}
    )
    record = record.model_copy(update={"label": record.prediction, "label_source": "human_review"})
    action = CostAction("auto", "q", record.prediction, 100.0, 1.0, 0.0)
    result = sweep(action, [record] * 30, n_boot=0)
    target = tmp_path / "thresholds.yaml"
    write_thresholds_yaml([result], target, generated_at=STAMP)
    entry = yaml.safe_load(target.read_text())["actions"]["auto"]

    assert entry["threshold"] == pytest.approx(expected)
    assert record.stated_probability == pytest.approx(expected)
    assert entry["comparison"] == COMPARISON
    assert evaluate_point(action, [record], expected).n_auto == 1
    assert evaluate_point(action, [record], expected + 0.01).n_auto == 0
    if question_type == "score":
        assert record.confidence == 0.4  # the reported spread, not the largest level probability
        assert record.calibration_point() is None


@pytest.mark.parametrize("n_records", [1, 30])
def test_threshold_cli_explains_the_comparison_field(tmp_path: Path, n_records: int) -> None:
    record = normalize_record(
        {
            "model": "m",
            "question_key": "q",
            "question_type": "choice",
            "prediction": "a",
            "confidence": 0.6,
            "label": "a",
            "label_source": "human_review",
        }
    )
    write_records([record] * n_records, tmp_path / ".jeval" / "records.jsonl")
    (tmp_path / "costs.yaml").write_text(
        "actions:\n  - name: auto\n    question: q\n    when: a\n"
        "    cost_false_accept: 100\n    cost_escalate: 1\n    cost_false_reject: 0\n"
    )
    result = CliRunner().invoke(app, ["threshold", "--root", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert "DecisionRecord.stated_probability >= threshold" in result.stdout
    assert "choice: P(prediction)" in result.stdout
    assert "noul: max(p, 1-p)" in result.stdout
    assert "score: reported confidence" in result.stdout
    assert "not a rescaled choice SDK confidence" in result.stdout
    entry = yaml.safe_load((tmp_path / "thresholds.yaml").read_text())["actions"]["auto"]
    assert entry["comparison"] == COMPARISON
    if n_records == 1:
        assert entry["threshold"] is None
        assert entry["insufficient_data"] is True


def test_native_choice_threshold_exports_an_explicit_comparison(tmp_path: Path) -> None:
    row = {
        "response": {
            "model": "jev-1.13.0",
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.6, "b": 0.3, "c": 0.1},
                    "confidence": 0.4,
                }
            },
        }
    }
    record = normalize_record(presets.rows_to_payloads(presets.JEV_NATIVE, row)[0])
    record = record.model_copy(update={"label": "a", "label_source": "human_review"})
    action = CostAction("auto", "intent", "a", 100.0, 1.0, 0.0)
    result = sweep(action, [record] * 30, n_boot=0)
    target = tmp_path / "thresholds.yaml"
    write_thresholds_yaml([result], target, generated_at=STAMP)
    entry = yaml.safe_load(target.read_text())["actions"]["auto"]

    assert record.confidence == pytest.approx(0.6)
    assert entry["threshold"] == pytest.approx(0.6)
    assert entry["auto_rate"] == 1.0
    # Comparing the SDK's rescaled 0.4 with this line would escalate this answer instead.
    assert row["response"]["answers"]["intent"]["confidence"] < entry["threshold"]
    assert evaluate_point(action, [record], entry["threshold"]).n_auto == 1
    assert entry["comparison"] == COMPARISON


def test_native_score_ingest_preserves_reported_confidence(tmp_path: Path) -> None:
    source = tmp_path / "native.jsonl"
    source.write_text(
        json.dumps(
            {
                "response": {
                    "model": "jev-1.13.0",
                    "answers": {
                        "severity": {
                            "type": "score",
                            "score": 1.6,
                            "confidence": 0.4,
                            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                        }
                    },
                }
            }
        )
        + "\n"
    )
    target = tmp_path / "records.jsonl"
    report = ingest_preset_files([source], target, presets.JEV_NATIVE)
    [record] = read_records(target)

    assert report.n_records == 1
    assert record.prediction == "1.6"
    assert record.confidence == 0.4
    assert record.probabilities == {"0": 0.1, "1": 0.2, "2": 0.7}
    assert record.stated_probability == 0.4
    assert record.calibration_point() is None


def test_variable_choice_option_counts_do_not_convert_the_threshold(tmp_path: Path) -> None:
    records = []
    native_confidences = []
    for n in (3, 4):
        probabilities = {"a": 0.84, **{str(i): 0.16 / (n - 1) for i in range(n - 1)}}
        native = (0.84 - 1 / n) / (1 - 1 / n)
        native_confidences.append(native)
        row = {
            "response": {
                "model": "jev-1.13.0",
                "answers": {
                    "intent": {
                        "type": "choice",
                        "choice": "a",
                        "probabilities": probabilities,
                        "confidence": native,
                    }
                },
            }
        }
        record = normalize_record(presets.rows_to_payloads(presets.JEV_NATIVE, row)[0])
        records.extend(
            [record.model_copy(update={"label": "a", "label_source": "human_review"})] * 30
        )
    action = CostAction("auto", "intent", "a", 100.0, 1.0, 0.0)
    result = sweep(action, records, n_boot=0)
    target = tmp_path / "thresholds.yaml"
    write_thresholds_yaml([result], target, generated_at=STAMP)
    entry = yaml.safe_load(target.read_text())["actions"]["auto"]

    assert native_confidences[0] != native_confidences[1]
    assert all(confidence < 0.84 for confidence in native_confidences)
    assert entry["threshold"] == 0.84
    assert entry["comparison"] == COMPARISON
    assert (
        entry["auto_rate"] == evaluate_point(action, records, entry["threshold"]).auto_rate == 1.0
    )
    assert "native_threshold" not in entry


def test_mixed_types_and_actions_use_type_dispatch_not_a_single_scale(tmp_path: Path) -> None:
    records = [
        normalize_record(
            {
                "model": "m",
                "question_key": "q",
                "question_type": kind,
                "prediction": "yes",
                "confidence": 0.4,
                "label": "yes",
                "label_source": "human_review",
            }
        )
        for kind in ("choice", "noul", "score")
    ] * 10
    actions = [CostAction(name, "q", "yes", 100.0, 1.0, 0.0) for name in ("auto", "other")]
    results = [sweep(action, records, n_boot=0) for action in actions]
    target = tmp_path / "thresholds.yaml"
    write_thresholds_yaml(results, target, generated_at=STAMP)
    payload = yaml.safe_load(target.read_text())

    assert "comparison" not in payload  # no misleading global scale
    for action in actions:
        entry = payload["actions"][action.name]
        assert entry["comparison"] == COMPARISON
        assert entry["threshold"] == 0.4
        assert evaluate_point(action, records, entry["threshold"]).n_auto == 30
    assert deployed_threshold(tmp_path, None) == 0.4
    # Older readers/files need no migration: comparison is additive, never required to read t.
    for entry in payload["actions"].values():
        del entry["comparison"]
    target.write_text(yaml.safe_dump(payload))
    assert deployed_threshold(tmp_path, None) == 0.4


def test_readme_comparison_contract_matches_the_export(tmp_path: Path) -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    [block] = re.findall(r"```yaml\n# Comparison contract[^\n]*\n(.*?)```", readme, re.DOTALL)
    target = tmp_path / "thresholds.yaml"
    result = sweep(CostAction("auto", "missing", "a", 1.0, 1.0, 0.0), [], n_boot=0)
    write_thresholds_yaml([result], target, generated_at=STAMP)
    entry = yaml.safe_load(target.read_text())["actions"]["auto"]

    assert yaml.safe_load(block)["comparison"] == entry["comparison"] == COMPARISON
    assert entry["threshold"] is None
