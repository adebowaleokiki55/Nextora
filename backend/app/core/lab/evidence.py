"""Deterministic evidence tools used by the lab explanation agent.

Moved unchanged from the original backend/main.py.
"""

from __future__ import annotations


def inspect_analysis_evidence(mode: str, analysis: dict) -> dict:
    if mode == "single_cell":
        required_fields = [
            "predicted_class",
            "prediction",
            "confidence",
            "probabilities",
            "model",
        ]

        missing_fields = [field for field in required_fields if field not in analysis]

        return {
            "mode": mode,
            "prediction": analysis.get("prediction"),
            "predicted_class": analysis.get("predicted_class"),
            "confidence": analysis.get("confidence"),
            "probabilities": analysis.get("probabilities"),
            "model": analysis.get("model"),
            "missing_fields": missing_fields,
            "disclaimer": analysis.get("disclaimer"),
        }

    cells = analysis.get("cells", [])

    if not isinstance(cells, list):
        cells = []

    infected = sum(
        1 for cell in cells if isinstance(cell, dict) and cell.get("predicted_class") == 0
    )
    uninfected = sum(
        1 for cell in cells if isinstance(cell, dict) and cell.get("predicted_class") == 1
    )

    return {
        "mode": mode,
        "image_width": analysis.get("image_width"),
        "image_height": analysis.get("image_height"),
        "reported_candidate_count": analysis.get("candidate_count"),
        "reported_infected_count": analysis.get("candidate_infected_count"),
        "reported_uninfected_count": analysis.get("candidate_uninfected_count"),
        "cells_returned": len(cells),
        "counts_from_returned_cell_records": {
            "infected_class": infected,
            "uninfected_class": uninfected,
        },
        "warning": (
            "Candidate regions are heuristic detections and are not "
            "necessarily confirmed individual cells."
        ),
    }


def check_analysis_uncertainty(mode: str, analysis: dict) -> dict:
    issues = []
    checks = {}

    if mode == "single_cell":
        confidence = analysis.get("confidence")
        probabilities = analysis.get("probabilities")

        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            issues.append("Confidence is missing or not numeric.")
        elif not 0 <= confidence <= 100:
            issues.append("Confidence is outside the 0–100 range.")
        else:
            checks["confidence_in_valid_range"] = True

        if not isinstance(probabilities, dict):
            issues.append("Class probabilities are missing.")
        else:
            p_infected = probabilities.get("parasitized")
            p_uninfected = probabilities.get("uninfected")

            valid_probabilities = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0 <= value <= 100
                for value in [p_infected, p_uninfected]
            )

            if not valid_probabilities:
                issues.append("Class probabilities are missing or invalid.")
            else:
                total = p_infected + p_uninfected
                checks["probability_total_percent"] = round(total, 2)
                checks["probabilities_sum_to_100"] = abs(total - 100) <= 1

                if abs(total - 100) > 1:
                    issues.append("Class probabilities do not sum to approximately 100%.")

                predicted_class = analysis.get("predicted_class")

                expected_confidence = (
                    p_infected if predicted_class == 0
                    else p_uninfected if predicted_class == 1
                    else None
                )

                if (
                    isinstance(confidence, (int, float))
                    and not isinstance(confidence, bool)
                    and expected_confidence is not None
                ):
                    matches = abs(confidence - expected_confidence) <= 1
                    checks["confidence_matches_predicted_class"] = matches

                    if not matches:
                        issues.append(
                            "Confidence does not match the probability for the predicted class."
                        )

        if analysis.get("prediction") not in {"Parasitized (Infected)", "Uninfected"}:
            issues.append("Prediction label is missing or unrecognized.")

        checks["clinical_calibration_established"] = False
        issues.append(
            "The model's confidence has not been established as a "
            "calibrated clinical probability."
        )

    else:
        candidate_count = analysis.get("candidate_count")
        cells = analysis.get("cells", [])

        if (
            not isinstance(candidate_count, int)
            or isinstance(candidate_count, bool)
            or candidate_count < 0
        ):
            issues.append("Candidate-region count is missing or invalid.")

        if not isinstance(cells, list):
            issues.append("Cell result records are missing or invalid.")
        else:
            checks["returned_cell_records"] = len(cells)

            if (
                isinstance(candidate_count, int)
                and not isinstance(candidate_count, bool)
                and candidate_count != len(cells)
            ):
                issues.append(
                    "Reported candidate count differs from the number "
                    "of returned cell records. Results may be filtered "
                    "or truncated."
                )

        checks["candidate_regions_confirmed_as_cells"] = False
        checks["parasitemia_validated"] = False

        issues.extend([
            "Whole-smear candidate regions are heuristic and may "
            "not represent confirmed individual cells.",
            "Candidate counts are not a validated parasitemia measurement.",
        ])

    checks["clinical_validation"] = False

    return {
        "mode": mode,
        "checks": checks,
        "issues": issues,
        "interpretation": (
            "These are software consistency and evidence-completeness "
            "checks. They do not validate the model clinically or "
            "establish whether a patient has malaria."
        ),
    }
