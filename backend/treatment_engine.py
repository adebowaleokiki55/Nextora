
from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# NEXTORA TREATMENT DECISION-SUPPORT PROTOTYPE
# ============================================================
# This module is for a hackathon research prototype.
#
# It does NOT diagnose malaria.
# It does NOT interpret image candidate counts as parasitaemia.
# It does NOT calculate medication doses.
#
# Medication protocols must be verified by a qualified clinician
# against the applicable current Nigerian guidance before dosing
# functionality is enabled.
# ============================================================


NIGERIA_PROTOCOL = {
    "name": "Guidelines for the Treatment of Malaria, 4th Edition",
    "publisher": "Nigeria National Malaria Elimination Programme",
    "year": 2020,
    "url": (
        "https://nmcp.gov.ng/about-learning-hub/case-management/"
    ),
    "dosing_table_verified": False,
}

WHO_REFERENCE = {
    "name": "WHO Guidelines for malaria",
    "publisher": "World Health Organization",
    "year": 2026,
    "url": "https://www.who.int/publications/i/item/guidelines-for-malaria/",
}


class TreatmentRequest(BaseModel):
    # A positive result from Nextora alone is not a confirmed diagnosis.
    laboratory_confirmed: bool = False

    confirmation_method: Literal[
        "microscopy",
        "rapid_diagnostic_test",
        "other",
        "not_confirmed",
    ] = "not_confirmed"

    malaria_species: Literal[
        "falciparum",
        "vivax",
        "ovale",
        "malariae",
        "knowlesi",
        "unknown",
    ] = "unknown"

    severity: Literal[
        "uncomplicated",
        "severe",
        "unknown",
    ] = "unknown"

    patient_weight_kg: float | None = Field(
        default=None,
        gt=0,
        le=300,
    )

    age_years: float | None = Field(
        default=None,
        ge=0,
        le=120,
    )

    pregnant: bool | None = None

    pregnancy_trimester: Literal[
        "first",
        "second",
        "third",
        "unknown",
        "not_applicable",
    ] = "not_applicable"

    known_allergies: list[str] = Field(default_factory=list)

    relevant_medications: list[str] = Field(default_factory=list)

    # A real integration should obtain this from a validated laboratory
    # report, not infer it from image detections.
    parasite_density_per_microlitre: float | None = Field(
        default=None,
        ge=0,
    )


def assess_treatment(request: TreatmentRequest) -> dict:
    """
    Produce a preliminary treatment-pathway assessment.

    Does not prescribe or calculate a drug dose.
    """

    missing_information = []
    warnings = []
    recommendation = None
    urgency = "routine_clinician_review"

    # --------------------------------------------------------
    # 1. Severe malaria safety branch
    # --------------------------------------------------------

    if request.severity == "severe":
        return {
            "status": "urgent_escalation",
            "urgency": "emergency",
            "recommendation": (
                "Suspected severe malaria requires immediate "
                "emergency hospital assessment and management "
                "under the applicable clinical protocol."
            ),
            "medication": None,
            "dose": None,
            "reasons": [
                "Severe malaria is a medical emergency.",
                "This prototype must not generate an outpatient "
                "oral-treatment plan for severe malaria.",
            ],
            "missing_information": [],
            "warnings": [
                "Do not delay emergency care while using this tool."
            ],
            "protocol": NIGERIA_PROTOCOL,
            "clinical_validation": False,
        }

    # --------------------------------------------------------
    # 2. Image prediction is not diagnostic confirmation
    # --------------------------------------------------------

    if not request.laboratory_confirmed:
        return {
            "status": "confirmation_required",
            "urgency": "clinical_review",
            "recommendation": (
                "Obtain parasitological confirmation using an "
                "appropriate validated test before selecting a "
                "routine antimalarial treatment pathway."
            ),
            "medication": None,
            "dose": None,
            "reasons": [
                (
                    "Nextora's image classifier is a research "
                    "prototype and has not been clinically validated."
                ),
                (
                    "A model prediction alone cannot establish "
                    "malaria species, severity, or parasite density."
                ),
            ],
            "missing_information": [
                "Laboratory confirmation",
                "Clinical severity assessment",
            ],
            "warnings": [
                (
                    "If severe malaria is suspected, arrange "
                    "emergency assessment immediately."
                )
            ],
            "protocol": NIGERIA_PROTOCOL,
            "clinical_validation": False,
        }

    # --------------------------------------------------------
    # 3. Require an appropriate clinical severity assessment
    # --------------------------------------------------------

    if request.severity == "unknown":
        missing_information.append(
            "Clinical severity classification"
        )

    if request.malaria_species == "unknown":
        missing_information.append(
            "Confirmed malaria species or clinician interpretation"
        )

    if request.patient_weight_kg is None:
        missing_information.append("Patient weight in kilograms")

    if request.pregnant is None:
        missing_information.append(
            "Pregnancy status, where clinically applicable"
        )

    if request.known_allergies:
        warnings.append(
            "Review known allergies before selecting any medication."
        )

    if request.relevant_medications:
        warnings.append(
            "Review concurrent medications and potential interactions."
        )

    # --------------------------------------------------------
    # 4. Do not create a regimen when critical details are missing
    # --------------------------------------------------------

    if missing_information:
        return {
            "status": "insufficient_clinical_information",
            "urgency": "clinician_review",
            "recommendation": (
                "Complete the missing clinical assessment before "
                "selecting a treatment regimen."
            ),
            "medication": None,
            "dose": None,
            "missing_information": missing_information,
            "warnings": warnings,
            "protocol": NIGERIA_PROTOCOL,
            "clinical_validation": False,
        }

    # --------------------------------------------------------
    # 5. Do not infer a drug regimen for species outside the
    #    supported initial pathway.
    # --------------------------------------------------------

    if request.malaria_species != "falciparum":
        return {
            "status": "specialist_protocol_review",
            "urgency": "clinician_review",
            "recommendation": (
                "Use the applicable species-specific treatment "
                "guidance. This prototype currently supports only "
                "preliminary pathway assessment for P. falciparum."
            ),
            "medication": None,
            "dose": None,
            "warnings": warnings,
            "protocol": NIGERIA_PROTOCOL,
            "clinical_validation": False,
        }

    # --------------------------------------------------------
    # 6. Pregnancy requires a pregnancy-specific pathway.
    #    Do not use a generic recommendation for this group.
    # --------------------------------------------------------

    if request.pregnant:
        return {
            "status": "pregnancy_specific_review",
            "urgency": "clinician_review",
            "recommendation": (
                "Use the applicable malaria-in-pregnancy protocol. "
                "A clinician must review gestational age and the "
                "patient's clinical circumstances before selecting "
                "a regimen."
            ),
            "medication": None,
            "dose": None,
            "warnings": warnings,
            "protocol": NIGERIA_PROTOCOL,
            "clinical_validation": False,
        }

    # --------------------------------------------------------
    # 7. Preliminary uncomplicated P. falciparum pathway
    # --------------------------------------------------------

    if request.severity == "uncomplicated":
        if not NIGERIA_PROTOCOL["dosing_table_verified"]:
            recommendation = (
                "The Nigerian guidance identifies artemisinin-based "
                "combination therapy (ACT) as a treatment class for "
                "uncomplicated falciparum malaria. Artemether-"
                "lumefantrine is a regimen to evaluate against the "
                "verified applicable protocol. The exact regimen, "
                "product strength, eligibility and dose have not "
                "been enabled because the relevant dosing table has "
                "not yet been verified in this application."
            )

            warnings.append(
                "No dose calculated. Do not use this output as a prescription."
            )

            return {
                "status": "protocol_review_required",
                "urgency": "clinician_review",
                "recommendation": recommendation,
                "medication": {
                    "class": "Artemisinin-based combination therapy",
                    "candidate_regimen": "Artemether-lumefantrine",
                    "selection_status": "requires protocol verification",
                },
                "dose": None,
                "reasons": [
                    "Species entered: P. falciparum",
                    "Severity entered: uncomplicated",
                    "Laboratory confirmation reported by requester",
                ],
                "warnings": warnings,
                "protocol": NIGERIA_PROTOCOL,
                "clinical_validation": False,
            }

    return {
        "status": "manual_review_required",
        "urgency": "clinician_review",
        "recommendation": (
            "The available inputs do not match a supported treatment "
            "pathway. Refer to the applicable clinical protocol."
        ),
        "medication": None,
        "dose": None,
        "warnings": warnings,
        "protocol": NIGERIA_PROTOCOL,
        "clinical_validation": False,
    }
