"""Treatment decision-support explanation agent (side page, not a prescriber)."""

from __future__ import annotations

import json
from typing import Any

from backend.app.agents.runner import AgentTool, run_tool_loop
from backend.app.core.llm import LLMClient, ToolSpec

SYSTEM_PROMPT = (
    "You are Nextora's treatment decision-support "
    "explanation agent. You must call the approved "
    "assess_treatment tool before responding. Use only "
    "the tool's returned assessment. Never invent a drug, "
    "dose, frequency, duration, contraindication, laboratory "
    "finding, or patient detail. Do not override the "
    "assessment's status or warnings. If the result says "
    "protocol_review_required, explain that the regimen "
    "requires verification against the applicable Nigerian "
    "protocol by a qualified clinician. If dose is null, "
    "explicitly state that no dose was calculated. If the "
    "result indicates emergency escalation, emphasize urgent "
    "hospital assessment. Do not present this output as a "
    "prescription. The system cannot independently verify "
    "that laboratory confirmation occurred. State that this "
    "prototype has not been clinically validated."
)

TOOL_SPEC = ToolSpec(
    name="assess_treatment",
    description=(
        "Run Nextora's deterministic treatment assessment "
        "using the clinical information submitted with this "
        "request. Never invent or change patient details. "
        "Do not prescribe or invent doses."
    ),
)


def build_treatment_report(
    client: LLMClient,
    assessment: dict[str, Any],
    submitted_data: dict[str, Any],
) -> dict[str, Any]:
    tools = [AgentTool(spec=TOOL_SPEC, handler=lambda _args: assessment)]

    user_message = (
        "Call the approved treatment assessment tool and explain "
        "its result, urgency, unverified information, and next "
        "step. Do not create a prescription.\n\n"
        "Submitted clinical information:\n"
        + json.dumps(submitted_data, ensure_ascii=False)
    )

    result = run_tool_loop(
        client,
        SYSTEM_PROMPT,
        user_message,
        tools,
        required_tools={"assess_treatment"},
        max_rounds=4,
        reminder=lambda _missing: (
            "You must call the approved assess_treatment tool before explaining the result."
        ),
        exhausted_detail=(
            "The treatment agent could not complete its required tool call and explanation."
        ),
    )

    return {
        "assessment": assessment,
        "explanation": result.answer,
        "tools_executed": ["assess_treatment"],
        "agent_model": client.model,
        "agent_runtime": client.runtime_label,
        "clinical_validation": False,
        "disclaimer": (
            "Research prototype only. This output is not "
            "a diagnosis or prescription. A qualified "
            "clinician must assess the patient and verify "
            "the applicable treatment protocol."
        ),
    }
