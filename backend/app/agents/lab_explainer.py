"""Lab research-assistant agent: explains a single-cell or whole-smear result."""

from __future__ import annotations

import json
from typing import Any

from backend.app.agents.runner import AgentTool, run_tool_loop
from backend.app.core.lab.evidence import check_analysis_uncertainty, inspect_analysis_evidence
from backend.app.core.llm import LLMClient, ToolSpec

SYSTEM_PROMPT = (
    "You are Nextora's research-assistant agent. You MUST "
    "call both inspect_analysis_evidence and "
    "check_analysis_uncertainty before your final report. "
    "Use their actual returned results. Never invent findings, "
    "counts, or probabilities. Do not diagnose patients or "
    "prescribe medication. Do not describe model confidence "
    "as a calibrated probability of disease. In single-cell "
    "mode, discuss only the isolated-cell result. In "
    "whole-smear mode, explain that candidate regions are "
    "heuristic and are not necessarily confirmed individual "
    "cells. Do not call candidate counts a validated parasite "
    "count or parasitemia. The model has not been clinically "
    "validated. Finish with limitations and qualified "
    "laboratory confirmation."
)

TOOL_SPECS = [
    ToolSpec(
        name="inspect_analysis_evidence",
        description=(
            "Inspect the supplied Nextora analysis and summarize "
            "the actual prediction evidence. Call before reporting."
        ),
    ),
    ToolSpec(
        name="check_analysis_uncertainty",
        description=(
            "Check missing fields, confidence validity, probability "
            "consistency, and known limitations. Call before reporting."
        ),
    ),
]

REQUIRED_TOOLS = {"inspect_analysis_evidence", "check_analysis_uncertainty"}


def explain_analysis(client: LLMClient, mode: str, analysis: dict[str, Any]) -> dict[str, Any]:
    evidence = inspect_analysis_evidence(mode, analysis)

    # The model never supplies tool inputs: every handler reads the analysis
    # submitted with the request.
    handlers = {
        "inspect_analysis_evidence": lambda _args: inspect_analysis_evidence(mode, analysis),
        "check_analysis_uncertainty": lambda _args: check_analysis_uncertainty(mode, analysis),
    }
    tools = [AgentTool(spec=spec, handler=handlers[spec.name]) for spec in TOOL_SPECS]

    user_message = (
        f"Analyze this Nextora {mode} result. Call both required "
        "Python tools, examine their results, and produce a concise "
        "evidence-based report.\n\nSupplied analysis:\n"
        + json.dumps(analysis, ensure_ascii=False)
    )

    result = run_tool_loop(
        client,
        SYSTEM_PROMPT,
        user_message,
        tools,
        required_tools=REQUIRED_TOOLS,
        max_rounds=6,
        exhausted_detail=(
            "The agent could not complete its required tool calls "
            "and final report within the allowed rounds."
        ),
    )

    return {
        "mode": mode,
        "explanation": result.answer,
        "evidence": evidence,
        "tool_results": result.tool_results,
        "tools_executed": result.tools_executed,
        "agent_model": client.model,
        "agent_runtime": client.runtime_label,
        "clinical_validation": False,
        "disclaimer": (
            "Research prototype only. This explanation is "
            "not a diagnosis. Seek qualified laboratory "
            "confirmation."
        ),
    }
