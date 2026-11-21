"""OFFLINE DEMO MODE: a scripted run that calls the same real tools.

Used when no AI provider is configured or there is no internet (the stage
backup). The sequence and wording are fixed templates filled with the real
tool results; no language model is involved, and every message says so.
"""

from __future__ import annotations

import time
from typing import Any

from backend.app.agents.command_centre import (
    COMMANDER,
    OUTREACH,
    POLICY,
    SUPPLY,
    SURVEILLANCE,
    CommandCentre,
)
from backend.app.models import AgentMessageKind

LABEL = "[Scripted, offline demo mode: no AI model] "


class OfflineScript:
    def __init__(self, centre: CommandCentre, delay: float = 0.6):
        self.centre = centre
        self.log = centre.logger.log
        self.delay = delay
        self.tools = {
            agent: {t.spec.name: t.handler for t in centre.tools(agent)}
            for agent in (SURVEILLANCE, SUPPLY, OUTREACH, POLICY, COMMANDER)
        }

    def _pause(self) -> None:
        if self.delay:
            time.sleep(self.delay)

    def call(self, agent: str, tool: str, args: dict | None = None) -> Any:
        args = args or {}
        self.log(agent, AgentMessageKind.TOOL_CALL, args, tool_name=tool)
        result = self.tools[agent][tool](args)
        self.log(agent, AgentMessageKind.TOOL_RESULT, result, tool_name=tool)
        self._pause()
        return result

    def say(self, agent: str, kind: AgentMessageKind, text: str, payload: dict | None = None) -> None:
        self.log(agent, kind, LABEL + text, payload=payload)
        self._pause()

    def handoff(self, specialist: str, task: str) -> None:
        self.log(COMMANDER, AgentMessageKind.HANDOFF, LABEL + task, payload={"to": specialist})
        self._pause()

    def report(self, specialist: str, text: str) -> None:
        self.log(specialist, AgentMessageKind.HANDOFF, LABEL + text, payload={"to": COMMANDER})
        self._pause()

    # ------------------------------------------------------------------ run

    def run(self) -> str:
        name = self.centre.location_name
        self.say(COMMANDER, AgentMessageKind.THOUGHT, f"Starting a coordinated response for {name}. Reviewing the situation first.")
        snapshot = self.call(COMMANDER, "get_situation_snapshot")

        surveillance = self._surveillance(name)
        supply = self._supply(name)
        outreach = self._outreach(name, snapshot)
        self._policy(name)

        summary = (
            f"{name}: {surveillance} {supply} {outreach} "
            "A human must review and approve or reject each pending proposal in the approval queue."
        )
        self.say(COMMANDER, AgentMessageKind.FINAL, summary)
        return LABEL + summary

    def _surveillance(self, name: str) -> str:
        self.handoff(SURVEILLANCE, f"Investigate the malaria risk signals for {name}.")
        signals = self.call(SURVEILLANCE, "get_surveillance_signals")
        risk = self.call(SURVEILLANCE, "get_risk_forecast")
        lab = self.call(SURVEILLANCE, "get_verified_lab_summary")

        elevated = [s["explanation"] for s in signals["signals"] if s["status"] != "normal"]
        rain = next((s for s in signals["signals"] if s["key"] == "rainfall_8wk_anomaly"), None)
        parts = [f"Surveillance level is {signals['level'].upper()}."] + elevated
        if rain and rain["status"] == "normal" and signals["level"] != "normal":
            parts.append("Rainfall is not above normal, so weather does not explain the rise.")
        if risk.get("available"):
            parts.append(
                f"Forecast risk score {risk['risk_score']}/100 ({risk['risk_level']}); early warning "
                f"{'active' if risk['early_warning'] else 'not active'}."
            )
        parts.append(
            f"Verified lab results (last {lab['verified_last_weeks']} weeks): {lab['verified_positive_smears']} positive "
            f"of {lab['verified_smears']}; {lab['pending_review']} awaiting review."
        )
        if signals["data_origin"].get("cases") == "SYNTHETIC":
            parts.append("Case data is SYNTHETIC (demo).")
        text = " ".join(parts)
        self.report(SURVEILLANCE, text)
        level = risk.get("risk_level", "unknown")
        return f"Risk is {level} (score {risk.get('risk_score')})."

    def _supply(self, name: str) -> str:
        self.handoff(SUPPLY, f"Check RDT kit and antimalarial stock in {name} and prepare redistribution if needed.")
        shortages = self.call(SUPPLY, "get_supply_shortages")["shortages"]
        if not shortages:
            self.report(SUPPLY, f"No facility in {name} is short of tracked malaria commodities.")
            return "No supply shortages."

        self.call(SUPPLY, "get_redistribution_plan")
        proposed = self.call(SUPPLY, "propose_redistribution")
        lines = [
            f"{s['facility']['name']}: {s['status']} {s['item']['code']} ({s['on_hand']} {s['item']['unit']} on hand, "
            f"needs {s['shortage_quantity']})"
            for s in shortages
        ]
        transfers = [p for p in proposed["created"] if p["proposal_type"] == "STOCK_TRANSFER"]
        orders = [p for p in proposed["created"] if p["proposal_type"] == "REPLENISHMENT_ORDER"]
        text = (
            f"{len(shortages)} shortage(s): " + "; ".join(lines) + ". "
            f"Proposed {len(transfers)} transfer(s) and {len(orders)} central order(s) for approval"
            + (f" ({proposed['already_pending']} were already pending)" if proposed["already_pending"] else "")
            + ". Stock data is SYNTHETIC (demo)."
        )
        self.report(SUPPLY, text)
        return f"{len(shortages)} facilities short; {len(transfers)} transfers and {len(orders)} orders await approval."

    def _outreach(self, name: str, snapshot: dict) -> str:
        self.handoff(OUTREACH, f"Decide whether {name} needs a public prevention message and draft one if so.")
        ctx = self.call(OUTREACH, "get_outreach_context")
        if not (ctx["early_warning"] or ctx["risk_level"] in ("HIGH", "VERY_HIGH")):
            self.report(OUTREACH, f"Risk in {name} is {ctx['risk_level']}; no public message is needed now.")
            return "No public message proposed."

        message = (
            f"Malaria cases are rising in {name}. Sleep under an insecticide-treated net every night, "
            "remove standing water around your home, and if anyone has a fever, get a malaria test at "
            "the nearest health facility within 24 hours."
        )
        result = self.call(
            OUTREACH,
            "propose_public_message",
            {"audience": f"Residents of {name}", "channel": "SMS", "message": message, "language": "English"},
        )
        if "error" in result:
            self.report(OUTREACH, f"Could not submit the draft: {result['error']}")
            return "Public message not proposed."
        self.report(OUTREACH, f"Drafted an SMS for residents of {name} and submitted it for approval: \"{message}\"")
        return "One public SMS draft awaits approval."

    def _policy(self, name: str) -> None:
        self.handoff(POLICY, f"Write a situation report for {name}.")
        s = self.call(POLICY, "get_situation_snapshot")
        risk = s["risk_forecast"]
        lab = s["verified_lab"]
        shortages = s["supply_shortages"]
        pending = s["pending_proposals"]
        elevated = [sig["explanation"] for sig in s["surveillance"]["signals"] if sig["status"] != "normal"]
        report = "\n".join(
            [
                f"## Situation\nSurveillance level for {name} is {s['surveillance']['level'].upper()} "
                f"(week of {s['surveillance']['as_of_week']})."
                + (f" Forecast risk {risk['risk_score']}/100 ({risk['risk_level']})." if risk.get("available") else ""),
                "## Evidence and confidence\n" + ("\n".join(f"- {e}" for e in elevated) or "- No elevated signals.")
                + f"\n- Verified lab: {lab['verified_positive_smears']} positive of {lab['verified_smears']} reviewed smears.",
                "## Actions awaiting approval\n" + ("\n".join(f"- {p['title']}" for p in pending) or "- None."),
                "## Recommendations\n"
                + ("- Approve or adjust the stock transfers so affected facilities keep testing.\n" if shortages else "")
                + "- Review the public message draft before dispatch.\n- Re-run forecasts when new weekly data arrives.",
                "## Data limitations\nCase counts, stock levels and facilities are SYNTHETIC demo data; weather is "
                + (
                    "real reanalysis (Open-Meteo)."
                    if s["surveillance"]["data_origin"].get("weather") == "REAL"
                    else "also SYNTHETIC."
                )
                + " Nextora is a decision-support prototype, not an official surveillance system.",
            ]
        )
        self.report(POLICY, report)
