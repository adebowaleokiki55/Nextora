"""Module 3: the multi-agent command centre.

Commander delegates to four specialists through tools:

    Commander ──delegate_to_surveillance──► Surveillance Agent
              ──delegate_to_supply────────► Supply Agent
              ──delegate_to_outreach──────► Outreach Agent
              ──delegate_to_policy────────► Policy Agent

Every tool is a deterministic Python function over the shared database,
scoped to the run's location. Agents can only *propose* consequential
actions; proposals wait in the approval queue for a human.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from backend.app.agents.run_log import RunLogger
from backend.app.agents.runner import AgentTool, run_tool_loop
from backend.app.core.command import context
from backend.app.core.command.proposals import MESSAGE_CHANNELS, ProposalError, propose_public_message, propose_redistribution
from backend.app.core.llm import LLMClient, ToolSpec
from backend.app.models import AgentMessageKind

COMMANDER = "Commander"
SURVEILLANCE = "Surveillance"
SUPPLY = "Supply"
OUTREACH = "Outreach"
POLICY = "Policy"

AGENT_MAX_TOKENS = 2000

COMMON_RULES = """You are part of Nextora's malaria command centre, a decision-support prototype for public-health teams in Lagos State, Nigeria.
Rules you must follow:
- Use only facts and numbers returned by your tools. Never invent or estimate case counts, quantities, probabilities, dates or names.
- Say plainly when data is SYNTHETIC (demo data) rather than real.
- You cannot act directly. Consequential actions (stock transfers, orders, public messages) can only be proposed through the propose_* tools; a named human must approve them before anything happens.
- Do not diagnose patients or give individual treatment advice. Do not name medicines or doses in public messages.
- Be concise and specific. Plain text or simple Markdown."""

PROMPTS = {
    SURVEILLANCE: """You are the Surveillance Agent. Investigate the malaria risk signals for {location}.
Call get_surveillance_signals, get_risk_forecast and get_verified_lab_summary, then report in under 150 words:
what is happening, how strong the evidence is, and what is or is not driving it (for example, say so if rainfall does not explain a rise).""",
    SUPPLY: """You are the Supply Agent. Check malaria commodity stock in {location}.
Call get_supply_shortages. If there are shortages, call get_redistribution_plan and then propose_redistribution to put the plan in the approval queue.
Report in under 150 words: which facilities are short, what was proposed (quantities exactly as returned by the tools) and what still needs ordering.""",
    OUTREACH: """You are the Outreach Agent. Decide whether {location} needs a public prevention message now.
Call get_outreach_context first. If the risk level is HIGH or VERY_HIGH, or an early warning is active, draft ONE message (40-480 characters, plain English, warm and clear, suitable for SMS or community health workers) and submit it with propose_public_message.
Use only the key messages from the context. Do not mention data sources, models, probabilities or medicines. If risk is low, do not propose anything and explain why.
Report in under 100 words.""",
    POLICY: """You are the Policy Agent. Write a situation report for {location} for the public-health response coordinator.
Call get_situation_snapshot, then write at most 250 words of Markdown with these headings:
## Situation, ## Evidence and confidence, ## Actions awaiting approval, ## Recommendations, ## Data limitations.
Under Data limitations, state which data is synthetic and that this is a decision-support prototype.""",
    COMMANDER: """You are the Commander Agent coordinating the malaria response for {location}.
First call get_situation_snapshot. Then delegate, giving each specialist a short, specific task:
delegate_to_surveillance, delegate_to_supply, delegate_to_outreach and finally delegate_to_policy.
Finish with a brief summary for the human response coordinator (under 150 words): the situation, what each specialist found or proposed, and exactly what a human must review and approve next.""",
}

REQUIRED = {
    SURVEILLANCE: {"get_surveillance_signals", "get_risk_forecast", "get_verified_lab_summary"},
    SUPPLY: {"get_supply_shortages"},
    OUTREACH: {"get_outreach_context"},
    POLICY: {"get_situation_snapshot"},
    COMMANDER: {
        "get_situation_snapshot",
        "delegate_to_surveillance",
        "delegate_to_supply",
        "delegate_to_outreach",
        "delegate_to_policy",
    },
}

KEY_MESSAGES = [
    "Sleep under an insecticide-treated mosquito net every night, especially children under five and pregnant women.",
    "Remove standing water around homes: empty containers, clear blocked gutters, cover water storage.",
    "Anyone with fever should get a malaria test at the nearest health facility within 24 hours.",
    "Finish any treatment exactly as prescribed by a health worker.",
]

_NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
_TASK_ARGS: dict[str, Any] = {
    "type": "object",
    "properties": {"task": {"type": "string", "description": "A short, specific instruction for the specialist."}},
    "required": ["task"],
}


class CommandCentre:
    """Builds the agents' tools for one run and one location."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        logger: RunLogger,
        run_id: int,
        location_id: int,
        client: LLMClient | None = None,
    ):
        self.factory = session_factory
        self.logger = logger
        self.run_id = run_id
        self.location_id = location_id
        self.client = client
        with session_factory() as session:
            self.location_name = context.get_location(session, location_id).name

    # ---------------------------------------------------------------- tools

    def _read(self, fn: Callable[[Session, int], Any]) -> Callable[[dict], Any]:
        def handler(_args: dict) -> Any:
            with self.factory() as session:
                return fn(session, self.location_id)

        return handler

    def get_outreach_context(self, _args: dict | None = None) -> dict[str, Any]:
        with self.factory() as session:
            risk = context.risk_forecast(session, self.location_id)
            signals = context.surveillance_signals(session, self.location_id)
        return {
            "location": self.location_name,
            "risk_level": risk.get("risk_level"),
            "risk_score": risk.get("risk_score"),
            "early_warning": risk.get("early_warning"),
            "surveillance_level": signals["level"],
            "elevated_signals": [s["label"] for s in signals["signals"] if s["status"] != "normal"],
            "key_messages": KEY_MESSAGES,
            "allowed_channels": list(MESSAGE_CHANNELS),
            "message_length": "40-480 characters",
        }

    def propose_redistribution(self, _args: dict | None = None) -> dict[str, Any]:
        with self.factory() as session:
            return propose_redistribution(session, self.location_id, proposed_by=f"{SUPPLY} Agent", agent_run_id=self.run_id)

    def propose_public_message(self, args: dict) -> dict[str, Any]:
        try:
            with self.factory() as session:
                return propose_public_message(
                    session,
                    self.location_id,
                    audience=str(args.get("audience", "")),
                    channel=str(args.get("channel", "")),
                    message=str(args.get("message", "")),
                    language=str(args.get("language") or "English"),
                    proposed_by=f"{OUTREACH} Agent",
                    agent_run_id=self.run_id,
                )
        except ProposalError as exc:
            # Returned to the model so it can correct the draft.
            return {"error": exc.detail}

    def tools(self, agent: str) -> list[AgentTool]:
        read = {
            "get_surveillance_signals": (
                "Surveillance signals for this location: cases vs the usual for this time of year, "
                "4-week trend vs the usual seasonal trend, rainfall anomaly, temperature, verified lab positives.",
                self._read(context.surveillance_signals),
            ),
            "get_risk_forecast": (
                "The prediction engine's 1-4 week case forecasts, alert threshold, risk score and early-warning status.",
                self._read(context.risk_forecast),
            ),
            "get_verified_lab_summary": (
                "Reviewer-verified blood-smear results for this location (pending results are not evidence).",
                self._read(context.lab_summary),
            ),
            "get_open_alerts": ("Open alerts for this location from all modules.", self._read(context.open_alerts)),
            "get_supply_shortages": (
                "Facilities in this location at risk of running out of RDT kits, ACTs or injectable artesunate.",
                self._read(context.supply_shortages),
            ),
            "get_redistribution_plan": (
                "The deterministic redistribution plan (transfers from facilities with spare stock, central orders).",
                self._read(context.redistribution_plan),
            ),
            "get_situation_snapshot": (
                "Compact overview: surveillance, forecast, verified lab results, shortages, open alerts, pending proposals.",
                self._read(context.situation_snapshot),
            ),
            "get_outreach_context": (
                "Risk level and approved key prevention messages for drafting public communication.",
                self.get_outreach_context,
            ),
        }
        actions = {
            "propose_redistribution": (
                "Put the current redistribution plan into the human approval queue. Takes no quantities: "
                "they come from the deterministic plan.",
                self.propose_redistribution,
                _NO_ARGS,
            ),
            "propose_public_message": (
                "Submit ONE draft public prevention message for human approval. Nothing is sent until approved.",
                self.propose_public_message,
                {
                    "type": "object",
                    "properties": {
                        "audience": {"type": "string", "description": "Who the message is for."},
                        "channel": {"type": "string", "enum": list(MESSAGE_CHANNELS)},
                        "message": {"type": "string", "description": "The message text, 40-480 characters."},
                        "language": {"type": "string", "description": "Language of the message, e.g. English."},
                    },
                    "required": ["audience", "channel", "message"],
                },
            ),
        }
        names = {
            SURVEILLANCE: ["get_surveillance_signals", "get_risk_forecast", "get_verified_lab_summary", "get_open_alerts"],
            SUPPLY: ["get_supply_shortages", "get_redistribution_plan", "propose_redistribution"],
            OUTREACH: ["get_outreach_context", "propose_public_message"],
            POLICY: ["get_situation_snapshot", "get_open_alerts"],
            COMMANDER: ["get_situation_snapshot"],
        }[agent]

        result = []
        for name in names:
            if name in read:
                description, handler = read[name]
                result.append(AgentTool(ToolSpec(name, description, _NO_ARGS), handler))
            else:
                description, handler, schema = actions[name]
                result.append(AgentTool(ToolSpec(name, description, schema), handler))

        if agent == COMMANDER:
            for specialist in (SURVEILLANCE, SUPPLY, OUTREACH, POLICY):
                result.append(
                    AgentTool(
                        ToolSpec(
                            f"delegate_to_{specialist.lower()}",
                            f"Hand a task to the {specialist} Agent and receive its report.",
                            _TASK_ARGS,
                        ),
                        self._delegate(specialist),
                    )
                )
        return result

    # ----------------------------------------------------------- delegation

    def _delegate(self, specialist: str) -> Callable[[dict], Any]:
        def handler(args: dict) -> dict[str, Any]:
            task = str(args.get("task") or f"Report on {self.location_name}.")[:1000]
            self.logger.log(COMMANDER, AgentMessageKind.HANDOFF, task, payload={"to": specialist})
            answer = self.run_agent(specialist, task)
            self.logger.log(specialist, AgentMessageKind.HANDOFF, answer, payload={"to": COMMANDER})
            return {"agent": specialist, "report": answer}

        return handler

    def run_agent(self, agent: str, task: str) -> str:
        assert self.client is not None, "Live agents need an LLM client."
        system = COMMON_RULES + "\n\n" + PROMPTS[agent].format(location=self.location_name)
        result = run_tool_loop(
            self.client,
            system,
            task,
            self.tools(agent),
            required_tools=REQUIRED[agent],
            max_rounds=10 if agent == COMMANDER else 6,
            on_event=self.logger.loop_hook(agent),
            max_tokens=AGENT_MAX_TOKENS,
            exhausted_detail=f"The {agent} Agent could not finish within the allowed rounds.",
        )
        return result.answer

    def run_live(self) -> str:
        return self.run_agent(
            COMMANDER,
            f"Coordinate the response for {self.location_name}. Start with the situation snapshot, "
            "then delegate to every specialist and summarise what needs human approval.",
        )
