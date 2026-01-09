import json
import os
import asyncio
import websockets
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from models import HospitalState, SeverityResult, TreatmentPlan
from pydantic import ValidationError

URGENT_DOCTOR_WS_URL = os.environ.get("URGENT_DOCTOR_WS_URL", "ws://doctor-urgent:8000/ws")
REGULAR_DOCTOR_WS_URL = os.environ.get("REGULAR_DOCTOR_WS_URL", "ws://doctor-regular:8000/ws")

ORCH_BASE = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("LLM_GATEWAY_URL"),
)
ORCH_TRIAGE = ORCH_BASE.with_structured_output(SeverityResult)


async def classify_severity(state: HospitalState) -> HospitalState:
    print("-> Triage Nurse: Reviewing patient symptoms...")
    
    issue = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a hospital intake nurse.\n"
                "Classify severity as urgent vs regular.\n"
                "Urgent if any red-flag symptoms are present (e.g., chest pain, stroke signs, "
                "severe shortness of breath, severe bleeding, fainting, confusion, severe allergic reaction).\n"
                "Return only the structured severity field."
            ),
        },
        {"role": "user", "content": issue},
    ]

    result: SeverityResult = await ORCH_TRIAGE.ainvoke(messages)
    return {
        "severity": result.severity,
        "messages": [AIMessage(content=f"Triage severity = {result.severity}")],
    }


def route(state: HospitalState) -> Literal["call_urgent", "call_regular"]:
    if state.get("severity") == "urgent":
        return "call_urgent"
    return "call_regular"


async def call_agent_over_ws(ws_url: str, issue: str) -> TreatmentPlan:
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"issue": issue}, ensure_ascii=False))
            raw = await ws.recv()
            data = json.loads(raw)

        # Validate WS response with Pydantic
        return TreatmentPlan.model_validate(data["plan"])
    except (KeyError, ValidationError, OSError) as e:
        # Fallback if remote server misbehaves or is unreachable
        print(f"Error calling agent at {ws_url}: {e}")
        return TreatmentPlan(
            urgency="urgent", # Default to urgent on system failure
            summary=f"System Error: Unable to reach specialist ({ws_url}). Please seek professional medical evaluation immediately.",
            home_care=["Seek professional evaluation."],
            red_flags=["Worsening symptoms", "Trouble breathing", "Fainting"],
            what_to_tell_clinician=[
                "Main symptoms",
                "When it started",
                "Medical history/meds/allergies",
            ],
            disclaimer=f"General information only; not a diagnosis. (Error: {str(e)})",
        )


async def urgency_doctor_node(state: HospitalState) -> HospitalState:
    issue = next(m.content for m in state["messages"] if isinstance(m, HumanMessage))
    plan = await call_agent_over_ws(URGENT_DOCTOR_WS_URL, issue)
    return {
        "plan": plan,
        "messages": [AIMessage(content="Urgency doctor plan received.")],
    }


async def regular_doctor_node(state: HospitalState) -> HospitalState:
    issue = next(m.content for m in state["messages"] if isinstance(m, HumanMessage))
    plan = await call_agent_over_ws(REGULAR_DOCTOR_WS_URL, issue)
    return {
        "plan": plan,
        "messages": [AIMessage(content="Regular doctor plan received.")],
    }


async def finalize(state: HospitalState) -> HospitalState:
    plan = state.get("plan")
    if not plan:
        return {
            "messages": [
                AIMessage(content="No plan generated. Please seek professional help.")
            ]
        }

    lines: list[str] = []
    lines.append(f"**Urgency:** {plan.urgency}")
    lines.append(f"**Summary:** {plan.summary}\n")

    if plan.likely_causes:
        lines.append("**Possible causes (not a diagnosis):**")
        lines += [f"- {x}" for x in plan.likely_causes]
        lines.append("")

    lines.append("**What you can do now (general treatments/home care):**")
    lines += [f"- {x}" for x in plan.home_care]
    lines.append("")

    if plan.otc_options:
        lines.append("**OTC options (no dosing):**")
        lines += [f"- {x}" for x in plan.otc_options]
        lines.append("")

    if plan.what_to_avoid:
        lines.append("**What to avoid:**")
        lines += [f"- {x}" for x in plan.what_to_avoid]
        lines.append("")

    lines.append("**Seek urgent/emergency care if:**")
    lines += [f"- {x}" for x in plan.red_flags]
    lines.append("")

    lines.append("**What to tell a clinician:**")
    lines += [f"- {x}" for x in plan.what_to_tell_clinician]
    lines.append("")
    lines.append(plan.disclaimer)

    return {"messages": [AIMessage(content="\n".join(lines))]}


def build_graph():
    g = StateGraph(HospitalState)
    g.add_node("classify", classify_severity)
    g.add_node("call_urgent", urgency_doctor_node)
    g.add_node("call_regular", regular_doctor_node)
    g.add_node("finalize", finalize)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route)
    g.add_edge("call_urgent", "finalize")
    g.add_edge("call_regular", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
