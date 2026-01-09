"""
LangGraph + WebSockets + OpenAI, upgraded to *Pydantic structured outputs* end-to-end.

- Two downstream WS "doctor agents" (urgent / regular) produce a TreatmentPlan validated by Pydantic.
- Orchestrator (LangGraph) triages severity and routes to the correct WS agent.
- Final node formats the validated plan for a user-facing response.

pip install langgraph langchain-openai langchain-core websockets pydantic python-dotenv

Set:
  export OPENAI_API_KEY="..."
  export LLM_GATEWAY_URL="..."

Run:
  uv run hospital_ws_langgraph_pydantic.py
"""

import asyncio
import json
import os
from typing import Annotated, Literal, Optional, TypedDict

import websockets
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


# ---------------------------
# 1) Pydantic models (structured outputs)
# ---------------------------
class TreatmentPlan(BaseModel):
    urgency: Literal["urgent", "regular"] = Field(
        description="Overall urgency classification."
    )
    summary: str = Field(
        description="Short summary of situation and recommended next step."
    )
    likely_causes: list[str] = Field(
        default_factory=list,
        description="General possibilities (not a diagnosis).",
        max_length=8,
    )
    home_care: list[str] = Field(
        default_factory=list,
        description="Safe, general treatments / home care steps.",
        min_length=3,
        max_length=12,
    )
    otc_options: list[str] = Field(
        default_factory=list,
        description="Common OTC options (NO dosing).",
        max_length=8,
    )
    what_to_avoid: list[str] = Field(
        default_factory=list,
        description="Actions/substances to avoid.",
        max_length=8,
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description="Symptoms/signs that require urgent/emergency care.",
        min_length=3,
        max_length=12,
    )
    what_to_tell_clinician: list[str] = Field(
        default_factory=list,
        description="Key information to share with clinician.",
        min_length=3,
        max_length=10,
    )
    disclaimer: str = Field(
        default="General information only; not a diagnosis or medical advice.",
        description="Safety disclaimer.",
    )


class SeverityResult(BaseModel):
    severity: Literal["urgent", "regular"]


# ---------------------------
# 2) LangGraph state
# ---------------------------
class HospitalState(TypedDict):
    messages: Annotated[list, add_messages]
    severity: Optional[Literal["urgent", "regular"]]
    plan: Optional[TreatmentPlan]


# ---------------------------
# 3) Downstream agents (WS servers) producing Pydantic outputs
# ---------------------------
def doctor_system_prompt(doctor_type: Literal["urgent", "regular"]) -> str:
    base = (
        "You are a clinician providing GENERAL medical information.\n"
        "Be conservative and safe.\n"
        "Do NOT diagnose.\n"
        "Do NOT recommend prescription meds.\n"
        "Do NOT provide dosing.\n"
        "Always include red flags and escalation guidance.\n"
        "Return output that conforms to the provided schema.\n"
    )
    return base + (
        "Role: Emergency/urgent care clinician.\n"
        if doctor_type == "urgent"
        else "Role: Primary care clinician.\n"
    )


def doctor_user_prompt(
    issue: str, default_urgency: Literal["urgent", "regular"]
) -> str:
    return (
        f"Patient issue: {issue}\n\n"
        "Create a treatment-oriented plan.\n"
        "- Provide actionable, safe home care steps.\n"
        "- Provide OTC options as categories/names only (NO dosing).\n"
        "- Provide red flags and when to seek urgent/emergency care.\n"
        f"- Default urgency: {default_urgency} (override only if safety demands).\n"
    )


async def doctor_agent_ws_server(
    host: str, port: int, doctor_type: Literal["urgent", "regular"]
) -> None:
    """
    WS request:  {"issue": "..."}
    WS response: {"doctor_type": "...", "plan": <TreatmentPlan dict>}
    """
    base_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("LLM_GATEWAY_URL"),
    )
    llm = base_llm.with_structured_output(TreatmentPlan)

    async def handler(ws):
        async for raw in ws:
            req = json.loads(raw)
            print(f"-> [{doctor_type} doctor] Patient data received. Analyzing...")
            await asyncio.sleep(3)
            issue = req.get("issue", "")

            messages = [
                {"role": "system", "content": doctor_system_prompt(doctor_type)},
                {"role": "user", "content": doctor_user_prompt(issue, doctor_type)},
            ]

            try:
                # Returns a Pydantic TreatmentPlan (LangChain handles parsing)
                plan: TreatmentPlan = await llm.ainvoke(messages)

            except Exception:
                # Very defensive fallback: still return a valid TreatmentPlan
                plan = TreatmentPlan(
                    urgency=doctor_type,
                    summary="Unable to generate a structured plan. Please seek professional medical evaluation.",
                    likely_causes=[],
                    home_care=["Seek professional evaluation as soon as possible."],
                    otc_options=[],
                    what_to_avoid=[],
                    red_flags=["Worsening symptoms", "Trouble breathing", "Fainting"],
                    what_to_tell_clinician=[
                        "Main symptoms",
                        "When it started",
                        "Any medical history/meds/allergies",
                    ],
                    disclaimer="General information only; not a diagnosis or medical advice.",
                )

            await ws.send(
                json.dumps(
                    {"doctor_type": doctor_type, "plan": plan.model_dump()},
                    ensure_ascii=False,
                )
            )

    async with websockets.serve(handler, host, port):
        print(f"[{doctor_type} doctor] ws://{host}:{port} ready")
        await asyncio.Future()  # run forever


# ---------------------------
# 4) Orchestrator graph nodes (Pydantic structured outputs too)
# ---------------------------
ORCH_BASE = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("LLM_GATEWAY_URL"),
)
ORCH_TRIAGE = ORCH_BASE.with_structured_output(SeverityResult)


async def classify_severity(state: HospitalState) -> HospitalState:
    print("-> Triage Nurse: Reviewing patient symptoms...")
    await asyncio.sleep(2)
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
    }  # log


def route(state: HospitalState) -> Literal["call_urgent", "call_regular"]:
    return "call_urgent" if state.get("severity") == "urgent" else "call_regular"


async def call_agent_over_ws(ws_url: str, issue: str) -> TreatmentPlan:
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"issue": issue}, ensure_ascii=False))
        raw = await ws.recv()
        data = json.loads(raw)

    # Validate WS response with Pydantic
    try:
        return TreatmentPlan.model_validate(data["plan"])
    except (KeyError, ValidationError) as e:
        # Fallback if remote server misbehaves
        return TreatmentPlan(
            urgency="urgent",
            summary="Unable to validate the returned plan. Please seek professional medical evaluation.",
            home_care=["Seek professional evaluation."],
            red_flags=["Worsening symptoms", "Trouble breathing", "Fainting"],
            what_to_tell_clinician=[
                "Main symptoms",
                "When it started",
                "Medical history/meds/allergies",
            ],
            disclaimer=f"General information only; not a diagnosis or medical advice. (validation error: {e})",
        )


async def urgency_doctor_node(state: HospitalState) -> HospitalState:
    issue = next(m.content for m in state["messages"] if isinstance(m, HumanMessage))
    plan = await call_agent_over_ws("ws://127.0.0.1:8765", issue)
    return {
        "plan": plan,
        "messages": [AIMessage(content="Urgency doctor plan received.")],
    }  # metadata


async def regular_doctor_node(state: HospitalState) -> HospitalState:
    issue = next(m.content for m in state["messages"] if isinstance(m, HumanMessage))
    plan = await call_agent_over_ws("ws://127.0.0.1:8766", issue)
    return {
        "plan": plan,
        "messages": [AIMessage(content="Regular doctor plan received.")],
    }  # metadata


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


# ---------------------------
# 5) Build LangGraph
# ---------------------------
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


# ---------------------------
# 6) Demo runner
# ---------------------------
async def main():
    # Start downstream agent servers
    servers = [
        asyncio.create_task(doctor_agent_ws_server("127.0.0.1", 8765, "urgent")),
        asyncio.create_task(doctor_agent_ws_server("127.0.0.1", 8766, "regular")),
    ]
    await asyncio.sleep(0.2)

    graph = build_graph()

    # save langgraph visualization as png
    image_data = graph.get_graph().draw_mermaid_png()

    # 2. Write the binary data to a file
    with open("agents/graph_output.png", "wb") as f:
        f.write(image_data)

    print("\nDescribe your symptoms:")
    user_issue = await asyncio.to_thread(input, "> ")

    if not user_issue.strip():
        user_issue = (
            "I have severe chest pain radiating to my left arm and I feel dizzy."
        )
        print(f"No input provided. Using default: {user_issue}")

    inputs: HospitalState = {
        "messages": [HumanMessage(content=user_issue)],
        "severity": None,
        "plan": None,
    }
    result = await graph.ainvoke(inputs)

    print("\n=== FINAL RESPONSE ===\n")
    print(result["messages"][-1].content)

    for t in servers:
        t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
