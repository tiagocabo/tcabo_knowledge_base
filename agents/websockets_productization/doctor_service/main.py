import json
import os
import asyncio
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from models import TreatmentPlan

load_dotenv()

app = FastAPI()

DOCTOR_TYPE = os.environ.get("DOCTOR_TYPE", "regular")

def doctor_system_prompt(doctor_type: str) -> str:
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

def doctor_user_prompt(issue: str, default_urgency: str) -> str:
    return (
        f"Patient issue: {issue}\n\n"
        "Create a treatment-oriented plan.\n"
        "- Provide actionable, safe home care steps.\n"
        "- Provide OTC options as categories/names only (NO dosing).\n"
        "- Provide red flags and when to seek urgent/emergency care.\n"
        f"- Default urgency: {default_urgency} (override only if safety demands).\n"
    )

# Initialize LLM
base_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("LLM_GATEWAY_URL"),
)
llm = base_llm.with_structured_output(TreatmentPlan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"[{DOCTOR_TYPE} doctor] Client connected.")
    try:
        while True:
            raw = await websocket.receive_text()
            req = json.loads(raw)
            print(f"-> [{DOCTOR_TYPE} doctor] Patient data received. Analyzing...")
            
            # Simulate work
            await asyncio.sleep(1) 
            
            issue = req.get("issue", "")
            messages = [
                {"role": "system", "content": doctor_system_prompt(DOCTOR_TYPE)},
                {"role": "user", "content": doctor_user_prompt(issue, DOCTOR_TYPE)},
            ]

            try:
                plan: TreatmentPlan = await llm.ainvoke(messages)
            except Exception as e:
                print(f"Error generating plan: {e}")
                # Defensive fallback
                plan = TreatmentPlan(
                    urgency=DOCTOR_TYPE, # type: ignore
                    summary="Unable to generate a structured plan. Please seek professional medical evaluation.",
                    likely_causes=[],
                    home_care=["Seek professional evaluation as soon as possible."],
                    otc_options=[],
                    what_to_avoid=[],
                    red_flags=["Worsening symptoms", "Trouble breathing", "Fainting"],
                    what_to_tell_clinician=[
                        "Main symptoms", 
                        "When it started", 
                        "Any medical history/meds/allergies"
                    ],
                    disclaimer="General information only; not a diagnosis or medical advice.",
                )

            response = {
                "doctor_type": DOCTOR_TYPE,
                "plan": plan.model_dump()
            }
            await websocket.send_text(json.dumps(response, ensure_ascii=False))
            
    except WebSocketDisconnect:
        print(f"[{DOCTOR_TYPE} doctor] Client disconnected")
