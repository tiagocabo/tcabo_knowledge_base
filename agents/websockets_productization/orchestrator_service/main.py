from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from graph import build_graph
from models import HospitalState

app = FastAPI()

graph = build_graph()

class SymptomInput(BaseModel):
    symptoms: str

class TriageResponse(BaseModel):
    response: str
    plan: dict | None

@app.post("/triage", response_model=TriageResponse)
async def triage_symptoms(input_data: SymptomInput):
    inputs: HospitalState = {
        "messages": [HumanMessage(content=input_data.symptoms)],
        "severity": None,
        "plan": None,
    }
    
    try:
        result = await graph.ainvoke(inputs)
        final_message = result["messages"][-1].content
        plan = result.get("plan")
        
        return TriageResponse(
            response=final_message,
            plan=plan.model_dump() if plan else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
