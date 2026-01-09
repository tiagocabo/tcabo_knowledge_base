from typing import Literal
from pydantic import BaseModel, Field

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
