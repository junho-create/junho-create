from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

class Suggestion(BaseModel):
    의료진명: Optional[str]
    진료과: Optional[str]
    환자의_구체적인_증상: Optional[List[str]] = Field(None, alias="환자의 구체적인 증상")
    이유: Optional[str]
    class Config:
        populate_by_name = True

class OutputSchema(BaseModel):
    patient_name: str
    patient_gender: str
    phone_num: str
    chat_start_date: str
    symptoms: List[str]
    visit_type: str
    preference_datetime: List[str]
    dept: str
    doctor_name: str
    top_k_suggestions: List[Suggestion]
    retrieval_evidence: List[str]

    @field_validator("top_k_suggestions")
    @classmethod
    def validate_k(cls, v):
        if not v or len(v) == 0:
            raise ValueError("top_k_suggestions must be non-empty")
        return v

def enforce_output(payload: dict) -> OutputSchema:
    return OutputSchema.model_validate(payload)
