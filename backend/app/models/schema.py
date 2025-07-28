from pydantic import BaseModel
from typing import List
from typing import Dict

class QueryRequest(BaseModel):
    documents: str  # Single URL string
    questions: List[str]

    class Config:
        schema_extra = {
            "example": {
                "query": "Does the policy cover mental health treatment?"
            }
        }

class QueryResponse(BaseModel):
    answers: List[str]

    class Config:
        schema_extra = {
            "example": {
                "answers": [
                    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
                ]
            }
        }
