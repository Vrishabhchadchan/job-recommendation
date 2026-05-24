from pydantic import BaseModel, Field
from typing import List, Optional

# API Request/Response Schemas
class RecommendRequest(BaseModel):
    resume_text: str

class CandidateSchema(BaseModel):
    name: str = "Unknown"
    skills: List[str] = []
    experience_years: float = 0.0
    preferred_roles: List[str] = []
    education: str = "Unknown"

class RankedJobSchema(BaseModel):
    id: int
    title: str
    company: str
    location: Optional[str] = None
    is_remote: Optional[bool] = None
    apply_link: Optional[str] = None
    similarity_score: float
    explanation: str

class RecommendResponse(BaseModel):
    candidate: CandidateSchema
    ranked_jobs: List[RankedJobSchema]
    clarifying_question: str

class RefineRequest(BaseModel):
    resume_text: str
    clarifying_question: str
    candidate_answer: str

class RefineResponse(BaseModel):
    ranked_jobs: List[RankedJobSchema]
    reasoning: str
