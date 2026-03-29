from pydantic import BaseModel, Field
from typing import List, Optional

# --- Agent Output Models (Used to strictly type LLM outputs) ---

class JobRequirements(BaseModel):
    core_necessities: List[str] = Field(description="Must-have skills and experience levels")
    soft_skills: List[str] = Field(description="Required interpersonal and communication skills")
    experience_metrics: str = Field(description="Summary of required years of experience or domain expertise")

class CandidateKPIs(BaseModel):
    speed: int = Field(description="Score 1-10. How fast can they start working based on context?")
    speed_rationale: str = Field(description="Why this speed score was given.")
    fit: int = Field(description="Score 1-10. How closely they match the JD.")
    fit_rationale: str = Field(description="Why this fit score was given.")
    risk: int = Field(description="Score 1-10 (Lower score = lower risk). Risk factors of hiring this person.")
    risk_rationale: str = Field(description="Why this risk score was given, and identifying risks.")
    impact: int = Field(description="Score 1-10. Past impact and predicted success.")
    impact_rationale: str = Field(description="Explanation of their predicted impact.")

class ProfiledCandidate(BaseModel):
    candidate_id: str
    name: str = Field(description="Extracted candidate name")
    is_internal: bool
    kpis: CandidateKPIs
    overall_summary: str = Field(description="A balanced, empathetic summary of the candidate's holistic profile.")

class FinalRecommendation(BaseModel):
    recommended_candidate_id: str = Field(description="The ID of the best candidate for the scenario")
    justification: str = Field(description="Pragmatic, cautious explanation of why this candidate was chosen based on the scenario")
    trade_offs: str = Field(description="What are the actual trade-offs of this decision compared to the runner-up?")
    candidate_rankings: List[str] = Field(description="Ordered list of candidate IDs from best to worst fit for scenario")

# --- API Response Models ---

class EvaluationResponse(BaseModel):
    scenario: str
    job_requirements: JobRequirements
    evaluated_candidates: List[ProfiledCandidate]
    final_decision: FinalRecommendation
