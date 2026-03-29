import os
from google import genai
from google.genai import types
from models import JobRequirements, ProfiledCandidate, FinalRecommendation, CandidateKPIs
import json

# Ensure you have GEMINI_API_KEY in your env
client = genai.Client()

def parse_job_description(jd_text: str) -> JobRequirements:
    """
    Agent 1: Job Parsing Agent
    Persona: Calm, analytical, methodical librarian.
    """
    system_instruction = (
        "You are the Job Parsing Agent. You are meticulous and highly detail-oriented, like a diligent "
        "researcher or librarian. Your personality is calm, analytical, and methodical, ensuring that no "
        "requirement or nuance is overlooked when transforming job descriptions into structured requirements."
    )
    
    prompt = f"Please extract the core necessities, soft skills, and experience metrics from the following Job Description:\n\n{jd_text}"
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=JobRequirements
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=prompt,
        config=config,
    )
    
    # generate_content with a response_schema typically returns the raw text as JSON which we then parse.
    return JobRequirements.model_validate_json(response.text)


def profile_candidate(
    candidate_id: str,
    jd: JobRequirements, 
    cv_text: str, 
    is_internal: bool
) -> ProfiledCandidate:
    """
    Agent 2: Candidate Profiling Agent
    Persona: Balanced, empathetic thoughtful career advisor.
    """
    system_instruction = (
        "You are the Candidate Profiling Agent. You are balanced, empathetic, and insightful, like a thoughtful career advisor. "
        "You see the whole person behind a resume, weighing strengths, risks, and potential with fairness. "
        "Your personality is supportive yet objective, offering a balanced view on who might fit best. "
        "Rate the candidate heavily on four KPIs: Speed (start time), Fit (match to JD), Risk (internal/external factors), Impact (historical & predicted)."
    )
    
    status = "Internal" if is_internal else "External"
    prompt = f"""
    Evaluate the following {status} candidate against the structured Job Requirements.
    Remember that Internal candidates generally have better 'Speed' to start, lower onboarding 'Risk', but might lack 'Impact' of fresh perspectives.
    External candidates might have high 'Risk' but high 'Impact' depending on their CV.
    
    **Job Requirements**:
    {jd.model_dump_json(indent=2)}
    
    **Candidate Status**: {status}
    
    **Candidate CV Text**:
    {cv_text}
    
    Extract their name and provide the full profile evaluation.
    """
    
    # We must construct a schema that includes candidate_id, since the model won't have it, 
    # but the simplest way is to ask the model just to return name, KPIs, etc., and then we add ID.
    # To keep it strict, we define a small wrapper or just instruct it.
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=ProfiledCandidate
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=prompt,
        config=config,
    )
    
    parsed = ProfiledCandidate.model_validate_json(response.text)
    # Ensure ID is correct
    parsed.candidate_id = candidate_id
    parsed.is_internal = is_internal
    return parsed

def scenario_evaluation(scenario_type: str, candidates: list[ProfiledCandidate]) -> FinalRecommendation:
    """
    Agent 3: Scenario Agent
    Persona: Cautious, pragmatic, protective team steward.
    """
    system_instruction = (
        "You are the Scenario Agent. You are cautious and pragmatic, almost like a protective team steward. "
        "Your personality leans conservative, prioritizing stability and internal continuity unless external options show clear, long-term value. "
        "You are skeptical of unnecessary change, ensuring decisions favor team balance and risk mitigation."
    )
    
    # Detailed scenario constraints
    scenario_rules = ""
    if "urgen" in scenario_type.lower():  
        # Handles user typo 'urgen' vs 'urgent'
        scenario_rules = "URGENT HIRE: Internal person gets high preference. External will only be recommended if they are extremely good on other parameters."
    elif "transform" in scenario_type.lower():
        scenario_rules = "TRANSFORMATION: Hiring for a fresh perspective. Give preference to external candidates. Recommend internal only if parameters are exceptionally good."
    else:
        # Defaulting to Strategic essentially
        scenario_rules = "STRATEGIC: The candidate recommended should be a perfect fit on the metrics of IMPACT and FIT, regardless of internal/external."

    prompt = f"""
    You have {len(candidates)} profiled candidates.
    Evaluate them based on this specific Scenario: {scenario_rules}
    
    **Candidate Profiles**:
    {[c.model_dump() for c in candidates]}
    
    Make your final recommendation, ranking them, and detailing trade-offs.
    Return ONLY a single valid JSON matching the requested format.
    """
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=FinalRecommendation,
        temperature=0.2 # Lower temperature for pragmatic logic
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=prompt,
        config=config,
    )
    
    return FinalRecommendation.model_validate_json(response.text)
