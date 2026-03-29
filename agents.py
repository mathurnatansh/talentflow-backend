import os
import json
from google import genai
from google.genai import types
from models import JobRequirements, ProfiledCandidate, FinalRecommendation, CandidateKPIs

# Fetching the key explicitly prevents the 30-second crash loop on Render
api_key = os.environ.get("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=api_key) if api_key else genai.Client()

def safe_generate_content(prompt: str, system_instruction: str, response_schema):
    """
    Helper to attempt gen-ai with a fallback to the pro model if flash fails.
    """
    config_flash = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=0.1
    )
    
    try:
        # Attempt 1: Fast Flash
        model_name = 'gemini-1.5-flash'
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config_flash,
        )
        return response.text
    except Exception as e:
        print(f"Flash model failed, falling back to Pro: {e}")
        # Attempt 2: Stable Pro (More resilient to certain identifier issues)
        config_pro = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.1
        )
        response = client.models.generate_content(
            model='gemini-1.5-pro',
            contents=prompt,
            config=config_pro,
        )
        return response.text

def parse_job_description(jd_text: str) -> JobRequirements:
    """
    Agent 1: Job Parsing Agent
    """
    system_instruction = (
        "You are the Job Parsing Agent. You are meticulous and highly detail-oriented. "
        "Transform job descriptions into structured requirements."
    )
    prompt = f"Please extract core necessities, soft skills, and experience metrics from this JD:\n\n{jd_text}"
    
    result_text = safe_generate_content(prompt, system_instruction, JobRequirements)
    return JobRequirements.model_validate_json(result_text)


def profile_candidate(
    candidate_id: str,
    jd: JobRequirements, 
    cv_text: str, 
    is_internal: bool
) -> ProfiledCandidate:
    """
    Agent 2: Candidate Profiling Agent
    """
    system_instruction = (
        "You are the Candidate Profiling Agent. Rate the candidate on Speed, Fit, Risk, and Impact."
    )
    
    status = "Internal" if is_internal else "External"
    prompt = f"""
    Evaluate the following {status} candidate:
    Job Requirements: {jd.model_dump_json()}
    
    Candidate CV Text:
    {cv_text}
    """
    
    result_text = safe_generate_content(prompt, system_instruction, ProfiledCandidate)
    parsed = ProfiledCandidate.model_validate_json(result_text)
    parsed.candidate_id = candidate_id
    parsed.is_internal = is_internal
    return parsed

def scenario_evaluation(scenario_type: str, candidates: list[ProfiledCandidate]) -> FinalRecommendation:
    """
    Agent 3: Scenario Agent
    """
    system_instruction = "You are the Scenario Agent. Apply mathematical fairness."
    
    if "urgen" in scenario_type.lower():  
        scenario_rules = "URGENT HIRE: +5% score bonus to INTERNAL candidates."
    elif "transform" in scenario_type.lower():
        scenario_rules = "TRANSFORMATION: +5% score bonus to EXTERNAL candidates."
    else:
        scenario_rules = "STRATEGIC GROWTH: Purely equal comparison."

    prompt = f"""
    Scenario: {scenario_rules}
    Candidates: {[c.model_dump() for c in candidates]}
    """
    
    result_text = safe_generate_content(prompt, system_instruction, FinalRecommendation)
    return FinalRecommendation.model_validate_json(result_text)
