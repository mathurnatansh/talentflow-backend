import os
import json
import logging
from google import genai
from google.genai import types
from models import JobRequirements, ProfiledCandidate, FinalRecommendation, CandidateKPIs

# Configure logging for better debugging in Render dashboard
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy initialization of the client to avoid crashing at import time
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.error("CRITICAL: GEMINI_API_KEY is not set in environment!")
            # Fallback to default if absolutely necessary, but this might crash if no ADC
            _client = genai.Client()
        else:
            # We use the key explicitly. We don't force 'v1' here unless flash fails.
            _client = genai.Client(api_key=api_key)
    return _client

def safe_generate_content(prompt: str, system_instruction: str, response_schema):
    """
    Indestructible generation helper with automated Flash -> Pro fallback.
    """
    client = get_client()
    
    config_flash = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=0.1
    )
    
    try:
        # Attempt 1: Fast Flash (using v1beta internally)
        logger.info("Attempting evaluation with gemini-1.5-flash...")
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
            config=config_flash,
        )
        return response.text
    except Exception as e:
        logger.warning(f"Flash model failed or not found: {e}. Falling back to Gemini Pro...")
        
        # Attempt 2: Stable Pro
        # We explicitly use gemini-1.5-pro which is highly stable for structured schemas.
        config_pro = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.1
        )
        
        try:
            response = client.models.generate_content(
                model='gemini-1.5-pro',
                contents=prompt,
                config=config_pro,
            )
            return response.text
        except Exception as e2:
            logger.error(f"Critical AI Failure: Both Flash and Pro failed: {e2}")
            raise RuntimeError(f"AI evaluation failed on both models. Error: {e2}")

def parse_job_description(jd_text: str) -> JobRequirements:
    """
    Agent 1: Job Parsing Agent
    """
    system_instruction = (
        "You are the Job Parsing Agent. Extract skills, necessities, and experience from the JD."
    )
    prompt = f"Extract requirements from:\n\n{jd_text}"
    
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
        "You are the Candidate Profiling Agent. Evaluate metrics for Speed, Fit, Risk, and Impact."
    )
    status = "Internal" if is_internal else "External"
    prompt = f"Evaluate {status} candidate vs JD: {jd.model_dump_json()}\n\nCV: {cv_text}"
    
    result_text = safe_generate_content(prompt, system_instruction, ProfiledCandidate)
    parsed = ProfiledCandidate.model_validate_json(result_text)
    parsed.candidate_id = candidate_id
    parsed.is_internal = is_internal
    return parsed

def scenario_evaluation(scenario_type: str, candidates: list[ProfiledCandidate]) -> FinalRecommendation:
    """
    Agent 3: Scenario Agent
    """
    system_instruction = "You are the Scenario Agent. Apply mathematical +5% scenario modifiers."
    
    if "urgen" in scenario_type.lower():
        scenario_rules = "URGENT HIRE: Apply +5% to Internal candidates."
    elif "transform" in scenario_type.lower():
        scenario_rules = "TRANSFORMATION: Apply +5% to External candidates. Sort by Fit/Impact."
    else:
        scenario_rules = "STRATEGIC: Pure evaluation."

    prompt = f"Scenario: {scenario_rules}\nCandidates: {[c.model_dump() for c in candidates]}"
    
    result_text = safe_generate_content(prompt, system_instruction, FinalRecommendation)
    return FinalRecommendation.model_validate_json(result_text)
