import json
import uuid
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from utils.pdf_parser import extract_text_from_pdf
from models import EvaluationResponse, ProfiledCandidate, JobRequirements
from agents import parse_job_description, profile_candidate, scenario_evaluation

load_dotenv()

app = FastAPI(
    title="TalentFlow Backend API",
    description="Immersive backend evaluating candidates vs constraints",
    version="1.0.0"
)

# CORS configuration for Loving/Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "TalentFlow Backend running"}

@app.post("/api/evaluate", response_model=EvaluationResponse)
def evaluate_candidates(
    job_description_text: str = Form(..., description="The raw JD text string"),
    scenario: str = Form(..., description="E.g., urgent, transformation, strategic"),
    # `candidate_metadata` is a JSON array string matching the files, e.g. [{"is_internal": true}, {"is_internal": false}]
    candidate_metadata: str = Form(..., description="JSON array matching files"),
    files: List[UploadFile] = File(...),
):
    try:
        metadata_list = json.loads(candidate_metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="candidate_metadata must be a valid JSON string array")
    
    if len(metadata_list) != len(files):
        raise HTTPException(status_code=400, detail="Number of metadata items must match number of uploaded files")

    # 1. Parse JD
    print("[1/3] Agent 1 parsing Job Description...")
    parsed_jd = parse_job_description(job_description_text)
    print(f"Parsed JD core necessities: {parsed_jd.core_necessities}")

    # 2. Profile every candidate
    print("[2/3] Agent 2 Profiling Candidates...")
    profiled_candidates = []
    
    for i, file in enumerate(files):
        # Read the file bytes
        file_bytes = file.file.read()
        
        # Only parse PDF; if text format, handle differently. Assuming PDF.
        if file.filename.endswith(".pdf"):
            cv_text = extract_text_from_pdf(file_bytes)
        else:
            # Fallback for plain text CVs
            cv_text = file_bytes.decode('utf-8', errors='ignore')
            
        if not cv_text.strip():
            cv_text = "No readable text found in CV."
            
        is_internal = metadata_list[i].get("is_internal", False)
        cand_id = str(uuid.uuid4())[:8] # Short tracking ID
        
        print(f"Profiling {file.filename} (Internal: {is_internal})...")
        try:
            profile = profile_candidate(
                candidate_id=cand_id,
                jd=parsed_jd,
                cv_text=cv_text,
                is_internal=is_internal
            )
            profiled_candidates.append(profile)
        except Exception as e:
            print(f"Error profiling {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to profile candidate {file.filename}")
            
    # 3. Apply Scenario
    print("[3/3] Agent 3 applying scenario constraints...")
    try:
        final_recommendation = scenario_evaluation(
            scenario_type=scenario,
            candidates=profiled_candidates
        )
    except Exception as e:
        print(f"Error evaluating scenario: {e}")
        raise HTTPException(status_code=500, detail="Failed scenario evaluation.")
        
    # Return the aggregated payload
    print("Done! Returning results.")
    return EvaluationResponse(
        scenario=scenario,
        job_requirements=parsed_jd,
        evaluated_candidates=profiled_candidates,
        final_decision=final_recommendation
    )
