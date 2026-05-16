from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import RecommendRequest, RecommendResponse, RefineRequest, RefineResponse
from .dependencies import (
    rank_jobs,
    parse_resume_with_groq,
    generate_job_explanations,
    process_refinement,
)

app = FastAPI(title="Smart Job Match Agent", description="AI-powered job recommendation backend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Smart Job Match Agent API is running"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend_jobs(request: RecommendRequest):
    if not request.resume_text or not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty")
        
    try:
        # Step 1: Extract candidate info using Groq Tool Calling
        candidate_info = parse_resume_with_groq(request.resume_text)
        
        # Step 2: Semantic ranking using ML layer
        # Include extracted candidate preferences to boost relevant job matches
        search_query = f"{' '.join(candidate_info.preferred_roles)} {' '.join(candidate_info.skills)} {request.resume_text}"
        top_jobs_dicts = rank_jobs(search_query, top_n=5)
        
        # Step 3: Match Reasoning Tool — returns explanations + clarifying question
        ranked_jobs, clarifying_question = generate_job_explanations(candidate_info, top_jobs_dicts)

        return RecommendResponse(
            candidate=candidate_info,
            ranked_jobs=ranked_jobs,
            clarifying_question=clarifying_question,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refine", response_model=RefineResponse)
def refine_jobs(request: RefineRequest):
    if not request.resume_text or not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty")
        
    try:
        result = process_refinement(
            request.resume_text, 
            request.clarifying_question, 
            request.candidate_answer
        )
        return RefineResponse(
            ranked_jobs=result["ranked_jobs"],
            reasoning=result["reasoning"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
