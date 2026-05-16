import os
import json
import requests
import random
from groq import Groq
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from .schemas import CandidateSchema, RankedJobSchema

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# HF sentence-similarity model (all-MiniLM-L6-v2 via the router serverless API)
_HF_MODEL_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

_jobs_path = os.path.join(os.path.dirname(__file__), "jobs.json")
with open(_jobs_path, "r") as _f:
    JOBS: List[Dict[str, Any]] = json.load(_f)

# Pre-build job text representations once at import time
_JOB_TEXTS: List[str] = [
    (
        f"{j['title']} at {j['company']}. Domain: {j['domain']}. "
        f"{j['description']} "
        f"Required skills: {', '.join(j['skills'])}. "
        f"Experience needed: {j['experience_years']} years."
    )
    for j in JOBS
]


# ─── Similarity helpers ───────────────────────────────────────────────────────

def _get_similarity_scores(resume_text: str, job_texts: List[str]) -> List[float]:
    """
    Compute cosine similarity between resume_text and every job text using
    the HuggingFace sentence-similarity pipeline (one batched API call).

    Falls back to deterministic mock scores when HF_API_TOKEN is not set.
    """
    if not HF_API_TOKEN:
        # Deterministic mock: score ~ string overlap fraction, seeded for reproducibility
        scores = []
        resume_words = set(resume_text.lower().split())
        for jt in job_texts:
            job_words = set(jt.lower().split())
            overlap = len(resume_words & job_words)
            base = overlap / max(len(resume_words | job_words), 1)
            # Add tiny per-job jitter so scores are always distinct
            seed = hash(resume_text + jt) % (2 ** 31)
            random.seed(seed)
            scores.append(min(base + random.uniform(0.0, 0.05), 1.0))
        return scores

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": {"source_sentence": resume_text, "sentences": job_texts},
        "options": {"wait_for_model": True},
    }

    try:
        resp = requests.post(_HF_MODEL_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()  # List[float] — one cosine score per job
    except Exception as exc:
        print(f"[HF Similarity Error] {exc} — falling back to mock scores")
        scores = []
        resume_words = set(resume_text.lower().split())
        for jt in job_texts:
            job_words = set(jt.lower().split())
            overlap = len(resume_words & job_words)
            base = overlap / max(len(resume_words | job_words), 1)
            seed = hash(resume_text + jt) % (2 ** 31)
            random.seed(seed)
            scores.append(min(base + random.uniform(0.0, 0.05), 1.0))
        return scores


def rank_jobs(search_text: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Return the top_n jobs most semantically similar to search_text.
    One batched HF API call scores all 50 jobs simultaneously.
    """
    scores = _get_similarity_scores(search_text, _JOB_TEXTS)
    ranked = sorted(
        zip(JOBS, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [
        {
            "id": j["id"],
            "title": j["title"],
            "company": j["company"],
            "location": j["location"],
            "remote": j["remote"],
            "domain": j["domain"],
            "skills": j["skills"],
            "experience_years": j["experience_years"],
            "salary_lpa": j["salary_lpa"],
            "description": j["description"],
            "similarity_score": round(s, 4),
        }
        for j, s in ranked[:top_n]
    ]


# ─── Groq tool definitions ────────────────────────────────────────────────────

_PARSE_RESUME_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_resume_info",
        "description": "Extract structured candidate information from raw resume text.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Candidate's full name",
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All technical and domain skills mentioned in the resume",
                },
                "experience_years": {
                    "type": "number",
                    "description": (
                        "Total years of professional work experience. "
                        "Use 0 for students or freshers."
                    ),
                },
                "preferred_roles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target job roles inferred from the resume content",
                },
                "education": {
                    "type": "string",
                    "description": "Highest education degree and institution name",
                },
            },
            "required": ["name", "skills", "experience_years", "preferred_roles", "education"],
        },
    },
}

_EXPLAIN_JOBS_TOOL = {
    "type": "function",
    "function": {
        "name": "provide_job_explanations",
        "description": (
            "For each matched job, write a 2-3 sentence explanation of fit. "
            "Also generate one smart follow-up clarifying question based on gaps "
            "or ambiguities noticed across the candidate and their matches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "explanations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "integer"},
                            "explanation": {
                                "type": "string",
                                "description": "2-3 sentence fit explanation covering skill overlap, gaps, and seniority",
                            },
                        },
                        "required": ["job_id", "explanation"],
                    },
                },
                "clarifying_question": {
                    "type": "string",
                    "description": (
                        "ONE specific, smart follow-up question to resolve a key ambiguity "
                        "or gap — e.g. remote preference, missing cloud skills, domain openness. "
                        "Must NOT be generic like 'tell me more about yourself'."
                    ),
                },
            },
            "required": ["explanations", "clarifying_question"],
        },
    },
}


# ─── Agentic functions ────────────────────────────────────────────────────────

def parse_resume_with_groq(resume_text: str) -> CandidateSchema:
    """
    Step 1 — Resume Parser Tool.
    Uses Groq native tool calling to extract structured fields from raw resume text.
    """
    if not groq_client:
        raise ValueError("GROQ_API_KEY is not configured.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert HR assistant. Extract all requested fields precisely "
                "from the resume text. If total experience is unclear, infer from "
                "internships, projects, or graduation year. Use 0 for freshers/students."
            ),
        },
        {"role": "user", "content": f"Parse this resume:\n\n{resume_text}"},
    ]

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=[_PARSE_RESUME_TOOL],
        tool_choice={"type": "function", "function": {"name": "extract_resume_info"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    return CandidateSchema(**args)


def generate_job_explanations(
    candidate: CandidateSchema,
    ranked_jobs: List[Dict[str, Any]],
) -> Tuple[List[RankedJobSchema], str]:
    """
    Step 2 — Match Reasoning Tool.
    Uses Groq native tool calling to explain each job match and produce a
    clarifying question in a single structured call.

    Returns: (list of ranked jobs with explanations, clarifying question string)
    """
    if not groq_client:
        raise ValueError("GROQ_API_KEY is not configured.")

    jobs_summary = [
        {
            "id": j["id"],
            "title": j["title"],
            "company": j["company"],
            "domain": j["domain"],
            "required_skills": j["skills"],
            "experience_needed_years": j["experience_years"],
            "remote": j["remote"],
            "description": j["description"][:300],
            "similarity_score": j["similarity_score"],
        }
        for j in ranked_jobs
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior technical recruiter. Analyse the candidate profile "
                "against each matched job. Be specific about skill overlaps and gaps. "
                "The clarifying question must target a real, observable gap or "
                "ambiguity — not a generic question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Candidate profile:\n{candidate.model_dump_json(indent=2)}\n\n"
                f"Top matched jobs:\n{json.dumps(jobs_summary, indent=2)}\n\n"
                "Call the tool with a 2-3 sentence explanation for every job "
                "and one targeted clarifying question."
            ),
        },
    ]

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=[_EXPLAIN_JOBS_TOOL],
        tool_choice={"type": "function", "function": {"name": "provide_job_explanations"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    explanation_map = {ex["job_id"]: ex["explanation"] for ex in args["explanations"]}
    clarifying_question = args.get("clarifying_question", "")

    final_jobs = [
        RankedJobSchema(
            id=j["id"],
            title=j["title"],
            company=j["company"],
            similarity_score=j["similarity_score"],
            explanation=explanation_map.get(j["id"], "Strong semantic match."),
        )
        for j in ranked_jobs
    ]

    return final_jobs, clarifying_question


def process_refinement(
    resume_text: str,
    question: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Bonus /refine endpoint.
    Re-ranks jobs by augmenting the resume with the candidate's clarifying answer,
    then explains why the ranking shifted.
    """
    if not groq_client:
        raise ValueError("GROQ_API_KEY is not configured.")

    # Fold the Q&A context into the query so the embedding shifts accordingly
    augmented_text = (
        f"{resume_text}\n\n"
        f"Recruiter question: {question}\n"
        f"Candidate answer: {answer}"
    )

    ranked_jobs_dicts = rank_jobs(augmented_text, top_n=5)
    candidate = parse_resume_with_groq(augmented_text)
    final_jobs, _ = generate_job_explanations(candidate, ranked_jobs_dicts)

    # Brief reasoning for why the ranking shifted
    jobs_snapshot = json.dumps(
        [
            {
                "id": j["id"],
                "title": j["title"],
                "similarity_score": j["similarity_score"],
            }
            for j in ranked_jobs_dicts
        ],
        indent=2,
    )
    reasoning_response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert technical recruiter."},
            {
                "role": "user",
                "content": (
                    f"The candidate answered a clarifying question:\n"
                    f"Q: {question}\nA: {answer}\n\n"
                    f"Updated job ranking:\n{jobs_snapshot}\n\n"
                    "In 2-3 sentences, explain how the candidate's answer "
                    "influenced the re-ranking and why the top results changed."
                ),
            },
        ],
    )
    reasoning = reasoning_response.choices[0].message.content.strip()

    return {"ranked_jobs": final_jobs, "reasoning": reasoning}
