import os
import re
import json
import requests
import random
from groq import Groq
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from .schemas import CandidateSchema, RankedJobSchema

load_dotenv()

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
HF_API_TOKEN  = os.getenv("HF_API_TOKEN")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
RAPIDAPI_KEY  = os.getenv("RAPIDAPI_KEY")

# HF sentence-similarity model (all-MiniLM-L6-v2 via the router serverless API)
_HF_MODEL_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ─── Location extraction ───────────────────────────────────────────────────────

# Well-known locations for fallback matching (case-insensitive)
_KNOWN_LOCATIONS = [
    # Countries
    "India", "USA", "United States", "UK", "United Kingdom", "Canada",
    "Australia", "Germany", "France", "Netherlands", "Singapore", "Japan",
    "Dubai", "UAE", "Sweden", "Norway", "New Zealand", "Ireland",
    # Indian cities
    "Bangalore", "Bengaluru", "Mumbai", "Delhi", "New Delhi", "Hyderabad",
    "Pune", "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Gurgaon", "Noida",
    # US cities
    "New York", "San Francisco", "Seattle", "Austin", "Boston",
    "Chicago", "Los Angeles", "Denver", "Atlanta", "Washington",
    # Other cities
    "London", "Toronto", "Vancouver", "Sydney", "Melbourne", "Berlin",
    "Amsterdam", "Paris", "Dublin", "Tokyo", "Zurich",
]


def _extract_location(text: str) -> str:
    """
    Extract a location/city/country from a free-text refinement answer.
    Checks 'in/from/based in <place>' patterns first, then falls back to
    a known-locations list. Returns '' when nothing is found.
    """
    # Pattern: "jobs in India", "based in New York", "from Bangalore" etc.
    m = re.search(
        r'\b(?:in|from|at|based\s+in|located\s+in|only\s+in|prefer\s+in)\s+'
        r'([A-Za-z][A-Za-z\s]{1,30}?)(?=\s*(?:only|please|,|\.|$|\band\b|\bor\b))',
        text, re.IGNORECASE
    )
    if m:
        loc = m.group(1).strip()
        if len(loc) > 1:
            return loc.title()

    # Fallback: scan for any known location name
    for loc in _KNOWN_LOCATIONS:
        if re.search(r'\b' + re.escape(loc) + r'\b', text, re.IGNORECASE):
            return loc

    return ""

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


def _build_job_text(job: Dict[str, Any]) -> str:
    skills_str = ", ".join(job.get("skills") or []) or "Not specified"
    return (
        f"{job['title']} at {job['company']}. "
        f"{job.get('description', '')} "
        f"Required skills: {skills_str}. "
        f"Remote: {job.get('remote', False)}."
    )


def fetch_live_jobs(
    candidate: "CandidateSchema",
    num_pages: int = 3,
    location: str = "",
) -> List[Dict[str, Any]]:
    """
    Fetch real-time jobs from JSearch (aggregates LinkedIn, Indeed, Glassdoor).
    When `location` is provided it is passed as a JSearch filter so results are
    restricted to that country/city. Falls back to static jobs on API failure.
    """
    roles  = " OR ".join(candidate.preferred_roles[:2]) if candidate.preferred_roles else ""
    skills = " ".join(candidate.skills[:3]) if candidate.skills else ""
    query  = f"{roles} {skills}".strip() or "software engineer"

    params: Dict[str, str] = {
        "query":     query,
        "page":      "1",
        "num_pages": str(num_pages),
        "date_posted": "all",
    }
    if location:
        params["location"] = location
        print(f"[JSearch] Filtering by location: {location}")

    resp = requests.get(
        "https://jsearch.p.rapidapi.com/search",
        headers={
            "X-RapidAPI-Key":  RAPIDAPI_KEY,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
        },
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    raw_jobs = resp.json().get("data", [])

    normalized = []
    for i, job in enumerate(raw_jobs):
        city    = job.get("job_city")    or ""
        state   = job.get("job_state")   or ""
        country = job.get("job_country") or ""
        location = ", ".join(p for p in [city, state, country] if p)[:2]
        location = ", ".join([p for p in [city, country] if p]) or "Location not specified"

        normalized.append({
            "id":               i,
            "title":            job.get("job_title", ""),
            "company":          job.get("employer_name", ""),
            "location":         location,
            "remote":           bool(job.get("job_is_remote", False)),
            "domain":           "Technology",
            "skills":           job.get("job_required_skills") or [],
            "experience_years": 0,
            "salary_lpa":       None,
            "description":      (job.get("job_description") or "")[:600],
            "apply_link":       job.get("job_apply_link") or job.get("job_google_link") or "",
        })

    return normalized


def rank_jobs(
    search_text: str,
    top_n: int = 5,
    candidate: "CandidateSchema | None" = None,
    location: str = "",
) -> List[Dict[str, Any]]:
    """
    Return the top_n jobs most semantically similar to search_text.
    Uses live JSearch jobs when RAPIDAPI_KEY is set; falls back to static dataset.
    When `location` is provided, JSearch restricts results to that country/city.
    """
    jobs: List[Dict[str, Any]]
    job_texts: List[str]

    if RAPIDAPI_KEY and candidate:
        try:
            jobs = fetch_live_jobs(candidate, location=location)
            if not jobs:
                raise ValueError("JSearch returned no results")
            job_texts = [_build_job_text(j) for j in jobs]
            print(f"[JSearch] Fetched {len(jobs)} live jobs" + (f" in {location}" if location else ""))
        except Exception as exc:
            print(f"[JSearch Error] {exc} — falling back to static jobs")
            jobs      = JOBS
            job_texts = _JOB_TEXTS
    else:
        jobs      = JOBS
        job_texts = _JOB_TEXTS

    scores = _get_similarity_scores(search_text, job_texts)
    ranked = sorted(zip(jobs, scores), key=lambda pair: pair[1], reverse=True)

    return [
        {
            "id":               j["id"],
            "title":            j["title"],
            "company":          j["company"],
            "location":         j.get("location", ""),
            "remote":           j.get("remote", False),
            "domain":           j.get("domain", ""),
            "skills":           j.get("skills", []),
            "experience_years": j.get("experience_years", 0),
            "salary_lpa":       j.get("salary_lpa"),
            "description":      j.get("description", ""),
            "apply_link":       j.get("apply_link", ""),
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

    # Re-index jobs 0…N-1 so explanation_map lookups are always reliable
    for idx, j in enumerate(ranked_jobs):
        j["_idx"] = idx

    jobs_summary = [
        {
            "id":               j["_idx"],
            "title":            j["title"],
            "company":          j["company"],
            "location":         j.get("location", ""),
            "remote":           j.get("remote", False),
            "required_skills":  j.get("skills", [])[:5],
            "description":      j.get("description", "")[:150],
            "similarity_score": j["similarity_score"],
        }
        for j in ranked_jobs
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior technical recruiter. Analyse the candidate profile "
                "against each matched job. Be concise — 1-2 sentences per job. "
                "The clarifying question must target a real, observable gap or "
                "ambiguity — not a generic question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Candidate profile:\n{candidate.model_dump_json(indent=2)}\n\n"
                f"Matched jobs ({len(ranked_jobs)} total):\n{json.dumps(jobs_summary, indent=2)}\n\n"
                f"Call the tool with a 1-2 sentence explanation for EVERY one of these "
                f"{len(ranked_jobs)} jobs (all {len(ranked_jobs)}) and one clarifying question."
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
    # Key by _idx (0-based) which we set above — always matches regardless of original job ID
    explanation_map = {ex["job_id"]: ex["explanation"] for ex in args["explanations"]}
    clarifying_question = args.get("clarifying_question", "")

    final_jobs = [
        RankedJobSchema(
            id=j["_idx"],
            title=j["title"],
            company=j["company"],
            location=j.get("location") or None,
            is_remote=j.get("remote") or False,
            apply_link=j.get("apply_link") or None,
            similarity_score=j["similarity_score"],
            explanation=explanation_map.get(
                j["_idx"],
                f"Strong semantic match for your profile at {j['company']}.",
            ),
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

    # Extract location from the candidate's answer (e.g. "jobs in India")
    location = _extract_location(answer)

    # Parse candidate first so rank_jobs can build a live JSearch query
    candidate = parse_resume_with_groq(augmented_text)
    ranked_jobs_dicts = rank_jobs(augmented_text, top_n=20, candidate=candidate, location=location)
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
