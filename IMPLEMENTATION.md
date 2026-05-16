# Smart Job Match Agent — Implementation Document

**Project:** AI Engineering Intern Assignment — Cantilever Labs  
**Developer:** Tanish Gupta  
**Stack:** Python · FastAPI · Groq API · HuggingFace Inference API  

---

## 1. Problem Statement

Build an intelligent Job Recommendation API that:
- Accepts a candidate's raw resume text
- Returns a ranked list of relevant job openings with natural-language explanations
- Generates a smart follow-up clarifying question to improve match quality
- Exposes everything as a production-style REST API deployable on Vercel

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Request                        │
│                  POST /recommend  { resume_text }            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│                       api/index.py                           │
└──────────┬──────────────────────────────────────────────────┘
           │
           ├─── Step 1 ──────────────────────────────────────────┐
           │    Groq Tool Call #1                                 │
           │    Tool: extract_resume_info                         │
           │    → name, skills, experience_years,                 │
           │      preferred_roles, education                      │
           │                                                      │
           ├─── Step 2 ──────────────────────────────────────────┤
           │    HuggingFace Sentence-Similarity API               │
           │    (ONE batched call for all 50 jobs)                │
           │    source_sentence = resume_text                     │
           │    sentences       = [job1_text, job2_text, ...]     │
           │    → [0.67, 0.58, 0.49, 0.41, ...]  cosine scores   │
           │    → Sort descending → Top 5 jobs                    │
           │                                                      │
           └─── Step 3 ──────────────────────────────────────────┘
                Groq Tool Call #2
                Tool: provide_job_explanations
                Input: candidate profile + top 5 jobs
                → per-job 2-3 sentence explanation
                → one smart clarifying question
                           │
                           ▼
              Final JSON Response to Client
```

---

## 3. Project Structure

```
d:\JOB Recommendation\
├── api/
│   ├── index.py          ← FastAPI app, route handlers
│   ├── dependencies.py   ← All business logic (embeddings, Groq calls)
│   ├── schemas.py        ← Pydantic request/response models
│   └── jobs.json         ← 50 diverse job postings (static dataset)
├── requirements.txt
├── vercel.json           ← Vercel deployment config
├── .env                  ← API keys (not committed)
├── .env.example          ← Template for API keys
├── WRITEUP.md
└── IMPLEMENTATION.md     ← This document
```

---

## 4. Tech Stack and Why

| Component | Choice | Reason |
|-----------|--------|--------|
| Web Framework | FastAPI | Auto-generates OpenAPI docs, async-ready, Pydantic validation built-in |
| LLM Provider | Groq (`llama-3.3-70b-versatile`) | Fastest inference for open-weight LLMs, free tier, native tool calling |
| Embedding / Similarity | HuggingFace `all-MiniLM-L6-v2` (sentence-similarity pipeline) | No PyTorch dependency → fits Vercel's 1024 MB memory limit |
| Deployment | Vercel | Serverless Python support, free tier, public HTTPS URL |

---

## 5. The Dataset — `jobs.json`

- **50 job postings** spanning: Tech, Finance, Healthcare, Legal, AgriTech, EdTech, Robotics, Embedded, IoT, Research
- Each job has: `id, title, company, location, remote, skills, experience_years, salary_lpa, domain, description`
- Deliberately diverse so naive keyword matching fails — semantic understanding is required

---

## 6. Part 1 — Semantic Similarity Ranking (Classical ML)

### How it works

Instead of embedding resume and jobs separately (which would require 51 API calls), the system uses the **sentence-similarity pipeline** in batch mode:

```python
# api/dependencies.py

def _get_similarity_scores(resume_text, job_texts):
    payload = {
        "inputs": {
            "source_sentence": resume_text,   # the resume
            "sentences": job_texts            # all 50 job descriptions
        }
    }
    resp = requests.post(HF_MODEL_URL, headers=headers, json=payload)
    return resp.json()   # [0.67, 0.58, 0.49, ...]  — one score per job
```

**Result:** A single HTTP call returns 50 cosine similarity scores simultaneously.

### Ranking

```python
def rank_jobs(search_text, top_n=5):
    scores = _get_similarity_scores(search_text, _JOB_TEXTS)
    ranked = sorted(zip(JOBS, scores), key=lambda p: p[1], reverse=True)
    return [format_job(j, s) for j, s in ranked[:top_n]]
```

### Why `all-MiniLM-L6-v2`

- Trained on 1 billion+ sentence pairs with contrastive learning
- 384-dimensional vectors — captures semantic meaning, not just keywords
- Example: "NLP Engineer" resume correctly matches "Conversational AI Engineer" job even though those exact words don't overlap
- Scores are well-differentiated: e.g. 0.67, 0.58, 0.50, 0.44, 0.34 (not all clustered near 1.0)

---

## 7. Part 2 — Agentic LLM Layer (Groq Tool Calling)

This is the core of the assignment. The agent makes **two real, structured tool calls** using Groq's native function-calling API.

### Tool Call 1 — Resume Parser

```python
_PARSE_RESUME_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_resume_info",
        "description": "Extract structured candidate information from raw resume text.",
        "parameters": {
            "type": "object",
            "properties": {
                "name":             { "type": "string" },
                "skills":           { "type": "array", "items": { "type": "string" } },
                "experience_years": { "type": "number" },
                "preferred_roles":  { "type": "array", "items": { "type": "string" } },
                "education":        { "type": "string" }
            },
            "required": ["name", "skills", "experience_years", "preferred_roles", "education"]
        }
    }
}
```

**How the call works:**

```python
def parse_resume_with_groq(resume_text):
    messages = [
        { "role": "system", "content": "You are an expert HR assistant..." },
        { "role": "user",   "content": f"Parse this resume:\n\n{resume_text}" }
    ]
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=[_PARSE_RESUME_TOOL],
        tool_choice={"type": "function", "function": {"name": "extract_resume_info"}}
    )
    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    return CandidateSchema(**args)
```

**Output example:**
```json
{
  "name": "Anika Sharma",
  "skills": ["Python", "NLP", "Transformers", "scikit-learn", "FastAPI", "SQL"],
  "experience_years": 0.5,
  "preferred_roles": ["Data Scientist", "ML Engineer", "NLP Engineer"],
  "education": "B.Tech Computer Science, NIT Trichy, 2023"
}
```

### Tool Call 2 — Match Reasoning + Clarifying Question

```python
_EXPLAIN_JOBS_TOOL = {
    "type": "function",
    "function": {
        "name": "provide_job_explanations",
        "parameters": {
            "type": "object",
            "properties": {
                "explanations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "job_id":      { "type": "integer" },
                            "explanation": { "type": "string" }
                        }
                    }
                },
                "clarifying_question": { "type": "string" }
            }
        }
    }
}
```

The LLM receives the candidate profile + top 5 jobs and produces:
- A 2-3 sentence explanation per job (skill overlap, gaps, seniority fit)
- One targeted clarifying question based on observed gaps

**Why two separate tool calls instead of one big prompt?**

| Reason | Detail |
|--------|--------|
| Schema enforcement | Each tool has its own JSON schema — malformed output in one step doesn't corrupt the other |
| Separation of concerns | Embedding/ranking is independent of the LLM; they don't need to run in the same prompt |
| Debuggability | If tool call 1 fails, you know the resume parser broke — not the reasoning engine |
| Assignment requirement | The spec explicitly requires native tool calling, not prompt chaining |

---

## 8. Part 3 — Clarifying Question Generation

The clarifying question is generated **inside Tool Call 2** (not a separate call). It has context on:
- What skills the candidate listed
- What the top matched jobs require
- Where the gaps are

**Example output:**
> "Your resume doesn't mention cloud platforms like AWS or GCP, but two of your top matches (MLOps Engineer, AI Infrastructure Engineer) require them. Do you have hands-on cloud experience that isn't listed?"

This is dynamically generated — never hardcoded or templated.

---

## 9. Part 4 — FastAPI REST API

### Endpoint 1: `POST /recommend`

**Request:**
```json
{ "resume_text": "string" }
```

**Response:**
```json
{
  "candidate": {
    "name": "Anika Sharma",
    "skills": ["Python", "NLP", "scikit-learn"],
    "experience_years": 0.5,
    "preferred_roles": ["Data Scientist", "ML Engineer"],
    "education": "B.Tech CS, NIT Trichy"
  },
  "ranked_jobs": [
    {
      "id": 6,
      "title": "Data Scientist",
      "company": "HealthAI",
      "similarity_score": 0.6412,
      "explanation": "The candidate's background in patient risk prediction..."
    }
  ],
  "clarifying_question": "Several top matches are in the healthcare domain — is that a preference, or are you open to other industries?"
}
```

### Endpoint 2: `POST /refine` (Bonus)

Takes the candidate's answer to the clarifying question, folds it into the resume text, and re-ranks:

```json
// Request
{
  "resume_text": "...",
  "clarifying_question": "Are you open to remote work?",
  "candidate_answer": "Yes, I strongly prefer fully remote positions"
}

// Response
{
  "ranked_jobs": [...],   // Re-ranked with remote jobs boosted
  "reasoning": "The candidate's preference for remote work shifted the ranking..."
}
```

### Input Validation

```python
@app.post("/recommend")
def recommend_jobs(request: RecommendRequest):
    if not request.resume_text or not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty")
```

- Empty/whitespace resume → `400 Bad Request`
- Missing API keys → `500` with descriptive message
- All errors caught and returned as structured JSON

---

## 10. Deployment — Vercel

### Configuration (`vercel.json`)

```json
{
  "version": 2,
  "builds": [{ "src": "api/index.py", "use": "@vercel/python" }],
  "routes": [{ "src": "/(.*)", "dest": "api/index.py" }]
}
```

### Vercel Constraints Handled

| Constraint | How handled |
|------------|-------------|
| No persistent disk | `jobs.json` loaded at module import; no file writes at runtime |
| 1024 MB memory | No PyTorch — uses HF API instead of local sentence-transformers |
| 60s timeout | Single batched HF call (not 50) + two Groq calls ≈ 8-15s total |
| Cold starts | First request is slow (~3s); documented in write-up |

---

## 11. Complete Request Flow (Interview Walk-through)

```
User sends POST /recommend with resume text
        │
        ▼
[FastAPI validates input — 400 if empty]
        │
        ▼
[Groq Tool Call 1]
  LLM reads resume → calls extract_resume_info tool
  Returns: { name, skills, experience_years, preferred_roles, education }
        │
        ▼
[HuggingFace Batch API Call]
  Sends resume as source_sentence
  Sends all 50 job texts as sentences array
  Returns: [score_1, score_2, ..., score_50]
  Code sorts descending → picks top 5
        │
        ▼
[Groq Tool Call 2]
  LLM reads candidate profile + top 5 jobs
  Calls provide_job_explanations tool
  Returns:
    - 2-3 sentence explanation per job
    - 1 smart clarifying question
        │
        ▼
[FastAPI assembles final JSON response]
  { candidate, ranked_jobs, clarifying_question }
        │
        ▼
200 OK → Client receives response
```

---

## 12. Key Design Decisions & Trade-offs

### Decision 1: Batch Similarity vs. Individual Embeddings

**Chosen:** One batched API call using the sentence-similarity pipeline  
**Alternative:** Embed resume + embed each job → compute cosine similarity manually  
**Why chosen:** 50x fewer API calls, no manual pooling, cosine score returned directly  
**Trade-off:** Less control over the embedding vector itself; can't persist embeddings separately

### Decision 2: Merge Clarifying Question into Tool Call 2

**Chosen:** Clarifying question is a field in `provide_job_explanations` tool  
**Alternative:** Separate third LLM call just for the question  
**Why chosen:** The question needs context on matched jobs — it belongs in the same call. Also saves one API round-trip.  
**Trade-off:** If reasoning fails, the question is also lost

### Decision 3: Groq over OpenAI/Gemini

**Chosen:** Groq with `llama-3.3-70b-versatile`  
**Why:** Fastest inference, free tier is generous, open-weight model, excellent tool calling  
**Trade-off:** Groq is a third-party runtime — reliability depends on them, not Meta

---

## 13. Honest Weaknesses

1. **Noisy resumes:** If the resume has typos, mixed languages, or is just bullet fragments, both the embedding and the LLM parser will degrade
2. **Scale:** No caching between requests — each cold start re-calls HF for embeddings. At 10,000 concurrent requests, the HF free tier would be overwhelmed immediately
3. **No input length limit:** A 100-page document pasted as text would overflow the LLM context window
4. **Deterministic fallback is not semantic:** When HF_API_TOKEN is absent, the mock fallback uses word overlap — which is keyword matching, not semantic similarity

---

## 14. Next Steps (If Given More Time)

**Highest-impact improvement: Pre-compute job embeddings at build time**

Currently, job text representations are sent to HF on every cold start. Since the 50 jobs never change, we can:
1. Run a one-time script to get all 50 embeddings
2. Store them in a `job_embeddings.json` file committed to the repo
3. Load at startup — zero API calls for the ranking step

This would reduce cold-start time from ~15s to under 1s and eliminate the HF API dependency for everything except the resume embedding.

---

## 15. How to Run Locally

```bash
# 1. Clone and enter project
cd "d:\JOB Recommendation"

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API keys
copy .env.example .env
# Edit .env with your real GROQ_API_KEY and HF_API_TOKEN

# 5. Start server
uvicorn api.index:app --reload

# 6. Open interactive docs
# http://127.0.0.1:8000/docs
```

---

*Document prepared for interview reference — Smart Job Match Agent, May 2026*
