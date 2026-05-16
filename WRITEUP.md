# Technical Write-up — Smart Job Match Agent

## 1. Design Choices

**Embedding model: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace Serverless Inference API (sentence-similarity pipeline)**

I chose `all-MiniLM-L6-v2` because it is the dominant baseline for semantic sentence similarity — trained on over 1 billion sentence pairs with a contrastive objective, producing cosine scores that capture meaning well beyond keyword overlap. Its compact size makes it practical, and the quality-to-speed ratio is well documented.

The critical deployment constraint was Vercel's free tier: 1024 MB memory and no persistent disk. Loading sentence-transformers locally requires PyTorch (~500 MB), which exhausts the memory budget. I instead call HuggingFace's serverless sentence-similarity endpoint. Crucially, I use the **batch** API: the resume is sent as `source_sentence` and all 50 job descriptions are sent as `sentences` in a **single HTTP call**, which returns 50 cosine scores simultaneously. This means the entire ranking step costs one API round-trip, not 51.

**Alternatives considered and rejected:**
- *Local sentence-transformers*: Ideal accuracy, but PyTorch exceeds Vercel's memory budget.
- *OpenAI `text-embedding-3-small`*: Excellent quality, but requires a second paid API key. Avoided to minimise credentials for the reviewer.
- *TF-IDF / BM25*: Fast and dependency-free, but purely lexical — the dataset is intentionally diverse, so a finance resume would not match an "Agri-Tech Data Scientist" role without semantic understanding.

**LLM: Groq + `llama-3.3-70b-versatile`**

Groq's inference engine is the fastest publicly available runtime for open-weight models. `llama-3.3-70b-versatile` has a 128k context window and strong structured tool-calling reliability. The free tier is generous enough for a demo at this scale. I expose the model name via `GROQ_MODEL` so evaluators can switch to `llama-3.1-8b-instant` if rate limits are hit.

---

## 2. Agentic Architecture

The agent makes **two real, sequential tool calls** per `/recommend` request:

```
Resume Text
    │
    ▼
[Tool Call 1] extract_resume_info
    │  → name, skills, experience_years, preferred_roles, education
    ▼
Semantic Ranking (cosine similarity, all 50 jobs)
    │  → top-5 jobs with similarity scores
    ▼
[Tool Call 2] provide_job_explanations
    │  → per-job 2-3 sentence explanation + one clarifying question
    ▼
Final JSON Response
```

**Why two tool calls instead of one large prompt?**

Each tool enforces its own JSON schema, which acts as a structural contract between the LLM and the application. If both tasks were merged into a single prompt, a single hallucinated field would corrupt the entire output with no clean way to retry just the broken part. Splitting them also lets the semantic ranking layer run independently — the embedding computation does not need the LLM, so the two concerns stay decoupled.

**Failure modes:**
- *Groq JSON truncation*: If the LLM output is cut before the closing brace, `json.loads` raises an exception. The `/recommend` endpoint catches this and returns a 500 with the raw error.
- *HF API cold start (503)*: The HF Inference API warms up models on demand. The first call may take 20–30 seconds and exceed Vercel's 60-second timeout. Mitigation: `"options": {"wait_for_model": true}` is set; users should expect cold-start latency.
- *Embedding shape mismatch*: The HF feature-extraction pipeline can return token-level embeddings (`[seq, dim]`) instead of a sentence vector. The code handles all three possible shapes and mean-pools where necessary.
- *Missing tool call in response*: If the LLM returns a plain text answer instead of a tool call (rare but possible), `response.choices[0].message.tool_calls` will be `None`, raising an `AttributeError`. A production system would retry with temperature=0 and an explicit reminder.

---

## 3. Honest Weaknesses

**Noisy or poorly written resumes**

`all-MiniLM-L6-v2` was trained on clean sentence pairs. A resume full of acronyms, bullet fragments, or non-standard formatting will produce embeddings that drift toward the mean, making scores cluster and differentiation collapse. The Groq parser may also hallucinate `experience_years` if the resume does not include explicit dates — it infers from graduation year, which is an approximation.

**At scale (10,000 concurrent requests)**

- Each request cold-computes embeddings for all 50 jobs (50 HF API calls) because Vercel serverless functions have no shared memory between invocations. At scale this would saturate the HF free tier in seconds.
- The Groq API is rate-limited; burst traffic would return 429 errors with no retry logic implemented.
- There is no request queue, caching layer, or circuit breaker.

**Corners cut due to time**

- Job embeddings are not pre-computed and stored. A build-time step that serialises all 50 embeddings to JSON would eliminate 50 API calls per cold start.
- No resume preprocessing (stripping headers, normalising whitespace, removing PII) before embedding.
- The `/refine` endpoint re-parses the resume on every call; the original parse result is not cached between `/recommend` and `/refine`.
- No input length limit — a 50-page PDF pasted as text would overflow the LLM context.

---

## 4. Next Steps

**Pre-compute and bundle job embeddings at build time.**

This single change would have the highest impact. Currently, the first request after a cold start triggers 50 sequential HF API calls (one per job), which takes 15–30 seconds and risks a Vercel timeout. If the embeddings were computed once during CI/CD and serialised into a `job_embeddings.json` file committed to the repo, each cold start would simply load a JSON file — reducing the cold-start penalty from 30 seconds to under 1 second and eliminating all HF API calls for the ranking step. The only remaining API calls would be the single resume embedding and the two Groq tool calls, putting the total response time comfortably within Vercel's 60-second limit for normal resumes.
