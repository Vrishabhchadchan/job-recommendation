# Smart Job Match Agent

A "Smart Job Match Agent" backend using Python and FastAPI, explicitly built for Vercel's free tier. 

## Features
1. **Classical ML Layer (Semantic Ranking):** Uses HuggingFace Inference API for generating embeddings to keep the bundle size small enough for Vercel. Calculates pure Python Cosine Similarity to find Top-5 jobs without relying on heavyweight libraries like `numpy` or `torch`.
2. **Agentic LLM Layer (Groq Native Tool Calling):**
   - Parses resume into structured data via tool calling.
   - Generates natural-language reasoning for why a job is or isn't a fit via tool calling.
3. **Clarifying Question Generation:** LLM generates a targeted follow-up question dynamically.
4. **Refinement Endpoint:** Updates the rankings based on candidate's answer to the clarifying question.

## API Endpoints
- `POST /recommend`
- `POST /refine`

## Setup & Local Development
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` or `venv\Scripts\activate` on Windows.
3. Install dependencies: `pip install -r requirements.txt`
4. Set up environment variables in `.env` (use `.env.example` as a template).
5. Run the dev server: `uvicorn api.index:app --reload`
