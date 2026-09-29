"""Uses Groq to produce a qualitative analysis of a resume against a job
description: an AI match score, a short summary, and concrete strengths /
gaps. This complements (not replaces) the fast local TF-IDF score in
matching.py — TF-IDF gives an instant, deterministic baseline; Groq adds
reasoning a recruiter can actually read.

If GROQ_API_KEY isn't set, analyze_application() returns None so callers
can degrade gracefully (show the TF-IDF score only).
"""
import json
import re
from typing import Optional, Dict
import config

_client = None


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def llm_available() -> bool:
    return bool(config.GROQ_API_KEY)


def analyze_application(jd_text: str, resume_text: str) -> Optional[Dict]:
    """Return {llm_score, summary, strengths[], gaps[]} or None if no API key
    is configured, or if the model call/parse fails."""
    if not llm_available():
        return None

    prompt = (
        "You are a recruiting assistant. Compare the candidate's resume to the "
        "job description and evaluate fit.\n\n"
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"RESUME:\n{resume_text}\n\n"
        "Respond ONLY with a JSON object, no preamble, no markdown fences, in "
        "exactly this shape:\n"
        '{"score": <integer 0-100, your overall fit assessment>, '
        '"summary": "<one or two sentence overall verdict>", '
        '"strengths": ["<short bullet>", "..."], '
        '"gaps": ["<short bullet>", "..."]}'
    )

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```json|```$", "", text).strip()
        data = json.loads(text)

        score = data.get("score")
        score = max(0, min(100, float(score))) if score is not None else None

        return {
            "llm_score": score,
            "summary": str(data.get("summary", "")).strip(),
            "strengths": [str(s).strip() for s in data.get("strengths", []) if str(s).strip()],
            "gaps": [str(g).strip() for g in data.get("gaps", []) if str(g).strip()],
        }
    except Exception as e:
        print(f"[groq_client] analysis failed, falling back to TF-IDF only: {e}")
        return None
