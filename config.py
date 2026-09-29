"""Central configuration for the job portal's AI features."""
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Fast, strong open-weight model on Groq's LPU inference.
# Other options: "llama-3.1-8b-instant" (cheaper/faster), "gemma2-9b-it".
LLM_MODEL = os.environ.get("JOB_PORTAL_LLM_MODEL", "llama-3.3-70b-versatile")
LLM_MAX_TOKENS = 700
