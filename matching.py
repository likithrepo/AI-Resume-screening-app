"""Computes how well a resume matches a job description.

Reuses the same idea as the Smart Document AI retrieval engine: TF-IDF
(word + bigram) vectors + cosine similarity. This needs no external model
downloads and works fully offline. The score is paired with the actual
overlapping key terms so a recruiter can see *why* a candidate scored
the way they did, not just a bare number.
"""
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_match_score(jd_text: str, resume_text: str) -> Tuple[float, List[str]]:
    """Return (score 0-100, matched_keywords) for a resume against a JD."""
    jd_text = (jd_text or "").strip()
    resume_text = (resume_text or "").strip()
    if not jd_text or not resume_text:
        return 0.0, []

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform([jd_text, resume_text])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    score = round(float(similarity) * 100, 1)

    matched_keywords = _matched_keywords(jd_text, resume_text)
    return score, matched_keywords


def _matched_keywords(jd_text: str, resume_text: str, top_n: int = 15) -> List[str]:
    """Top JD terms (by TF-IDF weight) that also appear in the resume."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=200)
    try:
        tfidf = vectorizer.fit_transform([jd_text])
    except ValueError:
        return []
    scores = tfidf.toarray()[0]
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda t: t[1], reverse=True)
    resume_lower = resume_text.lower()
    matched = [term for term, weight in ranked if weight > 0 and term in resume_lower]
    return matched[:top_n]
