from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    match_percentage: float = Field(description="TF-IDF cosine similarity as a 0-100 score.")
    ats_score: float = Field(description="Heuristic ATS-friendliness score from 0-100.")
    matched_keywords: list[str]
    missing_keywords: list[str]
    improvement_tips: list[str]
    resume_preview: str
    keyword_coverage: float
