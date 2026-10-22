from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.nlp import analyze
from app.parsers import ParseError, extract_text
from app.schemas import AnalyzeResponse

app = FastAPI(
    title="Resume Matcher API",
    description=(
        "Upload a resume and a job description. The API scores overlap with TF-IDF "
        "cosine similarity, finds missing keywords, estimates ATS-friendliness, "
        "and returns improvement tips."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    job_description: str = Form(..., min_length=40),
    resume: UploadFile = File(...),
) -> AnalyzeResponse:
    data = await resume.read()
    if not data:
        raise HTTPException(status_code=400, detail="Resume file is empty.")
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Resume must be 8 MB or smaller.")

    filename = resume.filename or "resume.txt"
    try:
        resume_text = extract_text(filename, data)
    except ParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = analyze(resume_text, job_description, filename)
    return AnalyzeResponse(**result.__dict__)
