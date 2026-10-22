from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.skills import KNOWN_SKILLS

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{8,}\d)")
STOP_EXTRA = {
    "experience",
    "responsibilities",
    "requirement",
    "requirements",
    "required",
    "preferred",
    "ability",
    "strong",
    "including",
    "using",
    "work",
    "working",
    "role",
    "position",
    "candidate",
    "team",
    "years",
    "year",
    "hiring",
    "plus",
    "must",
    "need",
    "needs",
    "looking",
    "join",
    "company",
    "opportunity",
    "job",
    "description",
    "engineer",
    "developer",
    "backend",
    "frontend",
}

SECTION_PATTERNS = {
    "experience": r"\b(experience|work history|employment|professional experience)\b",
    "education": r"\b(education|academic|degree|university|bachelor|master)\b",
    "skills": r"\b(skills|technical skills|competencies|proficiencies)\b",
    "summary": r"\b(summary|profile|objective|about me)\b",
}


@dataclass
class Analysis:
    match_percentage: float
    ats_score: float
    matched_keywords: list[str]
    missing_keywords: list[str]
    improvement_tips: list[str]
    resume_preview: str
    keyword_coverage: float


def analyze(resume_text: str, job_description: str, filename: str) -> Analysis:
    jd = job_description.strip()
    resume = resume_text.strip()
    match_percentage = _cosine_match(resume, jd)
    keywords = _extract_keywords(jd)
    matched, missing = _compare_keywords(resume, keywords)
    coverage = 0.0 if not keywords else round(100 * len(matched) / len(keywords), 1)
    ats_score, ats_notes = _ats_score(resume, filename)
    tips = _improvement_tips(match_percentage, missing, ats_notes, coverage)
    preview = resume[:400] + ("..." if len(resume) > 400 else "")
    return Analysis(
        match_percentage=match_percentage,
        ats_score=ats_score,
        matched_keywords=matched,
        missing_keywords=missing,
        improvement_tips=tips,
        resume_preview=preview,
        keyword_coverage=coverage,
    )


def _cosine_match(resume: str, job_description: str) -> float:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
    )
    matrix = vectorizer.fit_transform([job_description, resume])
    score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    return round(max(0.0, min(score, 1.0)) * 100, 1)


def _extract_keywords(job_description: str) -> list[str]:
    lowered = job_description.lower()
    found: list[str] = []
    seen: set[str] = set()

    for skill in sorted(KNOWN_SKILLS, key=len, reverse=True):
        if _contains_phrase(lowered, skill) and skill not in seen:
            seen.add(skill)
            found.append(skill)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=1,
        max_features=80,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#.]{1,}\b",
    )
    try:
        matrix = vectorizer.fit_transform([job_description])
    except ValueError:
        return found[:25]

    scores = zip(vectorizer.get_feature_names_out(), matrix.toarray()[0])
    ranked = sorted(scores, key=lambda item: item[1], reverse=True)
    for term, weight in ranked:
        cleaned = term.strip().lower()
        if weight <= 0 or cleaned in seen or cleaned in STOP_EXTRA:
            continue
        if any(part in STOP_EXTRA for part in cleaned.split()):
            continue
        if any(cleaned in existing or existing in cleaned for existing in seen):
            continue
        seen.add(cleaned)
        found.append(cleaned)
        if len(found) >= 25:
            break
    return found


def _compare_keywords(resume: str, keywords: list[str]) -> tuple[list[str], list[str]]:
    lowered = resume.lower()
    matched: list[str] = []
    missing: list[str] = []
    for keyword in keywords:
        if _contains_phrase(lowered, keyword):
            matched.append(keyword)
        else:
            missing.append(keyword)
    return matched, missing


def _contains_phrase(text: str, phrase: str) -> bool:
    if " " in phrase or any(ch in phrase for ch in "+#./"):
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _ats_score(resume: str, filename: str) -> tuple[float, list[str]]:
    score = 40.0
    notes: list[str] = []
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix in {"pdf", "docx"}:
        score += 15
    elif suffix == "txt":
        score += 8
        notes.append("Prefer a PDF or DOCX resume; plain text can lose formatting in some ATS tools.")
    else:
        notes.append("Use a PDF or DOCX file so applicant tracking systems can parse your resume.")

    if EMAIL_RE.search(resume):
        score += 10
    else:
        notes.append("Add a professional email address near the top of the resume.")

    if PHONE_RE.search(resume):
        score += 8
    else:
        notes.append("Include a phone number so recruiters can contact you.")

    present_sections = [
        name for name, pattern in SECTION_PATTERNS.items() if re.search(pattern, resume, re.I)
    ]
    score += min(5 * len(present_sections), 20)
    missing_sections = [name for name in SECTION_PATTERNS if name not in present_sections]
    if missing_sections:
        notes.append(
            "Add clear headings for: " + ", ".join(missing_sections) + "."
        )

    word_count = len(re.findall(r"\b\w+\b", resume))
    if 250 <= word_count <= 900:
        score += 7
    elif word_count < 250:
        notes.append("The resume looks short. Add measurable achievements from recent roles.")
        score -= 5
    else:
        notes.append("The resume is quite long. Keep it to about one or two pages.")
        score -= 3

    if re.search(r"[•●▪︎\-–]\s+\S", resume):
        score += 5
    else:
        notes.append("Use bullet points for achievements so both humans and ATS can scan them.")

    return round(max(0.0, min(score, 100.0)), 1), notes


def _improvement_tips(
    match_percentage: float,
    missing: list[str],
    ats_notes: list[str],
    coverage: float,
) -> list[str]:
    tips: list[str] = []
    if match_percentage < 45:
        tips.append(
            "The resume and job description have low overlap. Mirror important phrasing from the posting in your experience bullets."
        )
    elif match_percentage < 70:
        tips.append(
            "You are partially aligned. Tune the summary and skills section toward this role's wording."
        )
    else:
        tips.append("Strong overlap with this posting. Tailor a few bullets to the company's top priorities.")

    if missing:
        preview = ", ".join(missing[:8])
        extra = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        tips.append(
            f"Keywords from the job description that are missing or weakly present: {preview}{extra}."
        )
        tips.append(
            "Only add missing skills you actually have. If you used a tool under another name, spell it the way the posting does."
        )

    if coverage < 60:
        tips.append(
            "Keyword coverage is low. Put a dedicated Skills section near the top with tools from the job description."
        )

    tips.extend(ats_notes[:4])
    tips.append(
        "Quantify impact (%, time saved, users, revenue) so ATS-friendly keywords sit next to proof."
    )
    return tips[:8]
