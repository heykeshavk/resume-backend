# Resume Matcher Backend

FastAPI service that compares a resume to a job description:

1. Upload a resume (PDF, DOCX, or TXT) and paste a job description
2. Score overlap with **TF-IDF** and **cosine similarity**
3. Extract keywords and list missing skills
4. Estimate an **ATS-friendliness** score
5. Return concrete improvement tips

## Setup

```powershell
cd N:\resume-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open interactive docs at http://127.0.0.1:8000/docs

## Railway

This repo deploys as a FastAPI web service. Railway should run:

`uvicorn app.main:app --host 0.0.0.0 --port $PORT`

After the backend has a public URL, set `VITE_API_URL` on the frontend service to that URL (for example `https://resume-backend-production.up.railway.app`) and rebuild the frontend.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness check |
| POST | `/analyze` | Multipart form: `resume` file + `job_description` text |

Example:

```powershell
curl -X POST http://127.0.0.1:8000/analyze `
  -F "resume=@.\sample-resume.txt" `
  -F "job_description=We need a Python FastAPI developer with AWS, Docker, and SQL."
```

JSON fields in the response:

- `match_percentage` — TF-IDF cosine similarity (0–100)
- `ats_score` — parsing / structure heuristic (0–100)
- `matched_keywords` / `missing_keywords`
- `keyword_coverage`
- `improvement_tips`
- `resume_preview`
