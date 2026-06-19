# AI Resume Screener & Job Match System

An AI-powered recruitment tool that automatically screens CVs against a job description and ranks candidates by match score — built with LangChain, FAISS, and Hugging Face.

## How It Works
```
Job Description (PDF or text)
          +
Multiple CVs (PDF)
          ↓
Text extraction (PyPDF)
          ↓
Vector embeddings (Hugging Face sentence-transformers)
          ↓
Similarity search (FAISS)
          ↓
Match score per candidate (0-100%)
          ↓
Skill extraction (zero-shot classification)
          ↓
Ranked results with verdict
```

## Features
- Upload multiple CVs at once
- Paste or upload job description
- AI match score for each candidate (0-100%)
- Automatic skill detection using zero-shot classification
- Ranked leaderboard — best candidate highlighted
- Strong / Moderate / Weak match verdict

## Tech Stack
- Python 3.10+
- LangChain — document processing
- FAISS — vector similarity search
- Hugging Face Transformers — embeddings + zero-shot classification
- Streamlit — web interface
- PyPDF — PDF text extraction

## Setup
```bash
git clone https://github.com/Gopikrishna1547/ai-resume-screener.git
cd ai-resume-screener
pip install -r requirements.txt
streamlit run app.py
```

## Author
Gopikrishna Bojedla
[LinkedIn](https://www.linkedin.com/in/gopi-krishna-83856320a)
