import os
import tempfile
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import torch

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="AI",
    layout="wide"
)

st.title("AI Resume Screener & Job Match System")
st.markdown("Upload a job description and multiple CVs — AI will rank candidates by match score.")
st.divider()

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    os.unlink(tmp_path)
    return "\n".join([p.page_content for p in pages])

def get_match_score(cv_text, job_text, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    cv_chunks = splitter.create_documents([cv_text])
    job_chunks = splitter.create_documents([job_text])
    if not cv_chunks or not job_chunks:
        return 0.0
    cv_store = FAISS.from_documents(cv_chunks, embeddings)
    scores = []
    for jchunk in job_chunks[:5]:
        results = cv_store.similarity_search_with_score(jchunk.page_content, k=1)
        if results:
            distance = results[0][1]
            similarity = max(0, 1 - distance / 2)
            scores.append(similarity)
    return round(sum(scores) / len(scores) * 100, 1) if scores else 0.0

def extract_skills(text, classifier):
    skills = [
        "Python", "SQL", "Machine Learning", "NLP", "Deep Learning",
        "RAG", "LangChain", "Power BI", "Data Analysis", "FastAPI",
        "Docker", "AWS", "TensorFlow", "PyTorch", "Scikit-learn"
    ]
    try:
        result = classifier(text[:1000], skills, multi_label=True)
        matched = [
            label for label, score in zip(result["labels"], result["scores"])
            if score > 0.5
        ]
        return matched[:8]
    except:
        return []

def get_verdict(score):
    if score >= 75:
        return "Strong Match", "success"
    elif score >= 55:
        return "Moderate Match", "warning"
    else:
        return "Weak Match", "error"

# UI Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Job Description")
    job_input_type = st.radio(
        "Input method",
        ["Upload PDF", "Paste text"],
        horizontal=True
    )

    job_text = ""
    if job_input_type == "Upload PDF":
        job_file = st.file_uploader("Upload job description PDF", type=["pdf"], key="job")
        if job_file:
            job_text = extract_text_from_pdf(job_file)
            st.success("Job description loaded!")
    else:
        job_text = st.text_area(
            "Paste job description here",
            height=200,
            placeholder="We are looking for an AI/ML Engineer with experience in Python, NLP, RAG systems..."
        )

with col2:
    st.subheader("Candidate CVs")
    cv_files = st.file_uploader(
        "Upload one or more CVs (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key="cvs"
    )
    if cv_files:
        st.success(f"{len(cv_files)} CV(s) uploaded!")

st.divider()

if st.button("Screen Candidates", type="primary", use_container_width=True):
    if not job_text.strip():
        st.warning("Please provide a job description first!")
    elif not cv_files:
        st.warning("Please upload at least one CV!")
    else:
        embeddings = load_embeddings()
        classifier = load_classifier()

        results = []
        progress = st.progress(0, text="Analysing candidates...")

        for i, cv_file in enumerate(cv_files):
            progress.progress((i + 1) / len(cv_files), text=f"Analysing {cv_file.name}...")
            cv_text = extract_text_from_pdf(cv_file)
            score = get_match_score(cv_text, job_text, embeddings)
            skills = extract_skills(cv_text, classifier)
            verdict, verdict_type = get_verdict(score)
            results.append({
                "name": cv_file.name.replace(".pdf", ""),
                "score": score,
                "skills": skills,
                "verdict": verdict,
                "verdict_type": verdict_type,
                "text": cv_text
            })

        progress.empty()
        results.sort(key=lambda x: x["score"], reverse=True)

        st.subheader("Candidate Rankings")
        for rank, r in enumerate(results, 1):
            with st.expander(
                f"#{rank}  {r['name']}  —  Match Score: {r['score']}%  |  {r['verdict']}",
                expanded=(rank == 1)
            ):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.metric("Match Score", f"{r['score']}%")
                    if r["verdict_type"] == "success":
                        st.success(r["verdict"])
                    elif r["verdict_type"] == "warning":
                        st.warning(r["verdict"])
                    else:
                        st.error(r["verdict"])

                with col_b:
                    if r["skills"]:
                        st.markdown("**Detected Skills:**")
                        st.write("  ".join([f"`{s}`" for s in r["skills"]]))
                    else:
                        st.markdown("**Detected Skills:** N/A")

                with st.expander("View CV text"):
                    st.text(r["text"][:1000] + "...")

        st.divider()
        st.subheader("Top Recommendation")
        best = results[0]
        st.success(
            f"**{best['name']}** is the best match with a score of **{best['score']}%**"
        )
