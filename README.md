# ResumeAI Pro — AI-Powered Resume Analyzer & Career Intelligence Platform

> Final Year Project (FYP) — A comprehensive AI-driven platform for resume analysis, career guidance, and recruitment intelligence.

## 📋 Abstract

ResumeAI Pro is an intelligent web application that analyzes resumes using AI and ATS algorithms. It provides personalized suggestions, compares resumes with job descriptions, predicts interview success, generates interview questions, creates cover letters, recommends projects and certifications, and offers a complete career improvement roadmap.

The system combines Large Language Models (LLMs), Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), semantic search, and machine learning to deliver recruiter-level resume feedback.

## ✨ Features (12 Modules)

| # | Module | Description |
|---|--------|-------------|
| 1 | **AI Resume Parser** | Extract text from PDF/DOCX, detect sections, parse experience, skills, education, projects, certifications |
| 2 | **ATS Score Engine** | Multi-factor scoring: formatting, keywords, skills, experience, education, readability, grammar |
| 3 | **Job Description Matcher** | Keyword match %, missing skills, experience/education gap, ATS compatibility |
| 4 | **AI Resume Improver** | LLM-powered bullet point rewriting, summary enhancement, achievement quantification |
| 5 | **AI Cover Letter Generator** | Personalized cover letters from resume + company + job description |
| 6 | **AI Interview Generator** | HR, Technical, Behavioral, Coding, Project-based questions with difficulty levels |
| 7 | **Skill Gap Analysis** | Compare resume vs job skills, recommend learning resources |
| 8 | **Salary Predictor** | Predict salary based on skills, experience, education, role, location |
| 9 | **Resume Templates** | Harvard, Google, Microsoft, Modern, Minimal, Developer templates |
| 10 | **AI Career Roadmap** | Weekly learning path with projects, certifications, courses |
| 11 | **Recruiter Dashboard** | Upload resumes, rank candidates, filter, compare, download reports |
| 12 | **Student Dashboard** | Track ATS score history, applications, skill growth, resume versions |

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Streamlit, Plotly, Custom CSS |
| **Backend** | Python, FastAPI-compatible modules |
| **AI/ML** | OpenAI GPT, LangChain, Sentence Transformers, spaCy, Scikit-learn |
| **Data** | Pandas, NumPy, JSON skills database |
| **PDF** | PyPDF2, python-docx, pdfplumber, ReportLab |
| **Cloud** | Docker-ready, Streamlit Cloud deployable |

## 📁 Project Structure

```
resumeai_pro/
├── app.py                      # Main Streamlit application (12 module pages)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   └── skills_database.json    # Comprehensive skills taxonomy
├── modules/
│   ├── resume_parser.py        # Module 1: AI Resume Parser
│   ├── ats_engine.py           # Module 2: ATS Score Engine
│   ├── jd_matcher.py           # Module 3: Job Description Matcher
│   ├── resume_improver.py      # Module 4: AI Resume Improver
│   ├── cover_letter.py         # Module 5: AI Cover Letter Generator
│   ├── interview_gen.py        # Module 6: AI Interview Generator
│   ├── skill_gap.py            # Module 7: Skill Gap Analysis
│   ├── salary_predictor.py    # Module 8: Salary Predictor
│   ├── templates.py            # Module 9: Resume Templates
│   ├── career_roadmap.py       # Module 10: AI Career Roadmap
│   ├── recruiter_dash.py      # Module 11: Recruiter Dashboard
│   └── student_dash.py        # Module 12: Student Dashboard
└── utils/
    ├── ai_engine.py            # LLM integration (OpenAI + fallbacks)
    ├── nlp_utils.py            # NLP utilities (spaCy, TF-IDF, regex)
    └── pdf_generator.py        # PDF generation (ReportLab)
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Himanshu90909/ResumeAI-Pro.git
cd ResumeAI-Pro/resumeai_pro

# Install dependencies
pip install -r requirements.txt

# (Optional) Set OpenAI API key for AI features
export OPENAI_API_KEY=your_api_key_here

# Run the app
streamlit run app.py
```

> **Note:** The app works without an OpenAI API key — AI features fall back to intelligent rule-based algorithms.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           Streamlit Frontend (app.py)         │
│    12 Module Pages + Dashboard + Navigation   │
└──────────────────────┬──────────────────────┘
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  NLP     │ │ AI Engine │ │  Data    │
    │  Utils   │ │ (LLM +   │ │  Utils   │
    │ (spaCy)  │ │ Fallback)│ │ (PDF)    │
    └──────────┘ └──────────┘ └──────────┘
           │           │           │
           ▔───────────┴───────────┘
                       │
              ┌────────┴────────┐
              │  Skills Database │
              │  (JSON)          │
              └─────────────────┘
```

## 🎯 Key Highlights

- **Full-stack development** with Streamlit frontend and modular Python backend
- **AI and LLM integration** with OpenAI GPT and rule-based fallbacks
- **NLP and document parsing** with spaCy, PyPDF2, python-docx
- **Semantic search** using TF-IDF cosine similarity
- **Cloud deployment** ready for Streamlit Cloud
- **Real-world business value** with SaaS potential

## 👨‍💻 Author

**Himanshu Suthar**  
B.Tech CSE (AI & ML) — Lovely Professional University  
GitHub: [@Himanshu90909](https://github.com/Himanshu90909)  
Portfolio: [silent-neural-archive-core.base44.app](https://silent-neural-archive-core.base44.app/)

## 📄 License

This project is part of a Final Year Project submission. © 2026 Himanshu Suthar.
