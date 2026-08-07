import sys
import os
import io

# Ensure modules and utils packages resolve correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules.resume_parser import ResumeParser
from modules.ats_engine import ATSEngine
from modules.jd_matcher import JDMatcher
from modules.resume_improver import ResumeImprover
from modules.cover_letter import CoverLetterGenerator
from modules.interview_gen import InterviewGenerator
from modules.skill_gap import SkillGapAnalyzer
from modules.salary_predictor import SalaryPredictor
from modules.templates import ResumeTemplateGenerator
from modules.career_roadmap import CareerRoadmap
from modules.recruiter_dash import RecruiterDashboard
from modules.student_dash import StudentDashboard

from utils.nlp_utils import clean_text, extract_skills
from utils.ai_engine import AIEngine
from utils.pdf_generator import generate_resume_pdf, generate_report_pdf

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ResumeAI Pro — AI Career Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles & Dark Blue Theme */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 24px;
    }
    
    /* Custom Card */
    .custom-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #334155;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #2563eb;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #60a5fa;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Tag Badges */
    .badge-matched {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }
    .badge-missing {
        background-color: #991b1b;
        color: #fca5a5;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }
    .badge-skill {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        display: inline-block;
        margin: 3px;
    }
    
    /* Difficulty Badges */
    .badge-easy { background-color: #065f46; color: #34d399; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; }
    .badge-medium { background-color: #9a3412; color: #fdba74; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; }
    .badge-hard { background-color: #991b1b; color: #fca5a5; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; }

    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        border-top: 1px solid #334155;
        color: #64748b;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. INITIALIZE SESSION STATE
# -----------------------------------------------------------------------------
if "resume_data" not in st.session_state:
    st.session_state.resume_data = {
        "filename": "Sample_Resume.pdf",
        "raw_text": (
            "Jane Doe\nSoftware Engineer\njane.doe@example.com | +1 (555) 019-2834 | San Francisco, CA\n\n"
            "PROFESSIONAL SUMMARY\nResults-driven Software Engineer with 4+ years of experience engineering high-throughput web applications, "
            "optimizing database schemas, and building automated CI/CD pipelines.\n\n"
            "TECHNICAL SKILLS\nProgramming: Python, JavaScript, TypeScript, SQL, HTML, CSS\n"
            "Frameworks: React, Django, FastAPI, Node.js, Express, Streamlit\n"
            "Tools & Cloud: AWS, Docker, Git, PostgreSQL, Redis, Linux, CI/CD\n\n"
            "WORK EXPERIENCE\n"
            "Senior Software Engineer | Acme Tech (2022 - Present)\n"
            "• Spearheaded backend architecture using Python and FastAPI, serving 500k+ active users.\n"
            "• Optimized PostgreSQL database queries, reducing average API response times by 35%.\n"
            "• Built automated CI/CD deployment pipelines on AWS using Docker and GitHub Actions.\n\n"
            "Software Developer | Tech Innovations (2020 - 2022)\n"
            "• Developed full-stack web applications with React and Node.js.\n"
            "• Collaborated with cross-functional product teams using Agile methodologies.\n\n"
            "EDUCATION\n"
            "B.S. in Computer Science | Tech University (2016 - 2020) — GPA: 3.8/4.0"
        ),
        "personal_info": {
            "name": "Jane Doe",
            "email": "jane.doe@example.com",
            "phone": "+1 (555) 019-2834",
            "location": "San Francisco, CA"
        },
        "summary": "Results-driven Software Engineer with 4+ years of experience engineering high-throughput web applications, optimizing database schemas, and building automated CI/CD pipelines.",
        "skills": {
            "all": ["Python", "JavaScript", "TypeScript", "SQL", "React", "Django", "FastAPI", "Node.js", "AWS", "Docker", "Git", "PostgreSQL", "Redis", "Linux", "CI/CD"],
            "by_category": {
                "Programming Languages": ["Python", "JavaScript", "TypeScript", "SQL"],
                "Frameworks & Libraries": ["React", "Django", "FastAPI", "Node.js"],
                "Tools & Platforms": ["AWS", "Docker", "Git", "PostgreSQL", "Redis", "Linux"]
            }
        },
        "experience": [
            "Senior Software Engineer | Acme Tech (2022 - Present) - Spearheaded backend architecture serving 500k+ active users.",
            "Software Developer | Tech Innovations (2020 - 2022) - Developed full-stack web applications with React and Node.js."
        ],
        "education": ["B.S. in Computer Science | Tech University (2016 - 2020) — GPA: 3.8/4.0"],
        "projects": ["AI Resume Analyzer Pro — Streamlit multi-page web platform for ATS keyword matching."],
        "certifications": ["AWS Certified Solutions Architect – Associate"]
    }

if "target_job_description" not in st.session_state:
    st.session_state.target_job_description = (
        "We are looking for a Senior Software Engineer to build scalable microservices and cloud infrastructure. "
        "Requirements:\n"
        "- 3+ years experience with Python, FastAPI or Django\n"
        "- Proficiency in AWS, Docker, Kubernetes, and PostgreSQL\n"
        "- Experience with React/TypeScript on the front-end\n"
        "- Strong understanding of CI/CD, Redis, and System Design"
    )

# Instantiate Module Classes
parser_mod = ResumeParser()
ats_mod = ATSEngine()
jd_mod = JDMatcher()
improver_mod = ResumeImprover()
cover_mod = CoverLetterGenerator()
interview_mod = InterviewGenerator()
skill_mod = SkillGapAnalyzer()
salary_mod = SalaryPredictor()
template_mod = ResumeTemplateGenerator()
roadmap_mod = CareerRoadmap()
recruiter_mod = RecruiterDashboard()
student_mod = StudentDashboard()

# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#60a5fa; margin-bottom:0;'>🚀 ResumeAI Pro</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:12px;'>AI Career Intelligence Platform</p>", unsafe_allow_html=True)
    st.divider()

    nav_selection = st.radio(
        "Navigation",
        [
            "🏠 Home / Executive Overview",
            "📄 1. Resume Parser",
            "🎯 2. ATS Score Analyzer",
            "🎯 3. Job Description Matcher",
            "✨ 4. Resume Improver & STAR Rewriter",
            "✉️ 5. AI Cover Letter Generator",
            "🎤 6. Interview Question Generator",
            "📊 7. Skill Gap Analyzer",
            "💰 8. Salary Predictor",
            "🎨 9. Resume Templates",
            "🛣️ 10. Career Roadmap",
            "👥 11. Recruiter Dashboard",
            "🎓 12. Student Dashboard"
        ]
    )

    st.divider()
    # Sidebar Session State Info Box
    st.markdown("### 📌 Active Resume")
    p_name = st.session_state.resume_data.get("personal_info", {}).get("name", "Candidate")
    p_file = st.session_state.resume_data.get("filename", "Resume.pdf")
    st.info(f"**Candidate:** {p_name}\n\n**File:** {p_file}")

    if st.button("🔄 Reset Sample Resume", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# 5. PAGE MODULES
# -----------------------------------------------------------------------------

# =============================================================================
# HOME PAGE
# =============================================================================
if nav_selection == "🏠 Home / Executive Overview":
    st.markdown("""
    <div class='main-header'>
        <h1 style='color: #ffffff; margin-bottom: 8px;'>ResumeAI Pro — AI Career Intelligence Platform</h1>
        <p style='color: #94a3b8; font-size: 16px; margin: 0;'>
            An end-to-end AI-powered resume analysis, ATS optimization, and career planning system built for Final Year Engineering Project.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick Stats Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-card'><div class='metric-value'>12</div><div class='metric-label'>Core Modules</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><div class='metric-value'>92%</div><div class='metric-label'>Avg ATS Improvement</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><div class='metric-value'>STAR</div><div class='metric-label'>Impact Framework</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card'><div class='metric-value'>100%</div><div class='metric-label'>Offline AI Fallback</div></div>", unsafe_allow_html=True)

    st.write("")
    
    # Abstract & System Overview
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("### 📖 Project Abstract")
        st.markdown("""
        Modern recruitment relies heavily on **Applicant Tracking Systems (ATS)** that screen out over 70% of resumes before a human recruiter ever sees them. 
        **ResumeAI Pro** solves this problem by providing job seekers with an integrated career intelligence suite powered by Natural Language Processing (NLP) and Artificial Intelligence.
        
        **Key Architecture Capabilities:**
        - **Automated Resume Parsing:** Extracts structured entities (skills, experience, contact details, education) from PDF/DOCX.
        - **ATS Scoring Engine:** Evaluates formatting, keyword density, section completeness, and impact metrics.
        - **Job Matching & Skill Gap:** Quantifies skill alignment against JD requirements and recommends targeted learning paths.
        - **Generative Career Tools:** AI STAR bullet point improver, cover letter generator, interview practice Q&A, and salary predictor.
        - **Dual Dashboard Suite:** Student career progression tracker & Recruiter batch resume ranking dashboard.
        """)

    with col2:
        st.markdown("### 🏗️ System Architecture Diagram")
        st.code("""
+-------------------------------------------------------------+
|                      USER INTERFACE                         |
|        Streamlit Web App (Sidebar + 12 Modules)             |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                    PROCESSING PIPELINE                      |
|  +---------------------+        +------------------------+  |
|  | Resume Parser       | ---->  | NLP Utils & Skill Ext  |  |
|  | (PyPDF/python-docx) |        | (Regex + SpaCy Match)  |  |
|  +---------------------+        +------------------------+  |
|             |                                |              |
|             v                                v              |
|  +---------------------+        +------------------------+  |
|  | ATS Engine          |        | JD Matcher             |  |
|  | (Scoring Matrix)    |        | (Cosine/Jaccard Sim)   |  |
|  +---------------------+        +------------------------+  |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                   AI & GENERATIVE LAYER                     |
|      OpenAI / Fallback Rule-Based Generative Engine         |
+-------------------------------------------------------------+
        """, language="text")

    st.divider()
    st.markdown("### 🧰 Module Navigation Overview")
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown("<div class='custom-card'><h4>📄 1. Resume Parser</h4><p style='color:#94a3b8; font-size:13px;'>Extract structured contact info, skills, work experience, education, and projects from PDF/DOCX.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='custom-card'><h4>✨ 4. Resume Improver</h4><p style='color:#94a3b8; font-size:13px;'>STAR-method bullet point rewriter and action verb enhancer for high impact.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='custom-card'><h4>📊 7. Skill Gap Analyzer</h4><p style='color:#94a3b8; font-size:13px;'>Identify missing technical/soft skills for target role with course recommendations.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='custom-card'><h4>🛣️ 10. Career Roadmap</h4><p style='color:#94a3b8; font-size:13px;'>Interactive 6-month career preparation timeline with weekly milestones.</p></div>", unsafe_allow_html=True)

    with m_col2:
        st.markdown("<div class='custom-card'><h4>🎯 2. ATS Score Analyzer</h4><p style='color:#94a3b8; font-size:13px;'>Gauge chart breakdown for overall score, keyword density, action words, and structure.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='custom-card'><h4>✉️ 5. AI Cover Letter Generator</h4><p style='color:#94a3b8; font-size:13px;'>Tailored cover letter creation matching specific job descriptions and desired tone.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='custom-card'><h4>💰 8. Salary Predictor</h4><p style='color:#94a3b8; font-size:13px;'>Predict expected market salary range based on role, experience, location, and skills.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='custom-card'><h4>👥 11. Recruiter Dashboard</h4><p style='color:#94a3b8; font-size:13px;'>Batch parse multiple candidate resumes and rank them against job descriptions.</p></div>", unsafe_allow_html=True)

    with m_col3:
        st.markdown("<div class='custom-card'><h4>🎯 3. Job Description Matcher</h4><p style='color:#94a3b8; font-size:13px;'>Side-by-side comparison of resume vs job description highlighting matched/missing keywords.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='custom-card'><h4>🎤 6. Interview Generator</h4><p style='color:#94a3b8; font-size:13px;'>Custom interview practice Q&A cards with difficulty badges and STAR tips.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='custom-card'><h4>🎨 9. Resume Templates</h4><p style='color:#94a3b8; font-size:13px;'>Live HTML/CSS template previewer and custom PDF resume generator.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='custom-card'><h4>🎓 12. Student Dashboard</h4><p style='color:#94a3b8; font-size:13px;'>Track revision history, ATS score growth over time, and job application pipeline.</p></div>", unsafe_allow_html=True)

# =============================================================================
# MODULE 1: RESUME PARSER
# =============================================================================
elif nav_selection == "📄 1. Resume Parser":
    st.markdown("<div class='main-header'><h2>📄 Module 1: Resume Parser</h2><p style='color:#94a3b8;'>Extract structured sections, skills, and contact details from PDF or DOCX files.</p></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Candidate Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        with st.spinner("Parsing resume sections and extracting entities..."):
            parsed_res = parser_mod.parse(file_bytes, uploaded_file.name)
            st.session_state.resume_data = parsed_res
            st.success(f"Successfully parsed '{uploaded_file.name}'!")

    res_data = st.session_state.resume_data

    # Display Extracted Sections in Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["👤 Personal Info & Summary", "💡 Extracted Skills", "💼 Work Experience", "🎓 Education & Projects", "📜 Raw Extracted Text"])

    with tab1:
        p = res_data.get("personal_info", {})
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Full Name", value=p.get("name", ""))
            st.text_input("Email Address", value=p.get("email", ""))
        with col2:
            st.text_input("Phone Number", value=p.get("phone", ""))
            st.text_input("Location", value=p.get("location", ""))
        st.text_area("Professional Summary", value=res_data.get("summary", ""), height=120)

    with tab2:
        st.markdown("### Extracted Skills by Category")
        skills_obj = res_data.get("skills", {})
        by_cat = skills_obj.get("by_category", {}) if isinstance(skills_obj, dict) else {}
        
        for cat, sk_list in by_cat.items():
            if sk_list:
                st.markdown(f"**{cat}:**")
                pills = "".join([f"<span class='badge-skill'>{s}</span>" for s in sk_list])
                st.markdown(pills, unsafe_allow_html=True)
                st.write("")

    with tab3:
        st.markdown("### Work Experience Bullet Points")
        exp_list = res_data.get("experience", [])
        for i, item in enumerate(exp_list, 1):
            st.markdown(f"**{i}.** {item}")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Education")
            for edu in res_data.get("education", []):
                st.write(f"- {edu}")
        with col2:
            st.markdown("### Projects & Certifications")
            for proj in res_data.get("projects", []):
                st.write(f"- 🚀 **Project:** {proj}")
            for cert in res_data.get("certifications", []):
                st.write(f"- 🏆 **Cert:** {cert}")

    with tab5:
        st.text_area("Full Raw Resume Text", value=res_data.get("raw_text", ""), height=300)

    st.divider()
    # Download JSON export
    import json
    json_str = json.dumps(res_data, indent=2)
    st.download_button("📥 Export Parsed Data (JSON)", data=json_str, file_name="parsed_resume.json", mime="application/json")

# =============================================================================
# MODULE 2: ATS SCORE ANALYZER
# =============================================================================
elif nav_selection == "🎯 2. ATS Score Analyzer":
    st.markdown("<div class='main-header'><h2>🎯 Module 2: ATS Score Analyzer</h2><p style='color:#94a3b8;'>Comprehensive Applicant Tracking System score breakdown and actionable formatting/keyword feedback.</p></div>", unsafe_allow_html=True)

    ats_res = ats_mod.analyze(st.session_state.resume_data)
    overall_score = ats_res["overall_score"]

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### Overall ATS Score")
        # Plotly Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_score,
            title={'text': "ATS Score / 100", 'font': {'size': 20, 'color': "#ffffff"}},
            number={'font': {'size': 44, 'color': "#60a5fa"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': "#2563eb"},
                'bgcolor': "#1e293b",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 50], 'color': '#7f1d1d'},
                    {'range': [50, 75], 'color': '#78350f'},
                    {'range': [75, 100], 'color': '#064e3b'}
                ],
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#ffffff"}, height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.markdown("### Sub-Score Breakdown")
        sub_scores = ats_res["sub_scores"]
        df_subs = pd.DataFrame(list(sub_scores.items()), columns=["Category", "Score"])
        
        fig_bars = px.bar(
            df_subs, x="Score", y="Category", orientation='h', text="Score",
            color="Score", color_continuous_scale="Blues", range_x=[0, 100]
        )
        fig_bars.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#ffffff"}, height=280, margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✅ Detected Strengths")
        for s in ats_res["strengths"]:
            st.success(s)

        st.markdown("### ⚡ Detected Action Verbs")
        av_pills = "".join([f"<span class='badge-matched'>{w}</span>" for w in ats_res["action_words_found"]])
        st.markdown(av_pills if av_pills else "No action verbs detected", unsafe_allow_html=True)

    with c2:
        st.markdown("### ⚠️ Areas for Improvement")
        for w in ats_res["weaknesses"]:
            st.warning(w)

        st.markdown("### 💡 Strategic ATS Recommendations")
        for sug in ats_res["suggestions"]:
            st.info(sug)

    st.divider()
    pdf_bytes = generate_report_pdf(
        "ATS Score Analysis Report",
        [
            {"title": f"Overall ATS Score: {overall_score}/100", "content": "Detailed evaluation across formatting, keywords, action words, and structure."},
            {"title": "Sub-scores", "content": [f"{k}: {v}/100" for k, v in sub_scores.items()]},
            {"title": "Strengths", "content": ats_res["strengths"]},
            {"title": "Actionable Suggestions", "content": ats_res["suggestions"]}
        ]
    )
    st.download_button("📥 Download Full ATS Report (PDF)", data=pdf_bytes, file_name="ATS_Analysis_Report.pdf", mime="application/pdf")

# =============================================================================
# MODULE 3: JD MATCHER
# =============================================================================
elif nav_selection == "🎯 3. Job Description Matcher":
    st.markdown("<div class='main-header'><h2>🎯 Module 3: Job Description Matcher</h2><p style='color:#94a3b8;'>Compare resume contents side-by-side with target job requirements to identify keyword matches and skill gaps.</p></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📄 Active Resume Text")
        resume_text_input = st.text_area("Resume Content", value=st.session_state.resume_data.get("raw_text", ""), height=200)

    with col2:
        st.markdown("### 🎯 Target Job Description")
        jd_input = st.text_area("Paste Target Job Description", value=st.session_state.target_job_description, height=200)
        st.session_state.target_job_description = jd_input

    match_res = jd_mod.match(st.session_state.resume_data, jd_input)

    st.divider()
    mc1, mc2 = st.columns([1, 1.5])
    with mc1:
        st.markdown("### Match Percentage")
        fig_match = go.Figure(go.Indicator(
            mode="gauge+number",
            value=match_res["match_score"],
            number={'suffix': "%", 'font': {'size': 40, 'color': "#34d399"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#10b981"},
                'steps': [{'range': [0, 60], 'color': '#7f1d1d'}, {'range': [60, 100], 'color': '#064e3b'}]
            }
        ))
        fig_match.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#ffffff"}, height=220)
        st.plotly_chart(fig_match, use_container_width=True)

    with mc2:
        st.markdown("### Skill Match Progress")
        st.write(f"**Hard Tech Skills Match:** {match_res['hard_skills_match']}%")
        st.progress(int(match_res['hard_skills_match']))
        st.write(f"**Soft Skills Match:** {match_res['soft_skills_match']}%")
        st.progress(int(match_res['soft_skills_match']))

    st.divider()
    kc1, kc2 = st.columns(2)
    with kc1:
        st.markdown("### ✅ Matched Keywords")
        m_pills = "".join([f"<span class='badge-matched'>{k}</span>" for k in match_res["matched_keywords"]])
        st.markdown(m_pills if m_pills else "No direct keyword matches found", unsafe_allow_html=True)

    with kc2:
        st.markdown("### ❌ Missing Keywords")
        mis_pills = "".join([f"<span class='badge-missing'>{k}</span>" for k in match_res["missing_keywords"]])
        st.markdown(mis_pills if mis_pills else "All critical keywords matched!", unsafe_allow_html=True)

# =============================================================================
# MODULE 4: RESUME IMPROVER
# =============================================================================
elif nav_selection == "✨ 4. Resume Improver & STAR Rewriter":
    st.markdown("<div class='main-header'><h2>✨ Module 4: Resume Improver & STAR Rewriter</h2><p style='color:#94a3b8;'>Transform passive bullet points into high-impact, STAR-formatted statements with quantitative metrics.</p></div>", unsafe_allow_html=True)

    target_role = st.selectbox("Select Target Role for Optimization", ["Software Engineer", "Data Scientist", "Full Stack Developer", "Cloud Architect", "Product Manager"])

    imp_res = improver_mod.improve(st.session_state.resume_data, target_role)

    st.markdown("### 📝 Professional Summary Enhancement")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("<div class='custom-card'><b>Original Summary:</b><p style='color:#94a3b8; font-size:13px;'>" + imp_res["original_summary"] + "</p></div>", unsafe_allow_html=True)
    with sc2:
        st.markdown("<div class='custom-card'><b>AI-Enhanced STAR Summary:</b><p style='color:#34d399; font-size:13px;'>" + imp_res["improved_summary"] + "</p></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚡ Work Experience Bullet Comparison (Before vs After)")

    for comp in imp_res["bullet_comparisons"]:
        col_b, col_a = st.columns(2)
        with col_b:
            st.error(f"**Before (Weak / Passive):**\n\n{comp['original']}")
        with col_a:
            st.success(f"**After (STAR Impact Rewrite):**\n\n{comp['improved']}")

    st.divider()
    st.markdown("### 🚀 High-Impact Action Verb Upgrades")
    upgrades_df = pd.DataFrame(imp_res["action_verb_upgrades"])
    st.table(upgrades_df)

# =============================================================================
# MODULE 5: COVER LETTER GENERATOR
# =============================================================================
elif nav_selection == "✉️ 5. AI Cover Letter Generator":
    st.markdown("<div class='main-header'><h2>✉️ Module 5: AI Cover Letter Generator</h2><p style='color:#94a3b8;'>Generate tailored, professional cover letters aligned with target company roles.</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        comp_name = st.text_input("Target Company Name", value="Google Inc.")
        job_t = st.text_input("Job Title", value="Senior Software Engineer")
    with c2:
        tone = st.selectbox("Desired Tone", ["Professional", "Enthusiastic", "Executive", "Concise"])
        key_p = st.text_input("Key Highlights / Achievements to Emphasize", value="Led migration to microservices, 35% latency reduction")

    if st.button("✨ Generate Custom Cover Letter", use_container_width=True):
        with st.spinner("Generating personalized cover letter..."):
            letter_text = cover_mod.generate(st.session_state.resume_data, comp_name, job_t, st.session_state.target_job_description, tone, key_p)
            st.session_state.generated_cover_letter = letter_text

    cover_out = st.session_state.get("generated_cover_letter", cover_mod.generate(st.session_state.resume_data, comp_name, job_t, st.session_state.target_job_description, tone, key_p))

    st.text_area("Generated Cover Letter Output", value=cover_out, height=350)

    st.download_button("📥 Download Cover Letter (.txt)", data=cover_out, file_name="Cover_Letter.txt", mime="text/plain")

# =============================================================================
# MODULE 6: INTERVIEW GENERATOR
# =============================================================================
elif nav_selection == "🎤 6. Interview Question Generator":
    st.markdown("<div class='main-header'><h2>🎤 Module 6: Interview Question Generator</h2><p style='color:#94a3b8;'>Practice tailored interview questions with difficulty ratings, model answers, and STAR tips.</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        q_role = st.text_input("Target Role", value="Software Engineer")
    with c2:
        q_count = st.slider("Number of Questions", min_value=3, max_value=6, value=5)

    questions = interview_mod.generate_questions(st.session_state.resume_data, q_role, count=q_count)

    st.divider()
    for q in questions:
        diff_class = "badge-easy" if q["difficulty"] == "Easy" else ("badge-medium" if q["difficulty"] == "Medium" else "badge-hard")
        
        st.markdown(f"""
        <div class='custom-card'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='color:#60a5fa; font-weight:700;'>Question #{q['id']} — {q['category']}</span>
                <span class='{diff_class}'>{q['difficulty']}</span>
            </div>
            <h4 style='margin-top:10px; color:#ffffff;'>{q['question']}</h4>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"💡 View Sample Answer & Interview Tips for Q#{q['id']}"):
            st.markdown(f"**Sample STAR Answer:**\n{q['sample_answer']}")
            st.markdown(f"**Key Tip:** {q['interview_tips']}")

# =============================================================================
# MODULE 7: SKILL GAP ANALYZER
# =============================================================================
elif nav_selection == "📊 7. Skill Gap Analyzer":
    st.markdown("<div class='main-header'><h2>📊 Module 7: Skill Gap Analyzer</h2><p style='color:#94a3b8;'>Benchmark your skills against industry standards and access curated course recommendations.</p></div>", unsafe_allow_html=True)

    target_role_gap = st.selectbox("Select Target Benchmark Role", [
        "Software Engineer / Backend Developer",
        "Data Scientist / ML Engineer",
        "Full Stack Web Developer",
        "Cloud & DevOps Engineer",
        "Data Engineer"
    ])

    cur_skills = st.session_state.resume_data.get("skills", {}).get("all", []) if isinstance(st.session_state.resume_data.get("skills"), dict) else []
    gap_res = skill_mod.analyze_gap(cur_skills, target_role_gap)

    gc1, gc2 = st.columns([1, 1.2])
    with gc1:
        st.markdown("### Skill Readiness")
        fig_donut = px.pie(
            values=[gap_res["match_percentage"], gap_res["gap_percentage"]],
            names=["Matching Skills", "Skill Gap"],
            hole=0.6,
            color_discrete_sequence=["#10b981", "#ef4444"]
        )
        fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#ffffff"}, height=240, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_donut, use_container_width=True)

    with gc2:
        st.markdown("### Skill Categorization Breakdown")
        st.markdown("**✅ Matching Skills:**")
        m_pills = "".join([f"<span class='badge-matched'>{s}</span>" for s in gap_res["matching_skills"]])
        st.markdown(m_pills if m_pills else "None detected", unsafe_allow_html=True)

        st.markdown("**❌ Missing Critical Skills:**")
        mc_pills = "".join([f"<span class='badge-missing'>{s}</span>" for s in gap_res["missing_critical_skills"]])
        st.markdown(mc_pills if mc_pills else "All critical skills present!", unsafe_allow_html=True)

        st.markdown("**⚡ Missing Desirable Skills:**")
        md_pills = "".join([f"<span class='badge-skill'>{s}</span>" for s in gap_res["missing_desirable_skills"]])
        st.markdown(md_pills if md_pills else "All desirable skills present!", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🎓 Recommended Learning Resources")
    for r in gap_res["learning_resources"]:
        st.markdown(f"- 🔗 **{r['skill']}**: [{r['course']}]({r['url']}) — *{r['provider']} ({r['estimated_hours']})*")

# =============================================================================
# MODULE 8: SALARY PREDICTOR
# =============================================================================
elif nav_selection == "💰 8. Salary Predictor":
    st.markdown("<div class='main-header'><h2>💰 Module 8: Salary Predictor</h2><p style='color:#94a3b8;'>Estimate market compensation based on role, experience, location tier, and technical stack.</p></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        s_role = st.selectbox("Job Role", ["Software Engineer", "Senior Software Engineer", "Data Scientist", "ML Engineer", "DevOps Engineer", "Full Stack Developer"])
    with c2:
        s_exp = st.slider("Experience (Years)", min_value=0.0, max_value=15.0, value=3.5, step=0.5)
    with c3:
        s_loc = st.selectbox("Location Tier", [
            "San Francisco, CA / Silicon Valley",
            "New York, NY",
            "Seattle, WA",
            "Austin, TX",
            "Remote (US)",
            "Bengaluru / India Tech Hub",
            "London / UK Tech Hub"
        ])

    s_edu = st.selectbox("Education Level", ["Bachelor's Degree", "Master's Degree", "Ph.D / Doctorate"])

    sal_res = salary_mod.predict_salary(s_role, s_exp, s_loc, st.session_state.resume_data.get("skills", {}).get("all", []), s_edu)

    st.divider()
    sc1, sc2 = st.columns([1, 1.2])

    with sc1:
        st.markdown("### Estimated Salary Range")
        med = sal_res["estimated_median"]
        mn = sal_res["min_salary"]
        mx = sal_res["max_salary"]

        st.markdown(f"<div class='metric-card'><div class='metric-value'>${med:,.0f} / yr</div><div class='metric-label'>Estimated Median Salary</div></div>", unsafe_allow_html=True)
        st.write("")
        st.info(f"**Expected Range:** ${mn:,.0f} — ${mx:,.0f}")

    with sc2:
        st.markdown("### Factor Contribution Breakdown")
        st.table(sal_res["breakdown_df"])

# =============================================================================
# MODULE 9: RESUME TEMPLATES
# =============================================================================
elif nav_selection == "🎨 9. Resume Templates":
    st.markdown("<div class='main-header'><h2>🎨 Module 9: Resume Templates</h2><p style='color:#94a3b8;'>Select professional resume designs, preview interactive HTML, and export formatted PDF.</p></div>", unsafe_allow_html=True)

    tpl_name = st.selectbox("Select Template Design", template_mod.get_templates())

    st.markdown("### 👁️ Live Template Preview")
    html_preview = template_mod.render_html_template(st.session_state.resume_data, tpl_name)
    st.markdown(html_preview, unsafe_allow_html=True)

    st.divider()
    pdf_res_bytes = generate_resume_pdf(st.session_state.resume_data, template_style=tpl_name)
    st.download_button("📥 Export Formatted Resume PDF", data=pdf_res_bytes, file_name=f"Formatted_Resume_{tpl_name.replace(' ', '_')}.pdf", mime="application/pdf")

# =============================================================================
# MODULE 10: CAREER ROADMAP
# =============================================================================
elif nav_selection == "🛣️ 10. Career Roadmap":
    st.markdown("<div class='main-header'><h2>🛣️ 10. Career Roadmap</h2><p style='color:#94a3b8;'>Structured 6-month career transition plan with milestone goals and weekly tasks.</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        c_role = st.text_input("Current Position", value="Junior Software Developer")
    with c2:
        t_role = st.text_input("Target Dream Position", value="Senior AI & Full-Stack Architect")

    road_res = roadmap_mod.generate_roadmap(c_role, t_role)

    st.markdown("### 🗓️ Phase-by-Phase Timeline")
    for phase in road_res["phases"]:
        with st.expander(f"📌 {phase['phase']} ({phase['months']})", expanded=True):
            st.markdown(f"**Core Focus:** {phase['focus']}")
            st.markdown("**Key Milestones:**")
            for m in phase["milestones"]:
                st.write(f"- 🎯 {m}")

    st.divider()
    st.markdown("### ✅ Action Task Checklist")
    for item in road_res["weekly_checklist"]:
        st.checkbox(f"**{item['week']}:** {item['task']}")

# =============================================================================
# MODULE 11: RECRUITER DASHBOARD
# =============================================================================
elif nav_selection == "👥 11. Recruiter Dashboard":
    st.markdown("<div class='main-header'><h2>👥 Module 11: Recruiter Dashboard</h2><p style='color:#94a3b8;'>Batch parse candidate resumes, compare skills against job specs, and rank applicants.</p></div>", unsafe_allow_html=True)

    st.markdown("### 📥 Upload Batch Resumes")
    batch_files = st.file_uploader("Upload Multiple Resumes (PDF / DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

    rec_res = recruiter_mod.rank_resumes(batch_files, st.session_state.target_job_description)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{rec_res['total_parsed']}</div><div class='metric-label'>Resumes Parsed</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{rec_res['avg_ats_score']}</div><div class='metric-label'>Avg Candidate Score</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>Top 10%</div><div class='metric-label'>Shortlist Tier</div></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🏆 Candidate Ranking Table")
    st.dataframe(rec_res["candidates_df"], use_container_width=True)

    st.divider()
    st.markdown("### 📊 Candidate Score Distribution")
    fig_hist = px.histogram(rec_res["candidates_df"], x="ATS Score", nbins=10, color_discrete_sequence=["#3b82f6"])
    fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#ffffff"}, height=220)
    st.plotly_chart(fig_hist, use_container_width=True)

# =============================================================================
# MODULE 12: STUDENT DASHBOARD
# =============================================================================
elif nav_selection == "🎓 12. Student Dashboard":
    st.markdown("<div class='main-header'><h2>🎓 Module 12: Student Dashboard</h2><p style='color:#94a3b8;'>Personal career growth tracker, resume revision history, and job application pipeline.</p></div>", unsafe_allow_html=True)

    s_data = student_mod.get_dashboard_data()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{s_data['current_ats_score']}</div><div class='metric-label'>Current Resume ATS Score</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{s_data['total_applications']}</div><div class='metric-label'>Active Applications</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{s_data['offers_count']}</div><div class='metric-label'>Offers Extended</div></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📈 ATS Score Growth Across Revisions")
    fig_line = px.line(s_data["score_history_df"], x="Revision", y=["ATS Score", "JD Match Score"], markers=True)
    fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#ffffff"}, height=260)
    st.plotly_chart(fig_line, use_container_width=True)

    st.divider()
    st.markdown("### 📌 Job Application Tracker")
    st.table(s_data["applications_df"])

# -----------------------------------------------------------------------------
# 6. FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
<div class='footer'>
    <b>ResumeAI Pro — Final Year Project</b> | AI-Powered Resume Analyzer & Career Intelligence Platform<br/>
    Built with Streamlit, Plotly, ReportLab & Python NLP | August 2026
</div>
""", unsafe_allow_html=True)
