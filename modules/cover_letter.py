"""
AI Cover Letter Generator for ResumeAI Pro.
Generates tailored cover letters and post-interview thank you notes using LLMs or template fallbacks.
"""

import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try imports for AIEngine
try:
    from resumeai_pro.utils.ai_engine import AIEngine
except ImportError:
    try:
        from utils.ai_engine import AIEngine
    except ImportError:
        AIEngine = None


class CoverLetterGenerator:
    """Generates customized cover letters and thank you notes using AI or templates."""

    def __init__(self, ai_engine: Optional[Any] = None):
        """
        Initialize the CoverLetterGenerator.

        Args:
            ai_engine: Optional instance of AIEngine.
        """
        if ai_engine is not None:
            self.ai_engine = ai_engine
        elif AIEngine is not None:
            try:
                self.ai_engine = AIEngine()
            except Exception as e:
                logger.warning(f"Failed to initialize default AIEngine: {e}")
                self.ai_engine = None
        else:
            self.ai_engine = None

    def _extract_resume_info(self, resume_text: str) -> Dict[str, Any]:
        """Extract key elements from raw resume text for template filling."""
        info = {
            "name": "Applicant",
            "email": "",
            "phone": "",
            "skills": [],
            "years_exp": "several",
            "summary": ""
        }

        if not resume_text or not resume_text.strip():
            return info

        lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
        
        # Extract candidate name (first line or line near top without keywords)
        for line in lines[:5]:
            if not re.search(r'resume|curriculum|cv|email|phone|address|http|github|linkedin', line, re.IGNORECASE):
                if len(line.split()) <= 4 and re.match(r'^[A-Za-z\s\.\'-]+$', line):
                    info["name"] = line.title()
                    break

        # Extract Email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text)
        if email_match:
            info["email"] = email_match.group(0)

        # Extract Phone
        phone_match = re.search(r'(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}', resume_text)
        if phone_match:
            info["phone"] = phone_match.group(0)

        # Extract Skills keywords
        skills_keywords = [
            "Python", "Java", "C++", "JavaScript", "TypeScript", "React", "Node.js", "Angular",
            "SQL", "PostgreSQL", "MongoDB", "AWS", "Docker", "Kubernetes", "Machine Learning",
            "Data Analysis", "Project Management", "Git", "REST API", "DevOps", "Cybersecurity",
            "HTML", "CSS", "Tailwind", "PyTorch", "TensorFlow", "Pandas", "Scikit-Learn"
        ]
        found_skills = []
        for sk in skills_keywords:
            if re.search(r'\b' + re.escape(sk) + r'\b', resume_text, re.IGNORECASE):
                found_skills.append(sk)
        info["skills"] = found_skills[:6] if found_skills else ["problem-solving", "technical analysis", "collaboration"]

        # Extract Years of Experience
        exp_match = re.search(r'(\d+)\+?\s*years?\s*(of)?\s*experience', resume_text, re.IGNORECASE)
        if exp_match:
            info["years_exp"] = f"{exp_match.group(1)}+"

        return info

    def _generate_template_cover_letter(
        self,
        resume_text: str,
        company: str,
        job_title: str,
        job_description: str,
        tone: str = "professional"
    ) -> str:
        """Fallback template-based cover letter generator."""
        info = self._extract_resume_info(resume_text)
        candidate_name = info["name"]
        skills_str = ", ".join(info["skills"]) if info["skills"] else "software development and problem-solving"
        years_exp = info["years_exp"]
        email = info["email"] or "candidate@email.com"
        phone = info["phone"] or "(555) 000-0000"

        # Tone specific openers and closers
        tone_lower = (tone or "professional").lower()

        if tone_lower == "enthusiastic":
            opening = f"I am thrilled to submit my application for the {job_title} position at {company}! Having followed {company}'s innovative work with great admiration, I am eager to bring my passion and skills to your dynamic team."
            closing = f"I would love the opportunity to discuss how my energetic drive and background in {skills_str} can help {company} reach new heights. Thank you for your time and consideration!"
        elif tone_lower == "creative":
            opening = f"As a creative problem-solver passionate about impactful technology, I was immediately drawn to the {job_title} role at {company}."
            closing = f"I am excited about the prospect of building meaningful, cutting-edge solutions at {company}. I welcome the chance to share my portfolio and discuss how we can innovate together."
        elif tone_lower == "formal":
            opening = f"Please accept this letter as formal application for the position of {job_title} at {company}. With {years_exp} years of relevant experience and expertise in {skills_str}, I am confident in my ability to make an immediate contribution to your organization."
            closing = f"Thank you for evaluating my application. I welcome the opportunity to attend a formal interview to discuss my qualifications in greater detail."
        elif tone_lower == "executive":
            opening = f"I am writing to express my strong interest in leading and delivering strategic value as a {job_title} at {company}."
            closing = f"I look forward to discussing how my strategic background and proven track record can drive growth and excellence at {company}."
        else: # Default: professional
            opening = f"I am writing to express my enthusiastic interest in the {job_title} role at {company}. With my background in {skills_str} and over {years_exp} years of practical experience, I am confident in my ability to add significant value to your team."
            closing = f"Thank you for considering my application. I look forward to the opportunity to discuss how my skills, background, and passion align with the goals of {company}."

        jd_highlight = ""
        if job_description and len(job_description.strip()) > 20:
            jd_snippet = job_description.strip()[:150].replace('\n', ' ')
            jd_highlight = f"\n\nBased on your requirement for roles involving '{jd_snippet}...', my core strengths in {skills_str} position me well to meet and exceed your expectations."

        cover_letter = f"""{candidate_name}
Email: {email} | Phone: {phone}

Date: August 7, 2026

Hiring Manager / Talent Acquisition Team
{company}

Dear Hiring Team at {company},

{opening}

Throughout my career, I have developed strong competencies in {skills_str}. My hands-on experience has equipped me with a deep understanding of standard industry practices, efficient workflow execution, and collaborative problem-solving.{jd_highlight}

What excites me most about {company} is your commitment to excellence and innovation. I am driven by challenges that require analytical thinking, continuous learning, and impactful execution. I am confident that my technical proficiency and proactive mindset make me a strong candidate for the {job_title} position.

{closing}

Sincerely,

{candidate_name}"""

        return cover_letter.strip()

    def generate(
        self,
        resume_text: str,
        company: str,
        job_title: str,
        job_description: str = "",
        tone: str = "professional"
    ) -> str:
        """
        Generate a tailored cover letter.

        Args:
            resume_text: Raw text extracted from the candidate's resume.
            company: Target company name.
            job_title: Target job title.
            job_description: Optional job description text.
            tone: Desired tone ('professional', 'enthusiastic', 'creative', 'formal', 'executive').

        Returns:
            Generated cover letter string.
        """
        company = company.strip() if company else "Target Company"
        job_title = job_title.strip() if job_title else "Software Engineer"
        tone = tone.strip().lower() if tone else "professional"

        if self.ai_engine and hasattr(self.ai_engine, "is_available") and self.ai_engine.is_available():
            try:
                system_prompt = (
                    "You are an expert career consultant and professional resume writer. "
                    "Write a compelling, tailored, and error-free cover letter based on the candidate's resume, "
                    "target company, job title, job description, and requested tone. "
                    "Format the output nicely with candidate header, date, recipient, salutation, body, and closing."
                )
                prompt = f"""
Candidate Resume Text:
{resume_text[:2500]}

Target Company: {company}
Target Job Title: {job_title}
Job Description: {job_description[:1500]}
Tone: {tone}

Generate a complete, personalized cover letter in markdown or formatted plain text.
"""
                result = self.ai_engine.generate(prompt, system_prompt=system_prompt, temperature=0.7)
                if result and len(result.strip()) > 100:
                    return result.strip()
            except Exception as e:
                logger.warning(f"AI cover letter generation failed, falling back to template: {e}")

        # Fallback to template generator
        return self._generate_template_cover_letter(resume_text, company, job_title, job_description, tone)

    def generate_thank_you_note(
        self,
        company: str,
        job_title: str,
        interviewer_name: str = "Hiring Manager",
        key_topics: str = ""
    ) -> str:
        """
        Generate a post-interview thank you email.

        Args:
            company: Company name interviewed with.
            job_title: Job title interviewed for.
            interviewer_name: Name of interviewer(s).
            key_topics: Optional specific topics discussed during the interview.

        Returns:
            Formatted thank you email string.
        """
        company = company.strip() if company else "Company"
        job_title = job_title.strip() if job_title else "Role"
        interviewer_name = interviewer_name.strip() if interviewer_name else "Hiring Manager"

        if self.ai_engine and hasattr(self.ai_engine, "is_available") and self.ai_engine.is_available():
            try:
                system_prompt = "You are a professional career coach. Write a polite, engaging, and memorable post-interview thank you email."
                prompt = f"""
Write a post-interview thank you email with the following details:
- Interviewer Name: {interviewer_name}
- Job Title: {job_title}
- Company: {company}
- Key Topics Discussed: {key_topics if key_topics else 'The team vision, technical challenges, and role objectives'}

Include a clear Subject line and Body.
"""
                result = self.ai_engine.generate(prompt, system_prompt=system_prompt, temperature=0.7)
                if result and len(result.strip()) > 50:
                    return result.strip()
            except Exception as e:
                logger.warning(f"AI thank you note generation failed, using template: {e}")

        # Fallback template
        topic_clause = f" I especially enjoyed discussing {key_topics}." if key_topics else " I really enjoyed learning more about the team's ongoing initiatives and vision."

        return f"""Subject: Thank you - {job_title} Interview | [Your Name]

Dear {interviewer_name},

Thank you for taking the time to speak with me today regarding the {job_title} position at {company}.{topic_clause}

Our conversation reinforced my excitement about the opportunity. My background and enthusiasm for delivering impactful solutions align closely with what {company} is seeking.

Please feel free to reach out if you need any additional information or references from my end. I look forward to staying in touch and hearing about the next steps.

Best regards,

[Your Name]
[Your Phone Number]
[Your LinkedIn/Portfolio Link]""".strip()
