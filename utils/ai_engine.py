"""
AI Engine layer for ResumeAI Pro.
Provides OpenAI LLM integration with intelligent rule-based fallbacks
for bullet improvement, summary enhancement, cover letter generation,
interview preparation, career roadmap generation, and career chatbot.
"""

import os
import re
import json
import random
from typing import Dict, List, Optional, Union, Any

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class AIEngine:
    """
    AI Engine for ResumeAI Pro platform.
    Uses OpenAI API if OPENAI_API_KEY is available; otherwise falls back to
    intelligent rule-based algorithms.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize AIEngine instance.
        
        Args:
            api_key: Optional OpenAI API key string.
            model: Target OpenAI model (default: gpt-4o-mini).
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None

        if _OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None

    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> Optional[str]:
        """
        Internal helper to execute OpenAI Chat Completion safely.
        
        Returns string content if call succeeds, or None if client unavailable / error occurs.
        """
        if not self.client:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None

    def improve_bullet_point(self, bullet: str) -> str:
        """
        Rewrite a weak or plain bullet point using strong action verbs, quantifiable metrics, and impact.
        
        Args:
            bullet: Original bullet point string.
            
        Returns:
            Enhanced bullet point string.
        """
        if not bullet or not bullet.strip():
            return "Engineered core modules and optimized workflow performance by 25%."

        system_prompt = (
            "You are an expert resume writer. Improve the given resume bullet point. "
            "Start with a high-impact action verb, quantify results with percentages or metrics, "
            "and clearly state the technical impact. Return ONLY the improved bullet point without quotes or conversational text."
        )
        user_prompt = f"Original bullet point: {bullet}"

        llm_result = self._call_llm(system_prompt, user_prompt, temperature=0.5)
        if llm_result:
            return llm_result.strip('"-• ')

        # --- Rule-based Fallback ---
        weak_verbs = [
            "worked on", "helped with", "responsible for", "handled", "did", "assisted in", 
            "was involved in", "used", "made", "created", "built", "wrote"
        ]

        action_verbs = [
            "Architected", "Engineered", "Spearheaded", "Optimized", "Streamlined", 
            "Implemented", "Deployed", "Designed", "Executed", "Formulated"
        ]

        cleaned = bullet.strip().rstrip('.')
        lowered = cleaned.lower()

        # Replace weak leading verbs
        action_verb = random.choice(action_verbs)
        for weak in weak_verbs:
            if lowered.startswith(weak):
                cleaned = action_verb + cleaned[len(weak):]
                break
        else:
            if not any(cleaned.startswith(v) for v in action_verbs):
                cleaned = f"{action_verb} and {cleaned[0].lower() + cleaned[1:]}"

        # Add quantification if missing numbers
        if not re.search(r'\d+%|\d+\+|\$\d+', cleaned):
            metrics = [
                "improving execution efficiency by 35%",
                "reducing processing overhead by 25%",
                "enhancing system scalability across 10,000+ active users",
                "decreasing latency by 40% while ensuring 99.9% uptime"
            ]
            cleaned = f"{cleaned}, {random.choice(metrics)}"

        return cleaned + "."

    def improve_summary(self, summary: str) -> str:
        """
        Enhance a professional summary to be concise, high-impact, and ATS-optimized.
        
        Args:
            summary: Original summary text.
            
        Returns:
            Polished summary string.
        """
        if not summary or not summary.strip():
            summary = "Software Engineer with experience in web development, database design, and cloud deployments."

        system_prompt = (
            "You are an executive career coach. Enhance the given professional summary. "
            "Make it compelling, ATS-optimized, narrative-driven (3-4 sentences), highlighting core competencies, "
            "leadership, and problem-solving impact. Return ONLY the improved summary."
        )
        user_prompt = f"Original summary: {summary}"

        llm_result = self._call_llm(system_prompt, user_prompt, temperature=0.6)
        if llm_result:
            return llm_result.strip()

        # --- Rule-based Fallback ---
        clean_summary = summary.strip().rstrip('.')
        
        # Detect keywords in summary
        has_data = any(w in clean_summary.lower() for w in ['data', 'analytics', 'python', 'sql', 'machine learning'])
        has_lead = any(w in clean_summary.lower() for w in ['lead', 'manage', 'team', 'direct', 'head'])

        role = "Data & AI Professional" if has_data else ("Technical Leader" if has_lead else "Results-Driven Software Professional")

        enhanced = (
            f"{role} with hands-on expertise in delivering scalable, end-to-end technical solutions. "
            f"Demonstrated ability in {clean_summary[0].lower() + clean_summary[1:] if len(clean_summary) > 10 else 'building high-performance applications'}. "
            f"Adept at cross-functional collaboration, system optimization, and translating complex business requirements into high-impact products. "
            f"Passionate about continuous learning and leveraging modern technology stacks to drive organizational growth."
        )
        return enhanced

    def generate_cover_letter(
        self, 
        resume_text: str, 
        company: str, 
        job_title: str, 
        job_description: str
    ) -> str:
        """
        Generate a personalized cover letter matching resume experience with target job details.
        
        Args:
            resume_text: Candidate's full resume text.
            company: Name of target company.
            job_title: Title of target role.
            job_description: Job description text.
            
        Returns:
            Formatted cover letter string.
        """
        company = company.strip() if company else "Target Company"
        job_title = job_title.strip() if job_title else "Target Role"

        system_prompt = (
            "You are a professional cover letter writer. Generate a tailored, 4-paragraph cover letter "
            "for the candidate based on their resume and the job description. "
            "Include Salutation, Opening Hook, Alignment of Skills & Experience, and Strong Closing Call to Action."
        )
        user_prompt = (
            f"Company: {company}\nJob Title: {job_title}\n"
            f"Job Description: {job_description}\n\nCandidate Resume:\n{resume_text}"
        )

        llm_result = self._call_llm(system_prompt, user_prompt, temperature=0.7)
        if llm_result:
            return llm_result.strip()

        # --- Rule-based Fallback ---
        # Extract skills from resume
        skill_words = re.findall(r'\b[A-Z][a-zA-Z0-9+#.]{2,}\b', resume_text or "")
        unique_skills = list(dict.fromkeys([s for s in skill_words if s.lower() not in ['summary', 'experience', 'education', 'skills', 'jan', 'feb']]))[:5]
        top_skills_str = ", ".join(unique_skills) if unique_skills else "software engineering, system design, and project execution"

        cover_letter = f"""Dear Hiring Team at {company},

I am writing to express my strong enthusiasm for the {job_title} position at {company}. With a solid technical background and a proven track record of solving complex problems, I am confident in my ability to make an immediate, positive impact on your team.

Throughout my career, I have developed expertise in {top_skills_str}. My experience aligns closely with your core requirements, particularly in designing scalable applications, optimizing workflow efficiency, and collaborating effectively across teams to deliver high-quality solutions on schedule.

What excites me most about {company} is your commitment to technical innovation and quality. Reviewing the description for the {job_title} role, I see a clear alignment between your team's upcoming goals and my hands-on background in driving technical initiatives from concept to production.

Thank you for your time and consideration. I welcome the opportunity to discuss how my background, technical skills, and passion can contribute to the ongoing success of {company}.

Sincerely,
Candidate
"""
        return cover_letter

    def generate_interview_questions(
        self, 
        resume_text: str, 
        job_description: str, 
        question_type: str = "All", 
        difficulty: str = "Medium", 
        count: int = 5
    ) -> List[Dict[str, str]]:
        """
        Generate target interview questions with sample answers and preparation tips.
        
        Args:
            resume_text: Candidate's resume text.
            job_description: Job description text.
            question_type: 'HR', 'Technical', 'Behavioral', 'Coding', 'Project', or 'All'.
            difficulty: 'Easy', 'Medium', or 'Hard'.
            count: Number of questions to return (default: 5).
            
        Returns:
            List of dicts: [{'question': ..., 'type': ..., 'sample_answer': ..., 'tips': ...}]
        """
        system_prompt = (
            "You are an executive interviewer. Generate interview questions based on candidate resume and job description. "
            "Return a JSON array of objects, where each object has keys: 'question', 'type', 'sample_answer', 'tips'."
        )
        user_prompt = (
            f"Question Type: {question_type}\nDifficulty: {difficulty}\nCount: {count}\n"
            f"Job Description: {job_description}\nResume:\n{resume_text}"
        )

        llm_result = self._call_llm(system_prompt, user_prompt, temperature=0.7)
        if llm_result:
            try:
                # Parse JSON array out of response
                json_match = re.search(r'\[.*\]', llm_result, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, list) and len(parsed) > 0:
                        return parsed[:count]
            except Exception:
                pass

        # --- Rule-based Fallback ---
        question_bank = [
            {
                "question": f"Can you walk me through a complex technical project on your resume and how you handled key architectural decisions?",
                "type": "Project",
                "sample_answer": "I select a major project, outline the initial problem, describe my role in choosing the tech stack, explain trade-offs made during development, and quantify the resulting outcome.",
                "tips": "Use the STAR method (Situation, Task, Action, Result) and highlight specific technical choices."
            },
            {
                "question": f"How do you approach debugging and optimizing a slow system or pipeline under tight deadlines?",
                "type": "Technical",
                "sample_answer": "I isolate the bottleneck using profiling tools and metrics logs, formulate hypotheses, test incremental optimizations in staging, and monitor post-deployment performance.",
                "tips": "Emphasize systematic troubleshooting over guesswork."
            },
            {
                "question": f"Describe a situation where you had a conflict or difference of technical opinion with a team member. How did you resolve it?",
                "type": "Behavioral",
                "sample_answer": "I focused on data and objective trade-off analysis rather than personal opinion, conducted benchmark tests, and worked towards consensus aligned with project timelines.",
                "tips": "Focus on active listening, empathy, and team alignment."
            },
            {
                "question": f"Why are you interested in joining our company in the {difficulty} level {question_type if question_type != 'All' else 'Engineering'} role?",
                "type": "HR",
                "sample_answer": "I admire your recent engineering work and growth trajectory. My background in software design and automated testing aligns perfectly with your goals.",
                "tips": "Demonstrate knowledge of the company mission and connect your career goals."
            },
            {
                "question": f"How would you design a rate-limiting service or caching layer to handle high traffic spikes?",
                "type": "Coding",
                "sample_answer": "I would use Redis with a Sliding Window or Token Bucket algorithm, set appropriate TTLs, and ensure fallback gracefully under database load.",
                "tips": "Discuss edge cases like race conditions, memory limits, and distribution."
            },
            {
                "question": f"How do you ensure data integrity, unit testing, and code quality in continuous integration pipelines?",
                "type": "Technical",
                "sample_answer": "I enforce automated unit and integration test suites in GitHub Actions/CI, set minimum coverage thresholds, and conduct peer code reviews before merging.",
                "tips": "Mention specific CI/CD tools and code coverage strategies."
            }
        ]

        if question_type != "All":
            filtered = [q for q in question_bank if q["type"].lower() == question_type.lower()]
            if filtered:
                question_bank = filtered

        # Return requested count
        results = []
        for i in range(count):
            item = question_bank[i % len(question_bank)].copy()
            results.append(item)

        return results

    def generate_career_roadmap(
        self, 
        current_skills: List[str], 
        target_role: str, 
        timeline_weeks: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Generate a structured weekly learning roadmap to bridge skill gaps for a target role.
        
        Args:
            current_skills: List of skills candidate already possesses.
            target_role: Desired job title (e.g., 'Data Scientist', 'Full Stack Engineer').
            timeline_weeks: Total timeline duration in weeks (default: 12).
            
        Returns:
            List of dicts representing roadmap phases/weeks.
        """
        system_prompt = (
            "You are a senior tech mentor. Generate a weekly career learning roadmap. "
            "Return a JSON array of week objects, with keys: 'week', 'topic', 'focus_areas', 'recommended_resources', 'action_items'."
        )
        user_prompt = (
            f"Target Role: {target_role}\nTimeline: {timeline_weeks} weeks\n"
            f"Current Skills: {', '.join(current_skills)}"
        )

        llm_result = self._call_llm(system_prompt, user_prompt, temperature=0.6)
        if llm_result:
            try:
                json_match = re.search(r'\[.*\]', llm_result, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, list) and len(parsed) > 0:
                        return parsed
            except Exception:
                pass

        # --- Rule-based Fallback ---
        role_lower = target_role.lower() if target_role else "software engineer"

        if "data" in role_lower or "ai" in role_lower or "ml" in role_lower:
            phases = [
                ("Advanced Python, Mathematics & Pandas/NumPy", ["Data Structures", "Statistical Analysis", "Data Wrangling"]),
                ("Machine Learning & Model Building with Scikit-Learn", ["Supervised Learning", "Evaluation Metrics", "Feature Engineering"]),
                ("Deep Learning & LLM Foundations (PyTorch / OpenAI)", ["Neural Networks", "Fine-tuning", "RAG Pipelines"]),
                ("MLOps, Model Deployment & Portfolio Project", ["FastAPI", "Docker", "Model Monitoring", "Cap-Stone Project"])
            ]
        elif "cloud" in role_lower or "devops" in role_lower:
            phases = [
                ("Linux Administration & Containerization", ["Shell Scripting", "Docker", "Docker Compose"]),
                ("Cloud Architecture & CI/CD Pipelines", ["AWS/GCP Services", "GitHub Actions", "Terraform"]),
                ("Kubernetes & Cluster Management", ["EKS/GKS", "Helm Charts", "Monitoring with Prometheus"]),
                ("Production Security & Enterprise Deployment", ["IAM Policies", "Zero Trust", "DevSecOps Practice"])
            ]
        else:
            phases = [
                ("Core Stack & Modern Framework Fundamentals", ["Language Concepts", "REST APIs", "Database Design"]),
                ("Backend & Frontend System Integration", ["State Management", "Authentication", "Performance Tuning"]),
                ("Scalability, Cloud Infrastructure & DevOps", ["Microservices", "Docker", "CI/CD Setup"]),
                ("Real-World Capstone Application & Interview Prep", ["Full Stack Build", "System Design", "Mock Interviews"])
            ]

        # Distribute timeline_weeks across phases
        num_phases = len(phases)
        weeks_per_phase = max(1, timeline_weeks // num_phases)

        roadmap = []
        week_counter = 1

        for idx, (topic, focus) in enumerate(phases):
            duration = weeks_per_phase if idx < num_phases - 1 else (timeline_weeks - week_counter + 1)
            end_week = week_counter + duration - 1

            roadmap.append({
                "week": f"Week {week_counter}" if duration == 1 else f"Weeks {week_counter}-{end_week}",
                "topic": topic,
                "focus_areas": focus,
                "recommended_resources": [
                    "Official Documentation & Framework Guides",
                    "Coursera / Udemy Hands-on Projects",
                    "GitHub Open Source Repositories"
                ],
                "action_items": [
                    f"Build a mini project practicing {focus[0]}",
                    "Write technical documentation and push code to GitHub",
                    "Complete self-assessment quizzes and practice problems"
                ]
            })
            week_counter = end_week + 1

        return roadmap

    def career_chatbot(self, message: str, context: Optional[Union[Dict[str, Any], str]] = None) -> str:
        """
        AI Career Advice Chatbot.
        
        Args:
            message: User query or prompt.
            context: Context details (resume summary, target role, etc.).
            
        Returns:
            Chatbot response string.
        """
        if not message or not message.strip():
            return "Hello! I am your AI Career Advisor. How can I help you with your resume, interviews, or career growth today?"

        ctx_str = json.dumps(context) if isinstance(context, dict) else str(context or "")
        system_prompt = (
            "You are ResumeAI Pro's AI Career Advisor. Give clear, actionable, friendly, "
            "and professional career guidance regarding resumes, job search, interview prep, or skill roadmap."
        )
        user_prompt = f"User Context: {ctx_str}\n\nUser Question: {message}"

        llm_result = self._call_llm(system_prompt, user_prompt, temperature=0.7)
        if llm_result:
            return llm_result.strip()

        # --- Rule-based Fallback ---
        msg_lower = message.lower()

        if any(w in msg_lower for w in ['resume', 'bullet', 'experience', 'cv']):
            return (
                "To optimize your resume:\n"
                "1. **Start with strong action verbs** (e.g. Engineered, Spearheaded, Architected).\n"
                "2. **Quantify achievements** with metrics (% increase, hours saved, user scale).\n"
                "3. **Keep formatting ATS-friendly** by using standard section titles (Experience, Education, Skills).\n"
                "4. Tailor keywords directly to the job description."
            )
        elif any(w in msg_lower for w in ['interview', 'question', 'behavioral', 'star']):
            return (
                "For interview success:\n"
                "1. Use the **STAR method** (Situation, Task, Action, Result) for behavioral questions.\n"
                "2. Review core technical concepts and system design fundamentals.\n"
                "3. Prepare 2-3 questions for the interviewer about team culture and engineering roadmap.\n"
                "4. Practice mock interviews aloud or using our AI Interview Question generator!"
            )
        elif any(w in msg_lower for w in ['salary', 'negotiat', 'offer', 'pay']):
            return (
                "When negotiating salary:\n"
                "1. Research market rates on Levels.fyi, Glassdoor, and Payscale.\n"
                "2. Focus on total compensation (base, bonus, equity, signing bonus).\n"
                "3. Express enthusiasm first before sharing counter-offers grounded in data."
            )
        elif any(w in msg_lower for w in ['skill', 'learn', 'roadmap', 'framework']):
            return (
                "To master new skills efficiently:\n"
                "1. Focus on hands-on project building rather than passive video watching.\n"
                "2. Publish projects on GitHub with clean README files and live demos.\n"
                "3. Build in public and document your learning journey on LinkedIn."
            )
        else:
            return (
                "That's a great career question! To achieve your professional goals, I recommend:\n"
                "1. Keeping your technical skills aligned with high-demand job market requirements.\n"
                "2. Quantifying your impact across all projects and work experience.\n"
                "3. Networking actively on LinkedIn with recruiters and senior engineers in your field.\n\n"
                "Let me know if you would like specific help with your resume, cover letter, or interview preparation!"
            )
