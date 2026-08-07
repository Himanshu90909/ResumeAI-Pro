"""
ATS Score Engine Module for ResumeAI Pro.
Calculates overall ATS score and granular sub-scores with feedback reports.
"""

import re
from typing import Dict, List, Any, Optional

try:
    from resumeai_pro.utils.nlp_utils import (
        clean_text,
        extract_contact_info,
        extract_sections,
        extract_skills,
        extract_education,
        extract_years_experience,
        calculate_flesch_reading_ease,
        get_action_verbs,
        get_skills_database
    )
except ImportError:
    from utils.nlp_utils import (
        clean_text,
        extract_contact_info,
        extract_sections,
        extract_skills,
        extract_education,
        extract_years_experience,
        calculate_flesch_reading_ease,
        get_action_verbs,
        get_skills_database
    )


class ATSEngine:
    """
    Evaluates resumes against Applicant Tracking System (ATS) criteria.
    Calculates 8 specific sub-scores and overall readiness score.
    """

    # Weights for sub-score aggregation
    WEIGHTS: Dict[str, float] = {
        "keywords": 0.20,
        "experience": 0.18,
        "skills": 0.17,
        "formatting": 0.12,
        "education": 0.10,
        "recruiter_compatibility": 0.10,
        "readability": 0.08,
        "grammar": 0.05,
    }

    def score(self, resume_text: str, job_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate resume text and calculate overall ATS score and 8 sub-scores.

        :param resume_text: Plain text of resume
        :param job_description: Optional job description text
        :return: Dict containing 'overall_score' and 'sub_scores' dict
        """
        cleaned = clean_text(resume_text)
        if not cleaned:
            return {
                "overall_score": 0.0,
                "sub_scores": {
                    "formatting": 0.0,
                    "keywords": 0.0,
                    "skills": 0.0,
                    "experience": 0.0,
                    "education": 0.0,
                    "readability": 0.0,
                    "grammar": 0.0,
                    "recruiter_compatibility": 0.0
                }
            }

        sections = extract_sections(cleaned)
        contact = extract_contact_info(cleaned)

        sub_scores = {
            "formatting": self._score_formatting(cleaned, sections),
            "keywords": self._score_keywords(cleaned, job_description),
            "skills": self._score_skills(cleaned),
            "experience": self._score_experience(cleaned, sections),
            "education": self._score_education(cleaned, sections),
            "readability": self._score_readability(cleaned),
            "grammar": self._score_grammar(cleaned),
            "recruiter_compatibility": self._score_recruiter_compatibility(cleaned, sections, contact)
        }

        # Calculate weighted overall score
        overall = sum(sub_scores[key] * self.WEIGHTS[key] for key in self.WEIGHTS)
        overall_score = round(max(0.0, min(100.0, overall)), 1)

        return {
            "overall_score": overall_score,
            "sub_scores": sub_scores
        }

    def _score_formatting(self, text: str, sections: Dict[str, str]) -> float:
        """
        Sub-score for proper section structure, bullet usage, and document layout.
        """
        score = 0.0

        # Standard required sections (up to 60 pts)
        expected = ["summary", "experience", "education", "skills", "projects"]
        present_count = sum(1 for sec in expected if sections.get(sec, "").strip())
        score += (present_count / len(expected)) * 60.0

        # Bullet point formatting (up to 20 pts)
        bullet_matches = len(re.findall(r'^\s*[-\*•]\s+', text, re.M))
        if bullet_matches >= 5:
            score += 20.0
        elif bullet_matches > 0:
            score += 10.0

        # Text line density / formatting hygiene (up to 20 pts)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        avg_line_len = sum(len(line) for line in lines) / max(1, len(lines))
        if 20 <= avg_line_len <= 100:
            score += 20.0
        else:
            score += 10.0

        return round(min(100.0, max(0.0, score)), 1)

    def _score_keywords(self, text: str, job_description: Optional[str] = None) -> float:
        """
        Sub-score for keyword matching against JD or general high-value ATS terms.
        """
        text_lower = text.lower()

        if job_description and job_description.strip():
            jd_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', job_description.lower()))
            stopwords = {"the", "and", "for", "with", "that", "this", "from", "have", "you", "are", "will", "our", "all"}
            filtered_jd_words = jd_words - stopwords

            if not filtered_jd_words:
                return 70.0

            matches = sum(1 for word in filtered_jd_words if word in text_lower)
            match_ratio = matches / len(filtered_jd_words)
            # Match ratio scaled to 100 (e.g. 40% match = ~80 score)
            score = min(100.0, match_ratio * 200.0)
            return round(score, 1)

        # General ATS industry keyword dictionary if no JD supplied
        common_ats_keywords = [
            "development", "management", "engineering", "analysis", "design", "testing",
            "architecture", "optimization", "strategy", "implementation", "agile",
            "database", "api", "project", "collaboration", "integration", "performance"
        ]
        matched = sum(1 for kw in common_ats_keywords if kw in text_lower)
        score = (matched / len(common_ats_keywords)) * 100.0
        return round(min(100.0, max(0.0, score)), 1)

    def _score_skills(self, text: str) -> float:
        """
        Sub-score for counting relevant technical/soft skills and category diversity.
        """
        skills = extract_skills(text)
        num_skills = len(skills)

        # Skill volume score (up to 70 pts)
        if num_skills >= 12:
            volume_score = 70.0
        elif num_skills >= 8:
            volume_score = 55.0
        elif num_skills >= 4:
            volume_score = 40.0
        elif num_skills >= 1:
            volume_score = 25.0
        else:
            volume_score = 0.0

        # Category diversity score (up to 30 pts)
        taxonomy = get_skills_database()
        categories_found = set()
        skills_lower = [s.lower() for s in skills]

        for category, cat_skills in taxonomy.items():
            for cs in cat_skills:
                if cs.lower() in skills_lower:
                    categories_found.add(category)
                    break

        diversity_score = min(30.0, len(categories_found) * 7.5)
        return round(min(100.0, volume_score + diversity_score), 1)

    def _score_experience(self, text: str, sections: Dict[str, str]) -> float:
        """
        Sub-score based on years of experience, action verb usage, and quantification.
        """
        years = extract_years_experience(text)
        exp_text = sections.get("experience", "") or text

        # Years of experience contribution (up to 50 pts)
        exp_years_score = min(50.0, years * 10.0)

        # Action verbs usage (up to 25 pts)
        action_verbs = get_action_verbs()
        words = re.findall(r'\b[a-zA-Z]+\b', exp_text.lower())
        action_count = sum(1 for w in words if w in action_verbs)
        verb_score = min(25.0, action_count * 2.5)

        # Metrics / Quantification presence (up to 25 pts)
        metrics = re.findall(r'\b\d+%\b|\$\d+|\b\d+\+\b|\b\d+x\b|\b\d+\s*(?:users|clients|projects|million|k)\b', exp_text, re.I)
        metric_score = min(25.0, len(metrics) * 8.33)

        return round(min(100.0, exp_years_score + verb_score + metric_score), 1)

    def _score_education(self, text: str, sections: Dict[str, str]) -> float:
        """
        Sub-score for presence of degree, university, and relevant coursework.
        """
        edu_text = sections.get("education", "") or text
        education_entries = extract_education(edu_text)

        if not education_entries:
            return 30.0

        score = 60.0  # Base for having a recognized degree entry

        # Check degree level
        degrees_str = " ".join([e.get("degree", "") for e in education_entries]).lower()
        if "master" in degrees_str or "ph.d" in degrees_str:
            score += 20.0
        elif "bachelor" in degrees_str:
            score += 15.0

        # Check for institution / university mentions
        if re.search(r'university|college|institute|school|academy', edu_text, re.I):
            score += 10.0

        # Check for graduation year or GPA
        if re.search(r'\b(19\d{2}|20\d{2})\b|gpa', edu_text, re.I):
            score += 10.0

        return round(min(100.0, score), 1)

    def _score_readability(self, text: str) -> float:
        """
        Sub-score based on normalized Flesch Reading Ease score.
        Target FRE for technical/professional documents is 30 to 70.
        """
        fre = calculate_flesch_reading_ease(text)

        if 40.0 <= fre <= 70.0:
            return 95.0
        elif 30.0 <= fre < 40.0 or 70.0 < fre <= 80.0:
            return 80.0
        elif 20.0 <= fre < 30.0 or 80.0 < fre <= 90.0:
            return 65.0
        else:
            return 50.0

    def _score_grammar(self, text: str) -> float:
        """
        Basic grammar, capitalization, and formatting mechanics sub-score.
        """
        score = 100.0

        # Penalty for double spaces
        double_spaces = len(re.findall(r' {2,}', text))
        score -= min(15.0, double_spaces * 1.5)

        # Penalty for missing space after punctuation
        punct_errors = len(re.findall(r'[.,;:!][a-zA-Z]', text))
        score -= min(15.0, punct_errors * 2.0)

        # Check line start capitalization
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            uncap_starts = sum(1 for line in lines if line[0].islower() and not line.startswith(("http", "www", "mailto")))
            score -= min(20.0, uncap_starts * 2.5)

        return round(max(0.0, score), 1)

    def _score_recruiter_compatibility(self, text: str, sections: Dict[str, str], contact: Dict[str, Any]) -> float:
        """
        Sub-score for executive presentation, clear contact channels, and readability balance.
        """
        score = 0.0

        # Contact detail completeness (up to 40 pts)
        if contact.get("email"):
            score += 15.0
        if contact.get("phone"):
            score += 10.0
        if contact.get("linkedin") or contact.get("github") or contact.get("website"):
            score += 15.0

        # Professional summary present (20 pts)
        if sections.get("summary", "").strip():
            score += 20.0

        # Experience density balance (20 pts)
        if sections.get("experience", "").strip():
            score += 20.0

        # Length check (1 to 2 pages equivalent in text: 300 to 1200 words) (20 pts)
        word_count = len(re.findall(r'\b\w+\b', text))
        if 300 <= word_count <= 1200:
            score += 20.0
        elif word_count > 100:
            score += 10.0

        return round(min(100.0, score), 1)

    def generate_report(self, scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a formatted evaluation report dict with sub-score feedback and suggestions.

        :param scores: Dictionary produced by score()
        :return: Structured report dict
        """
        overall = scores.get("overall_score", 0.0)
        sub = scores.get("sub_scores", {})

        # Letter grade determination
        if overall >= 90:
            grade = "A+"
        elif overall >= 80:
            grade = "A"
        elif overall >= 70:
            grade = "B"
        elif overall >= 60:
            grade = "C"
        else:
            grade = "D"

        category_suggestions = {}
        strengths = []
        improvements = []

        # Category specific advice logic
        feedback_rules = {
            "formatting": (
                "Ensure clean standard headers (Summary, Experience, Education, Skills, Projects) and consistent bullet points.",
                "Structure resume into standard headers and use uniform bullet points."
            ),
            "keywords": (
                "Excellent keyword alignment detected.",
                "Incorporate more industry keywords and terms matching target job descriptions."
            ),
            "skills": (
                "Strong technical and soft skill variety displayed.",
                "Add more hard technical skills, tools, and categorized competencies."
            ),
            "experience": (
                "Compelling experience section with action verbs and quantifiable results.",
                "Start each bullet point with strong action verbs and include metrics (%, $, numbers)."
            ),
            "education": (
                "Clear education details and degree credentials.",
                "Clearly specify degree title, institution, graduation year, and relevant coursework."
            ),
            "readability": (
                "Optimal readability ease score for recruiter scanning.",
                "Shorten overly long sentences and break text blocks into concise bullet points."
            ),
            "grammar": (
                "Flawless grammar and punctuation mechanics.",
                "Fix double spacing, missing punctuation spaces, and lowercase bullet starts."
            ),
            "recruiter_compatibility": (
                "Highly professional layout with complete contact information and executive appeal.",
                "Include LinkedIn link, phone number, professional email, and an impact-driven summary."
            )
        }

        for cat, val in sub.items():
            good_msg, bad_msg = feedback_rules.get(cat, ("Good performance.", "Needs improvement."))
            if val >= 75.0:
                category_suggestions[cat] = f"Good ({val}/100): {good_msg}"
                strengths.append(cat.replace("_", " ").title())
            else:
                category_suggestions[cat] = f"Needs Improvement ({val}/100): {bad_msg}"
                improvements.append(cat.replace("_", " ").title())

        return {
            "overall_score": overall,
            "grade": grade,
            "summary": f"Your resume scored {overall}/100 ({grade} grade) for overall ATS compatibility.",
            "sub_scores": sub,
            "strengths": strengths,
            "improvements": improvements,
            "category_suggestions": category_suggestions
        }
