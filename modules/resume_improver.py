"""
AI Resume Improver Module for ResumeAI Pro.
Enhances resume summary, bullet points, project descriptions, replaces weak phrasing,
and quantifies achievements using AIEngine and rule-based NLP filters.
"""

import re
from typing import Dict, List, Any, Union, Optional

try:
    from resumeai_pro.utils.ai_engine import AIEngine
    from resumeai_pro.utils.nlp_utils import (
        clean_text,
        get_action_verbs,
        get_weak_words_map
    )
except ImportError:
    from utils.ai_engine import AIEngine
    from utils.nlp_utils import (
        clean_text,
        get_action_verbs,
        get_weak_words_map
    )


class ResumeImprover:
    """
    Optimizes resume text by replacing passive language, adding metrics,
    improving experience bullet points, and polishing professional summaries.
    """

    def __init__(self, ai_engine: Optional[AIEngine] = None):
        self.ai_engine = ai_engine or AIEngine()

    def improve(self, resume_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Master method to optimize an entire resume dataset or raw text.

        :param resume_data: Dictionary from ResumeParser.extract_all() or raw text string
        :return: Dict containing improved summary, experience, projects, and full text
        """
        if isinstance(resume_data, str):
            # Parse simple structure if plain string passed
            sections = {"summary": "", "experience": [], "projects": []}
            summary = self.improve_summary(resume_data[:300])
            improved_text = self.remove_weak_wording(resume_data)
            improved_text = self.quantify_achievements(improved_text)
            return {
                "improved_summary": summary,
                "improved_text": improved_text,
                "original_data": resume_data
            }

        # Handle dictionary input
        sections = resume_data.get("sections", {})
        original_summary = sections.get("summary", "")
        exp_entries = resume_data.get("experience", {}).get("entries", [])
        projects_entries = resume_data.get("projects", [])

        improved_summary = self.improve_summary(original_summary)
        improved_experience = self.improve_bullet_points(exp_entries)
        improved_projects = self.improve_projects(projects_entries)

        # Assemble full improved resume plain text
        formatted_parts = []
        
        # Summary
        if improved_summary:
            formatted_parts.append("PROFESSIONAL SUMMARY\n" + improved_summary)

        # Experience
        if improved_experience:
            formatted_parts.append("WORK EXPERIENCE")
            for entry in improved_experience:
                tc = entry.get("title_company", "")
                if tc:
                    formatted_parts.append(tc)
                for b in entry.get("bullets", []):
                    formatted_parts.append(f"• {b}")

        # Projects
        if improved_projects:
            formatted_parts.append("PROJECTS")
            for p in improved_projects:
                desc = p.get("description", "")
                if desc:
                    formatted_parts.append(f"• {desc}")

        # Skills & Education pass-through
        skills = resume_data.get("skills", [])
        if skills:
            formatted_parts.append("TECHNICAL SKILLS\n" + ", ".join(skills))

        education = resume_data.get("education", [])
        if education:
            edu_str = "\n".join([e.get("raw_text", e.get("degree", "")) for e in education])
            formatted_parts.append("EDUCATION\n" + edu_str)

        full_improved_text = "\n\n".join(formatted_parts)

        return {
            "improved_summary": improved_summary,
            "improved_experience": improved_experience,
            "improved_projects": improved_projects,
            "improved_text": full_improved_text
        }

    def improve_bullet_points(self, experience_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rewrite bullet points with strong action verbs, STAR structure, and quantification.

        :param experience_list: List of experience entry dicts
        :return: Updated list with improved bullets
        """
        improved_entries = []

        for entry in experience_list:
            title_company = entry.get("title_company", "")
            bullets = entry.get("bullets", [])

            improved_bullets = []
            for b in bullets:
                # 1. Eliminate weak phrasing
                b_strong = self.remove_weak_wording(b)
                
                # 2. Add quantification if absent
                b_quant = self.quantify_achievements(b_strong)

                # 3. Apply LLM refinement if available
                if self.ai_engine and self.ai_engine.is_available():
                    b_final = self.ai_engine.improve_bullet_point(b_quant)
                else:
                    b_final = b_quant

                improved_bullets.append(b_final)

            improved_entries.append({
                "title_company": title_company,
                "bullets": improved_bullets
            })

        return improved_entries

    def improve_summary(self, summary: str) -> str:
        """
        Enhance executive summary to be impact-driven, concise, and keyword-rich.

        :param summary: Existing summary text
        :return: Improved summary text
        """
        if not summary or len(summary.strip()) < 10:
            return ("Driven technology professional with expertise in designing scalable architectures, "
                    "optimizing system performance, and delivering high-value software solutions. "
                    "Demonstrated track record of technical innovation and cross-functional team leadership.")

        # Clean and strengthen summary wording
        summary_clean = self.remove_weak_wording(summary)

        if self.ai_engine and self.ai_engine.is_available():
            prompt = f"Enhance this resume summary to make it executive, impact-oriented, and concise:\n'{summary_clean}'"
            res = self.ai_engine.generate(prompt)
            if res:
                return res.strip()

        # Heuristic enhancement
        if not any(word in summary_clean.lower() for word in ["results-driven", "experienced", "accomplished", "proven"]):
            summary_clean = "Results-driven " + summary_clean[0].lower() + summary_clean[1:]

        return summary_clean

    def improve_projects(self, projects_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add quantification, impact framing, and action verbs to project descriptions.

        :param projects_list: List of project dicts
        :return: Improved project dicts list
        """
        improved_projects = []

        for proj in projects_list:
            desc = proj.get("description", "")
            if not desc:
                continue

            # Apply weak wording removal and quantification
            desc_strong = self.remove_weak_wording(desc)
            desc_quant = self.quantify_achievements(desc_strong)

            improved_projects.append({"description": desc_quant})

        return improved_projects

    def quantify_achievements(self, text: str) -> str:
        """
        Detect qualitative claim phrases without numbers and append realistic metric suggestions.

        :param text: Input text line or bullet
        :return: Text with metrics added where missing
        """
        if not text:
            return ""

        # If already contains metrics (digits, %, $, X), return
        if re.search(r'\d+%|\$\d+|\b\d+\+|\b\d+x\b|\b\d+\s*(?:users|ms|seconds|hrs|hours|clients|percent)\b', text, re.I):
            return text

        # Rule-based metrics enhancement based on domain action phrases
        text_lower = text.lower()
        if "performance" in text_lower or "latency" in text_lower or "speed" in text_lower:
            text += " — boosting performance by 35% and reducing latency."
        elif "efficiency" in text_lower or "process" in text_lower or "automation" in text_lower:
            text += " — saving 15+ hours of manual overhead per week."
        elif "revenue" in text_lower or "sales" in text_lower or "cost" in text_lower:
            text += " — driving a 20% growth in operational yield."
        elif "accuracy" in text_lower or "model" in text_lower or "data" in text_lower:
            text += " — achieving 94%+ model accuracy across validation sets."
        elif "user" in text_lower or "client" in text_lower or "customer" in text_lower:
            text += " — serving over 10,000+ active monthly users."
        else:
            text += " — improving operational throughput by 25%."

        return text

    def remove_weak_wording(self, text: str) -> str:
        """
        Replace weak words and phrases ("responsible for", "helped", "worked on", "assisted in")
        with strong power action verbs.

        :param text: Input text string
        :return: Text with weak phrases replaced
        """
        if not text:
            return ""

        weak_map = get_weak_words_map()

        for weak_phrase, replacements in weak_map.items():
            pattern = re.compile(r'\b' + re.escape(weak_phrase) + r'\b', re.I)
            if pattern.search(text):
                # Replace with first strong action verb
                replacement = replacements[0]
                text = pattern.sub(replacement, text)

        # Fix starting sentence case if first word became lowercase replacement
        lines = text.splitlines()
        fixed_lines = []
        for line in lines:
            if line and line[0].islower():
                line = line[0].upper() + line[1:]
            fixed_lines.append(line)

        return "\n".join(fixed_lines)
