"""
Job Description Matcher Module for ResumeAI Pro.
Performs semantic comparison between resume and job description using TF-IDF,
skill gap analysis, experience/education validation, and compatibility scoring.
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from resumeai_pro.utils.nlp_utils import (
        clean_text,
        extract_skills,
        extract_years_experience,
        extract_education,
        get_skills_database
    )
except ImportError:
    from utils.nlp_utils import (
        clean_text,
        extract_skills,
        extract_years_experience,
        extract_education,
        get_skills_database
    )


class JDMatcher:
    """
    Compares candidate resume text against job description specifications.
    Calculates TF-IDF cosine similarity, keyword match, and gap metrics.
    """

    DEGREE_HIERARCHY: Dict[str, int] = {
        "High School Diploma": 1,
        "Associate's": 2,
        "Bachelor's": 3,
        "Master's": 4,
        "Ph.D.": 5
    }

    def match(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Execute comprehensive resume vs. JD match analysis.

        :param resume_text: Resume text
        :param job_description: Target job description text
        :return: Match evaluation result dict
        """
        cleaned_resume = clean_text(resume_text)
        cleaned_jd = clean_text(job_description)

        if not cleaned_resume or not cleaned_jd:
            return {
                "keyword_match_pct": 0.0,
                "semantic_similarity_pct": 0.0,
                "missing_keywords": [],
                "missing_skills": [],
                "experience_gap": {
                    "required_years": 0.0,
                    "candidate_years": 0.0,
                    "gap_years": 0.0,
                    "meets_requirement": True,
                    "note": "Insufficient data"
                },
                "education_gap": {
                    "required_degree": "Unspecified",
                    "candidate_degree": "Unspecified",
                    "meets_requirement": True,
                    "note": "Insufficient data"
                },
                "ats_compatibility": 0.0,
                "final_match_score": 0.0
            }

        # 1. Semantic TF-IDF Cosine Similarity
        semantic_sim = self._calculate_tfidf_similarity(cleaned_resume, cleaned_jd)

        # 2. Extract JD requirements
        jd_reqs = self._extract_jd_requirements(cleaned_jd)

        # 3. Skills Analysis
        resume_skills = set(s.lower() for s in extract_skills(cleaned_resume))
        jd_skills = set(s.lower() for s in extract_skills(cleaned_jd))

        missing_skills_lower = jd_skills - resume_skills
        missing_skills = [s.title() if len(s) > 3 else s.upper() for s in sorted(list(missing_skills_lower))]

        # Skill match percentage
        if jd_skills:
            skill_match_pct = round((len(jd_skills - missing_skills_lower) / len(jd_skills)) * 100.0, 1)
        else:
            skill_match_pct = 75.0

        # 4. Keyword Match Analysis
        keyword_match_pct, missing_keywords = self._analyze_keywords(cleaned_resume, cleaned_jd)

        # 5. Experience Gap Analysis
        candidate_years = extract_years_experience(cleaned_resume)
        required_years = jd_reqs["required_years"]
        gap_years = max(0.0, round(required_years - candidate_years, 1))
        meets_exp = candidate_years >= required_years

        exp_note = (
            f"Candidate meets or exceeds required {required_years} years ({candidate_years} yrs detected)."
            if meets_exp
            else f"Experience gap of {gap_years} years (Requires {required_years} yrs, candidate has {candidate_years} yrs)."
        )

        experience_gap = {
            "required_years": required_years,
            "candidate_years": candidate_years,
            "gap_years": gap_years,
            "meets_requirement": meets_exp,
            "note": exp_note
        }

        # 6. Education Gap Analysis
        education_gap = self._analyze_education_gap(cleaned_resume, jd_reqs["required_degree"])

        # 7. ATS Compatibility & Final Combined Match Score
        ats_compatibility = round((semantic_sim * 0.4) + (keyword_match_pct * 0.4) + (skill_match_pct * 0.2), 1)

        exp_penalty = 15.0 if not meets_exp and gap_years > 2 else (5.0 if not meets_exp else 0.0)
        edu_penalty = 10.0 if not education_gap["meets_requirement"] else 0.0

        raw_final = (ats_compatibility * 0.5) + (skill_match_pct * 0.3) + (keyword_match_pct * 0.2) - exp_penalty - edu_penalty
        final_match_score = round(max(0.0, min(100.0, raw_final)), 1)

        return {
            "keyword_match_pct": keyword_match_pct,
            "semantic_similarity_pct": round(semantic_sim, 1),
            "missing_keywords": missing_keywords[:15],  # top 15 missing terms
            "missing_skills": missing_skills,
            "experience_gap": experience_gap,
            "education_gap": education_gap,
            "ats_compatibility": ats_compatibility,
            "final_match_score": final_match_score
        }

    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate TF-IDF Cosine Similarity between resume and JD.
        """
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return round(float(cosine_sim) * 100.0, 1)
        except Exception:
            return 50.0

    def _extract_jd_requirements(self, jd_text: str) -> Dict[str, Any]:
        """
        Extract required years of experience and education level from job description text.
        """
        # Years of experience extraction regex
        exp_match = re.search(r'(\d+)\s*\+?\s*(?:-\s*\d+\s*)?years?(?:\s+of)?\s+experience', jd_text, re.I)
        if not exp_match:
            exp_match = re.search(r'minimum\s*(?:of\s*)?(\d+)\s*years?', jd_text, re.I)

        required_years = float(exp_match.group(1)) if exp_match else 0.0

        # Degree requirement regex
        required_degree = "Unspecified"
        if re.search(r'ph\.?d|doctorate', jd_text, re.I):
            required_degree = "Ph.D."
        elif re.search(r'master|\bm\.?s\.?\b|\bm\.?tech\b|\bmba\b', jd_text, re.I):
            required_degree = "Master's"
        elif re.search(r'bachelor|\bb\.?s\.?\b|\bb\.?tech\b|\bb\.?a\.?\b', jd_text, re.I):
            required_degree = "Bachelor's"
        elif re.search(r'associate', jd_text, re.I):
            required_degree = "Associate's"

        return {
            "required_years": required_years,
            "required_degree": required_degree
        }

    def _analyze_keywords(self, resume_text: str, jd_text: str) -> Tuple[float, List[str]]:
        """
        Extract high-frequency terms from JD and check presence in resume.
        """
        # Tokenize JD words
        jd_words = re.findall(r'\b[a-zA-Z]{3,}\b', jd_text.lower())
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "have", "you", "are",
            "will", "our", "all", "must", "about", "able", "work", "team", "role",
            "company", "looking", "join", "help", "candidate", "position", "apply"
        }
        filtered_jd_words = [w for w in jd_words if w not in stopwords]

        if not filtered_jd_words:
            return 80.0, []

        # Count frequencies in JD
        word_freq: Dict[str, int] = {}
        for w in filtered_jd_words:
            word_freq[w] = word_freq.get(w, 0) + 1

        # Pick top 25 candidate keywords by frequency
        top_jd_keywords = sorted(word_freq.keys(), key=lambda k: word_freq[k], reverse=True)[:25]

        resume_lower = resume_text.lower()
        matched = []
        missing = []

        for kw in top_jd_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', resume_lower):
                matched.append(kw)
            else:
                missing.append(kw)

        match_pct = round((len(matched) / len(top_jd_keywords)) * 100.0, 1)
        return match_pct, missing

    def _analyze_education_gap(self, resume_text: str, required_degree: str) -> Dict[str, Any]:
        """
        Evaluate candidate education degree against JD required degree.
        """
        candidate_entries = extract_education(resume_text)
        candidate_degree = candidate_entries[0]["degree"] if candidate_entries else "Unspecified"

        req_level = self.DEGREE_HIERARCHY.get(required_degree, 0)
        cand_level = self.DEGREE_HIERARCHY.get(candidate_degree, 0)

        meets_req = cand_level >= req_level or req_level == 0

        if meets_req:
            note = f"Candidate degree ({candidate_degree}) satisfies requirement ({required_degree})."
        else:
            note = f"Education gap: Required degree is {required_degree}, but candidate has {candidate_degree}."

        return {
            "required_degree": required_degree,
            "candidate_degree": candidate_degree,
            "meets_requirement": meets_req,
            "note": note
        }
