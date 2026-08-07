"""Recruiter Dashboard Module for ResumeAI Pro.

Provides candidate ranking, filtering, side-by-side comparison, report generation,
and aggregate analytics for recruiters using TF-IDF matching and NLP analysis.
"""

from typing import Dict, Any, List, Optional, Union, Set
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecruiterDashboard:
    """Recruiter portal services for candidate evaluation and talent analytics."""

    COMMON_TECH_SKILLS = {
        "python", "java", "c++", "c#", "javascript", "typescript", "html", "css",
        "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring",
        "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "git", "ci/cd",
        "machine learning", "deep learning", "nlp", "pandas", "numpy", "scikit-learn",
        "tensorflow", "pytorch", "rest api", "graphql", "agile", "scrum"
    }

    def __init__(self) -> None:
        pass

    def rank_candidates(
        self, resumes: List[Dict[str, Any]], job_description: str
    ) -> pd.DataFrame:
        """Ranks multiple candidates against a job description.

        Args:
            resumes: List of candidate resume dictionaries.
            job_description: Job description text.

        Returns:
            pandas DataFrame sorted by match score descending.
        """
        if not resumes:
            return pd.DataFrame(columns=[
                "rank", "candidate_id", "name", "match_score",
                "skill_match_percentage", "matched_skills", "missing_skills",
                "experience_years", "education"
            ])

        if not job_description or not job_description.strip():
            results = []
            for idx, res in enumerate(resumes):
                info = self._extract_basic_info(res, idx)
                results.append({
                    "rank": idx + 1,
                    "candidate_id": info["candidate_id"],
                    "name": info["name"],
                    "match_score": 0.0,
                    "skill_match_percentage": 0.0,
                    "matched_skills": "",
                    "missing_skills": "",
                    "experience_years": info["experience_years"],
                    "education": info["education"]
                })
            return pd.DataFrame(results)

        # 1. TF-IDF Text Similarity
        jd_skills = self._extract_skills_from_text(job_description)
        candidate_texts = [self._extract_full_text(r) for r in resumes]
        corpus = [job_description] + candidate_texts

        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        except Exception:
            similarities = [0.0] * len(resumes)

        # 2. Skill Overlap & Metric Calculation
        ranked_list = []
        for idx, (res, tfidf_sim) in enumerate(zip(resumes, similarities)):
            info = self._extract_basic_info(res, idx)
            cand_skills = self._extract_candidate_skills(res)

            if jd_skills:
                matched = sorted(list(jd_skills.intersection(cand_skills)))
                missing = sorted(list(jd_skills.difference(cand_skills)))
                skill_ratio = len(matched) / len(jd_skills)
            else:
                matched = sorted(list(cand_skills))
                missing = []
                skill_ratio = float(tfidf_sim)

            # Combine TF-IDF similarity (50%) and Skill Overlap (50%)
            raw_score = (float(tfidf_sim) * 0.5) + (float(skill_ratio) * 0.5)
            final_score = round(min(100.0, max(0.0, raw_score * 100.0)), 1)
            skill_pct = round(min(100.0, max(0.0, float(skill_ratio) * 100.0)), 1)

            ranked_list.append({
                "candidate_id": info["candidate_id"],
                "name": info["name"],
                "match_score": final_score,
                "skill_match_percentage": skill_pct,
                "matched_skills": ", ".join(matched),
                "missing_skills": ", ".join(missing),
                "experience_years": info["experience_years"],
                "education": info["education"],
                "raw_data": res
            })

        # Sort by match score descending
        df = pd.DataFrame(ranked_list)
        df = df.sort_values(by="match_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        cols = [
            "rank", "candidate_id", "name", "match_score",
            "skill_match_percentage", "matched_skills", "missing_skills",
            "experience_years", "education"
        ]
        return df[cols]

    def filter_candidates(
        self, resumes: List[Dict[str, Any]], filters_dict: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filters candidates based on criteria dictionary.

        Supported filter keys:
            - min_experience (float)
            - max_experience (float)
            - required_skills (List[str] or str)
            - education_level (str or List[str])
            - min_match_score (float)
            - location (str)
            - query (str)

        Returns:
            List of resume dicts matching all filters.
        """
        if not resumes:
            return []

        filtered = []
        min_exp = filters_dict.get("min_experience")
        max_exp = filters_dict.get("max_experience")
        req_skills = filters_dict.get("required_skills", [])
        if isinstance(req_skills, str):
            req_skills = [s.strip().lower() for s in req_skills.split(",") if s.strip()]
        else:
            req_skills = [str(s).strip().lower() for s in req_skills]

        edu_levels = filters_dict.get("education_level", [])
        if isinstance(edu_levels, str):
            edu_levels = [edu_levels.lower()]
        else:
            edu_levels = [str(e).lower() for e in edu_levels]

        location_query = filters_dict.get("location", "").strip().lower()
        search_query = filters_dict.get("query", "").strip().lower()
        min_score = filters_dict.get("min_match_score")

        for idx, res in enumerate(resumes):
            info = self._extract_basic_info(res, idx)
            cand_skills = self._extract_candidate_skills(res)
            cand_text = self._extract_full_text(res).lower()

            # Filter 1: Min/Max Experience
            if min_exp is not None and info["experience_years"] < float(min_exp):
                continue
            if max_exp is not None and info["experience_years"] > float(max_exp):
                continue

            # Filter 2: Required Skills
            if req_skills:
                has_all_skills = all(s in cand_skills or s in cand_text for s in req_skills)
                if not has_all_skills:
                    continue

            # Filter 3: Education
            if edu_levels:
                cand_edu = info["education"].lower()
                has_edu = any(el in cand_edu for el in edu_levels)
                if not has_edu:
                    continue

            # Filter 4: Location
            if location_query:
                cand_loc = info.get("location", "").lower()
                if location_query not in cand_loc and location_query not in cand_text:
                    continue

            # Filter 5: Search Query
            if search_query:
                if search_query not in cand_text and search_query not in info["name"].lower():
                    continue

            # Filter 6: Min Match Score (if present)
            if min_score is not None:
                score = res.get("match_score", res.get("score", 0.0))
                if float(score) < float(min_score):
                    continue

            filtered.append(res)

        return filtered

    def compare_applicants(
        self, resume1: Dict[str, Any], resume2: Dict[str, Any], job_description: str = ""
    ) -> Dict[str, Any]:
        """Performs side-by-side comparison of two candidates.

        Returns:
            Dict containing comparison matrix, individual metrics, and recommendation.
        """
        info1 = self._extract_basic_info(resume1, 0)
        info2 = self._extract_basic_info(resume2, 1)

        skills1 = self._extract_candidate_skills(resume1)
        skills2 = self._extract_candidate_skills(resume2)

        if job_description:
            rank_df = self.rank_candidates([resume1, resume2], job_description)
            row1 = rank_df[rank_df["candidate_id"] == info1["candidate_id"]].iloc[0]
            row2 = rank_df[rank_df["candidate_id"] == info2["candidate_id"]].iloc[0]
            score1 = row1["match_score"]
            score2 = row2["match_score"]
            matched1 = row1["matched_skills"]
            matched2 = row2["matched_skills"]
            missing1 = row1["missing_skills"]
            missing2 = row2["missing_skills"]
        else:
            score1, score2 = 0.0, 0.0
            matched1, matched2 = ", ".join(sorted(list(skills1))), ", ".join(sorted(list(skills2)))
            missing1, missing2 = "N/A", "N/A"

        unique_to_1 = sorted(list(skills1 - skills2))
        unique_to_2 = sorted(list(skills2 - skills1))
        common_skills = sorted(list(skills1.intersection(skills2)))

        comparison_data = {
            "Metric": [
                "Candidate Name",
                "Job Match Score (%)",
                "Years of Experience",
                "Education Level",
                "Total Skills Count",
                "Matched Skills",
                "Missing Skills",
            ],
            f"Candidate 1: {info1['name']}": [
                info1["name"],
                f"{score1}%" if job_description else "N/A",
                f"{info1['experience_years']} yrs",
                info1["education"],
                len(skills1),
                matched1 or "None",
                missing1 or "None",
            ],
            f"Candidate 2: {info2['name']}": [
                info2["name"],
                f"{score2}%" if job_description else "N/A",
                f"{info2['experience_years']} yrs",
                info2["education"],
                len(skills2),
                matched2 or "None",
                missing2 or "None",
            ],
        }

        comparison_df = pd.DataFrame(comparison_data)

        if job_description:
            if score1 > score2:
                winner = info1["name"]
                reason = f"{info1['name']} scored higher ({score1}% vs {score2}%) with stronger skill alignment."
            elif score2 > score1:
                winner = info2["name"]
                reason = f"{info2['name']} scored higher ({score2}% vs {score1}%) with stronger skill alignment."
            else:
                winner = "Tie"
                reason = "Both candidates scored identically against the job description."
        else:
            if info1["experience_years"] > info2["experience_years"]:
                winner = info1["name"]
                reason = f"{info1['name']} has more total experience ({info1['experience_years']} yrs vs {info2['experience_years']} yrs)."
            elif info2["experience_years"] > info1["experience_years"]:
                winner = info2["name"]
                reason = f"{info2['name']} has more total experience ({info2['experience_years']} yrs vs {info1['experience_years']} yrs)."
            else:
                winner = "Tie"
                reason = "Both candidates have comparable background experience."

        return {
            "candidate1": {
                "info": info1,
                "skills": sorted(list(skills1)),
                "unique_skills": unique_to_1,
                "score": score1,
            },
            "candidate2": {
                "info": info2,
                "skills": sorted(list(skills2)),
                "unique_skills": unique_to_2,
                "score": score2,
            },
            "comparison_matrix": comparison_df,
            "common_skills": common_skills,
            "recommendation": {
                "recommended_candidate": winner,
                "reasoning": reason,
            },
        }

    def generate_report(
        self, candidate_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]
    ) -> Dict[str, Any]:
        """Generates a comprehensive downloadable report for candidate evaluation.

        Args:
            candidate_data: Single resume dict, list of resume dicts, or ranked DataFrame.

        Returns:
            Dict containing report_text, executive_summary, top_candidates, and csv_export.
        """
        if isinstance(candidate_data, pd.DataFrame):
            df = candidate_data.copy()
        elif isinstance(candidate_data, list):
            df = pd.DataFrame(candidate_data)
        elif isinstance(candidate_data, dict):
            df = pd.DataFrame([candidate_data])
        else:
            df = pd.DataFrame()

        total_candidates = len(df)
        if total_candidates == 0:
            return {
                "report_text": "No candidate data available.",
                "executive_summary": "Empty candidate pool.",
                "top_candidates": pd.DataFrame(),
                "skill_gap_analysis": [],
                "csv_export": "",
            }

        csv_export = df.to_csv(index=False)

        top_score = df["match_score"].max() if "match_score" in df.columns else 0.0
        avg_score = round(float(df["match_score"].mean()), 1) if "match_score" in df.columns else 0.0

        exec_summary = (
            f"Evaluated {total_candidates} candidate(s). "
            f"Top match score: {top_score}%, Average match score: {avg_score}%."
        )

        lines = [
            "==================================================",
            "RECRUITER EVALUATION REPORT - RESUMEAI PRO",
            "==================================================",
            f"Total Candidates Evaluated: {total_candidates}",
            f"Executive Summary: {exec_summary}",
            "--------------------------------------------------",
            "\nCANDIDATE RANKINGS SUMMARY:",
        ]

        if "rank" in df.columns and "name" in df.columns:
            for _, row in df.iterrows():
                score_str = f"{row.get('match_score', 0)}%" if "match_score" in row else "N/A"
                exp_str = f"{row.get('experience_years', 0)} yrs" if "experience_years" in row else "N/A"
                lines.append(
                    f"#{row.get('rank', '-')}: {row.get('name', 'Candidate')} | Score: {score_str} | Exp: {exp_str}"
                )

        missing_skills_list = []
        if "missing_skills" in df.columns:
            for ms in df["missing_skills"].dropna():
                if isinstance(ms, str) and ms.strip():
                    missing_skills_list.extend([s.strip() for s in ms.split(",") if s.strip()])

        missing_counts = pd.Series(missing_skills_list).value_counts().to_dict() if missing_skills_list else {}
        top_gaps = [f"{skill} (missing in {cnt} candidates)" for skill, cnt in list(missing_counts.items())[:5]]

        lines.append("\nTOP SKILL GAPS IDENTIFIED IN POOL:")
        if top_gaps:
            for gap in top_gaps:
                lines.append(f" - {gap}")
        else:
            lines.append(" - No major skill gaps detected.")

        lines.append("\n==================================================")

        return {
            "report_text": "\n".join(lines),
            "executive_summary": exec_summary,
            "top_candidates": df.head(10),
            "skill_gap_analysis": top_gaps,
            "csv_export": csv_export,
        }

    def analytics_summary(self, resumes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Computes aggregate analytics statistics across candidate resumes.

        Returns:
            Dict containing metric summaries and DataFrames for Streamlit charts.
        """
        if not resumes:
            return {
                "total_candidates": 0,
                "avg_experience_years": 0.0,
                "top_skills_df": pd.DataFrame(columns=["skill", "count"]),
                "education_distribution_df": pd.DataFrame(columns=["education", "count"]),
                "experience_level_counts": {},
                "summary_metrics": {},
            }

        exp_years_list = []
        all_skills_list = []
        education_list = []
        locations_list = []

        for idx, res in enumerate(resumes):
            info = self._extract_basic_info(res, idx)
            cand_skills = self._extract_candidate_skills(res)

            exp_years_list.append(info["experience_years"])
            all_skills_list.extend(list(cand_skills))
            education_list.append(info["education"])
            if info.get("location"):
                locations_list.append(info["location"])

        total = len(resumes)
        avg_exp = round(float(np.mean(exp_years_list)), 1) if exp_years_list else 0.0
        max_exp = round(float(np.max(exp_years_list)), 1) if exp_years_list else 0.0
        min_exp = round(float(np.min(exp_years_list)), 1) if exp_years_list else 0.0

        skills_series = pd.Series(all_skills_list).value_counts()
        top_skills_df = skills_series.reset_index()
        top_skills_df.columns = ["skill", "count"]

        edu_series = pd.Series(education_list).value_counts()
        edu_df = edu_series.reset_index()
        edu_df.columns = ["education", "count"]

        entry_cnt = sum(1 for e in exp_years_list if e <= 2)
        mid_cnt = sum(1 for e in exp_years_list if 2 < e <= 5)
        senior_cnt = sum(1 for e in exp_years_list if e > 5)

        exp_brackets = {
            "Entry Level (0-2 yrs)": entry_cnt,
            "Mid Level (3-5 yrs)": mid_cnt,
            "Senior Level (5+ yrs)": senior_cnt,
        }

        return {
            "total_candidates": total,
            "avg_experience_years": avg_exp,
            "max_experience_years": max_exp,
            "min_experience_years": min_exp,
            "top_skills_df": top_skills_df.head(15),
            "education_distribution_df": edu_df,
            "experience_level_counts": exp_brackets,
            "summary_metrics": {
                "total_candidates": total,
                "avg_experience": avg_exp,
                "unique_skills_count": len(skills_series),
                "most_common_skill": top_skills_df.iloc[0]["skill"] if not top_skills_df.empty else "N/A",
            },
        }

    # -------------------------------------------------------------------------
    # Internal Utility Extraction Helpers
    # -------------------------------------------------------------------------

    def _extract_basic_info(self, res: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        """Extracts basic candidate info fields safely."""
        personal = res.get("personal_info", res.get("personal", {}))
        if not isinstance(personal, dict):
            personal = {}

        candidate_id = str(res.get("id", res.get("candidate_id", f"CAND-{index+1:03d}")))
        name = personal.get("name", res.get("name", f"Candidate {index+1}"))
        location = personal.get("location", res.get("location", ""))

        exp_years = res.get("experience_years", res.get("total_experience", None))
        if exp_years is None:
            exp_years = self._calculate_experience_years(res)

        education = self._extract_highest_education(res)

        return {
            "candidate_id": candidate_id,
            "name": name,
            "location": location,
            "experience_years": float(exp_years),
            "education": education,
        }

    def _calculate_experience_years(self, res: Dict[str, Any]) -> float:
        """Estimates experience years from resume work experience items."""
        exp_list = res.get("experience", res.get("work_experience", []))
        if not isinstance(exp_list, list) or not exp_list:
            return 1.0

        total_years = 0.0
        for item in exp_list:
            if isinstance(item, dict):
                dates = str(item.get("dates", ""))
                start = str(item.get("start_date", ""))
                end = str(item.get("end_date", ""))
                combined = f"{dates} {start} {end}"

                years = re.findall(r"\b(19\d\d|20\d\d)\b", combined)
                if len(years) >= 2:
                    y1, y2 = int(years[0]), int(years[1])
                    total_years += max(1.0, float(abs(y2 - y1)))
                elif len(years) == 1:
                    total_years += 2.0
                else:
                    total_years += 1.5
        return round(max(0.5, total_years), 1)

    def _extract_highest_education(self, res: Dict[str, Any]) -> str:
        """Extracts highest degree level achieved."""
        edu_list = res.get("education", [])
        if not isinstance(edu_list, list) or not edu_list:
            return "Bachelor's Degree"

        degrees = []
        for item in edu_list:
            if isinstance(item, dict):
                degree = str(item.get("degree", ""))
                degrees.append(degree)
            elif isinstance(item, str):
                degrees.append(item)

        all_degrees_str = " ".join(degrees).lower()
        if "phd" in all_degrees_str or "doctor" in all_degrees_str:
            return "Ph.D."
        if "master" in all_degrees_str or "ms" in all_degrees_str or "m.s." in all_degrees_str or "mba" in all_degrees_str:
            return "Master's Degree"
        if "bachelor" in all_degrees_str or "bs" in all_degrees_str or "b.s." in all_degrees_str or "btech" in all_degrees_str:
            return "Bachelor's Degree"
        if "associate" in all_degrees_str:
            return "Associate Degree"

        return degrees[0] if degrees else "Bachelor's Degree"

    def _extract_candidate_skills(self, res: Dict[str, Any]) -> Set[str]:
        """Extracts candidate skills as a normalized set of lowercase strings."""
        skills_set = set()
        raw_skills = res.get("skills", [])

        if isinstance(raw_skills, list):
            for s in raw_skills:
                skills_set.add(str(s).strip().lower())
        elif isinstance(raw_skills, dict):
            for cat, items in raw_skills.items():
                if isinstance(items, list):
                    for i in items:
                        skills_set.add(str(i).strip().lower())
                elif isinstance(items, str):
                    for i in items.split(","):
                        skills_set.add(i.strip().lower())

        full_text = self._extract_full_text(res).lower()
        for tech in self.COMMON_TECH_SKILLS:
            if re.search(r"\b" + re.escape(tech) + r"\b", full_text):
                skills_set.add(tech)

        return {s for s in skills_set if s}

    def _extract_skills_from_text(self, text: str) -> Set[str]:
        """Extracts technical skills present in arbitrary text."""
        found = set()
        text_lower = text.lower()
        for tech in self.COMMON_TECH_SKILLS:
            if re.search(r"\b" + re.escape(tech) + r"\b", text_lower):
                found.add(tech)

        words = re.findall(r"\b[a-zA-Z\+\#]{2,20}\b", text_lower)
        for w in words:
            if w in self.COMMON_TECH_SKILLS:
                found.add(w)

        return found

    def _extract_full_text(self, res: Dict[str, Any]) -> str:
        """Combines all resume fields into a single text blob."""
        parts = []
        personal = res.get("personal_info", res.get("personal", {}))
        if isinstance(personal, dict):
            parts.extend([
                str(personal.get("name", "")),
                str(personal.get("title", "")),
                str(personal.get("summary", "")),
            ])

        parts.append(str(res.get("summary", "")))

        exp_list = res.get("experience", res.get("work_experience", []))
        if isinstance(exp_list, list):
            for exp in exp_list:
                if isinstance(exp, dict):
                    parts.extend([
                        str(exp.get("title", "")),
                        str(exp.get("company", "")),
                        str(exp.get("description", "")),
                    ])
                    resp = exp.get("responsibilities", [])
                    if isinstance(resp, list):
                        parts.extend([str(r) for r in resp])

        raw_skills = res.get("skills", [])
        if isinstance(raw_skills, list):
            parts.extend([str(s) for s in raw_skills])
        elif isinstance(raw_skills, dict):
            for k, v in raw_skills.items():
                parts.append(str(k))
                if isinstance(v, list):
                    parts.extend([str(x) for x in v])

        for section in ["education", "projects"]:
            items = res.get(section, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        parts.extend([str(val) for val in item.values()])

        return " ".join([p for p in parts if p])
