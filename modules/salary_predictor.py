"""
Salary Predictor module for ResumeAI Pro.
Predicts salary ranges using ML regression models trained on synthetic market data and rule-based databases.
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Scikit-learn import
try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SalaryPredictor:
    """Predicts salary ranges based on skills, experience, education, role, and location."""

    # Currency mappings per city/country
    INDIAN_CITIES = {
        "bangalore", "bengaluru", "mumbai", "delhi", "noida", "gurgaon", "gurugram",
        "hyderabad", "pune", "chennai", "kolkata", "ahmedabad", "india"
    }

    GLOBAL_CURRENCIES = {
        "london": "GBP",
        "uk": "GBP",
        "singapore": "SGD",
        "toronto": "CAD",
        "vancouver": "CAD",
        "canada": "CAD",
        "tokyo": "JPY",
        "japan": "JPY",
        "sydney": "AUD",
        "australia": "AUD",
        "berlin": "EUR",
        "germany": "EUR",
        "paris": "EUR",
        "dubai": "AED",
        "uae": "AED"
    }

    # Location Multipliers relative to national base
    LOCATION_MULTIPLIERS = {
        # Indian Cities
        "bangalore": 1.25,
        "bengaluru": 1.25,
        "mumbai": 1.20,
        "gurgaon": 1.20,
        "gurugram": 1.20,
        "noida": 1.15,
        "hyderabad": 1.15,
        "pune": 1.10,
        "chennai": 1.10,
        "delhi": 1.15,
        "kolkata": 0.95,
        "ahmedabad": 0.90,
        "india": 1.0,

        # Global Cities
        "san francisco": 1.50,
        "new york": 1.45,
        "seattle": 1.35,
        "austin": 1.20,
        "boston": 1.25,
        "london": 1.20,
        "singapore": 1.15,
        "toronto": 1.05,
        "berlin": 1.00,
        "sydney": 1.10,
        "dubai": 1.25,
        "tokyo": 1.05
    }

    # Base salary database for entry-level (0-2 YOE)
    # INR expressed in full annual amount (e.g. 900,000 INR = 9 LPA), USD in annual USD
    ROLE_BASE_SALARIES = {
        "software engineer": {"INR": 850000, "USD": 105000},
        "full stack developer": {"INR": 900000, "USD": 110000},
        "backend developer": {"INR": 900000, "USD": 110000},
        "frontend developer": {"INR": 800000, "USD": 100000},
        "data scientist": {"INR": 1000000, "USD": 125000},
        "ml engineer": {"INR": 1150000, "USD": 135000},
        "machine learning engineer": {"INR": 1150000, "USD": 135000},
        "ai engineer": {"INR": 1200000, "USD": 140000},
        "data engineer": {"INR": 950000, "USD": 120000},
        "devops engineer": {"INR": 950000, "USD": 120000},
        "cloud architect": {"INR": 1500000, "USD": 160000},
        "product manager": {"INR": 1300000, "USD": 140000},
        "cyber security analyst": {"INR": 850000, "USD": 110000},
        "qa engineer": {"INR": 650000, "USD": 85000},
        "mobile developer": {"INR": 850000, "USD": 105000},
        "solutions architect": {"INR": 1600000, "USD": 165000}
    }

    # High-demand skill premiums (percentage increase)
    SKILL_PREMIUMS = {
        "machine learning": 0.08,
        "deep learning": 0.10,
        "pytorch": 0.08,
        "tensorflow": 0.07,
        "kubernetes": 0.08,
        "system design": 0.10,
        "aws": 0.06,
        "gcp": 0.06,
        "distributed systems": 0.10,
        "generative ai": 0.12,
        "llm": 0.12,
        "blockchain": 0.08,
        "cybersecurity": 0.07,
        "rust": 0.09,
        "go": 0.07
    }

    def __init__(self):
        """Initialize SalaryPredictor and train synthetic ML model if scikit-learn is available."""
        self.ml_model_usd = None
        self.ml_model_inr = None
        if HAS_SKLEARN:
            try:
                self._train_ml_models()
            except Exception as e:
                logger.warning(f"Could not train ML model: {e}")

    def _train_ml_models(self):
        """Train RandomForestRegressor models on synthetic USD and INR market data."""
        np.random.seed(42)
        n_samples = 800

        yoe = np.random.uniform(0, 15, n_samples)
        edu = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
        loc_mult = np.random.choice([0.9, 1.0, 1.15, 1.25, 1.45], n_samples)
        num_skills = np.random.poisson(2, n_samples)

        exp_factor = 1.0 + (yoe * 0.12)
        edu_factor = 1.0 + (edu * 0.10)
        skills_factor = 1.0 + (num_skills * 0.05)

        # USD Model
        role_base_usd = np.random.choice([85000, 105000, 125000, 140000, 160000], n_samples)
        y_usd = role_base_usd * exp_factor * edu_factor * loc_mult * skills_factor
        X_usd = np.column_stack([yoe, edu, loc_mult, role_base_usd, num_skills])
        self.ml_model_usd = RandomForestRegressor(n_estimators=50, random_state=42)
        self.ml_model_usd.fit(X_usd, y_usd)

        # INR Model
        role_base_inr = np.random.choice([650000, 850000, 1000000, 1200000, 1500000], n_samples)
        y_inr = role_base_inr * exp_factor * edu_factor * loc_mult * skills_factor
        X_inr = np.column_stack([yoe, edu, loc_mult, role_base_inr, num_skills])
        self.ml_model_inr = RandomForestRegressor(n_estimators=50, random_state=42)
        self.ml_model_inr.fit(X_inr, y_inr)

    def _determine_currency(self, location: str) -> str:
        """Determine currency based on location name."""
        loc_clean = (location or "").lower().strip()
        if any(city in loc_clean for city in self.INDIAN_CITIES):
            return "INR"
        for city, curr in self.GLOBAL_CURRENCIES.items():
            if city in loc_clean:
                return curr
        return "USD"

    def _get_education_bonus(self, education: str) -> float:
        """Get bonus percentage multiplier based on degree level."""
        edu_clean = (education or "").lower()
        if "phd" in edu_clean or "doctorate" in edu_clean:
            return 0.20
        elif "master" in edu_clean or "m.tech" in edu_clean or "m.s" in edu_clean or "mba" in edu_clean:
            return 0.10
        elif "bachelor" in edu_clean or "b.tech" in edu_clean or "b.s" in edu_clean or "b.e" in edu_clean:
            return 0.0
        return 0.0

    def _get_experience_multiplier(self, yoe: float) -> float:
        """Calculate experience multiplier."""
        yoe = max(0.0, yoe)
        if yoe <= 2.0:
            return 1.0 + (yoe * 0.15)
        elif yoe <= 5.0:
            return 1.30 + ((yoe - 2.0) * 0.12)
        elif yoe <= 10.0:
            return 1.66 + ((yoe - 5.0) * 0.09)
        else:
            return 2.11 + ((yoe - 10.0) * 0.05)

    def predict(
        self,
        skills: List[str],
        experience_years: float,
        education: str,
        job_role: str,
        location: str
    ) -> Dict[str, Any]:
        """
        Predict expected salary range for given candidate parameters.

        Args:
            skills: List of technical and professional skills.
            experience_years: Years of relevant experience.
            education: Highest education level (e.g. "Bachelor's", "Master's", "PhD").
            job_role: Target job title (e.g. "Software Engineer", "Data Scientist").
            location: City or country location.

        Returns:
            Dictionary with min_salary, max_salary, median_salary, currency, confidence, factors.
        """
        currency = self._determine_currency(location)
        loc_clean = (location or "").lower().strip()
        role_clean = (job_role or "").lower().strip()

        # Find location multiplier
        loc_mult = 1.0
        for city, mult in self.LOCATION_MULTIPLIERS.items():
            if city in loc_clean:
                loc_mult = mult
                break

        # Find base role salary
        matched_role = "software engineer"
        for db_role in self.ROLE_BASE_SALARIES:
            if db_role in role_clean or role_clean in db_role:
                matched_role = db_role
                break

        base_sal = self.ROLE_BASE_SALARIES[matched_role].get(currency)
        if base_sal is None:
            base_sal = self.ROLE_BASE_SALARIES[matched_role]["INR"] if currency == "INR" else self.ROLE_BASE_SALARIES[matched_role]["USD"]

        # Experience Multiplier
        exp_mult = self._get_experience_multiplier(experience_years)

        # Education Bonus
        edu_bonus = self._get_education_bonus(education)

        # Skill Premiums
        skill_bonus = 0.0
        premium_count = 0
        skills_lower = [s.lower().strip() for s in (skills or [])]
        for p_skill, prem in self.SKILL_PREMIUMS.items():
            if any(p_skill in s for s in skills_lower):
                skill_bonus += prem
                premium_count += 1
        skill_bonus = min(0.35, skill_bonus)

        # Rule-based median calculation
        total_multiplier = exp_mult * (1.0 + edu_bonus + skill_bonus) * loc_mult
        rule_median = base_sal * total_multiplier

        final_median = rule_median
        confidence = 0.82

        # Combine with ML model prediction if available
        if HAS_SKLEARN:
            try:
                edu_score = 2 if edu_bonus >= 0.20 else (1 if edu_bonus >= 0.10 else 0)
                if currency == "INR" and self.ml_model_inr is not None:
                    X_sample = np.array([[experience_years, edu_score, loc_mult, base_sal, premium_count]])
                    ml_pred = self.ml_model_inr.predict(X_sample)[0]
                    final_median = (0.5 * rule_median) + (0.5 * ml_pred)
                    confidence = 0.88
                elif self.ml_model_usd is not None:
                    X_sample = np.array([[experience_years, edu_score, loc_mult, base_sal, premium_count]])
                    ml_pred = self.ml_model_usd.predict(X_sample)[0]
                    final_median = (0.5 * rule_median) + (0.5 * ml_pred)
                    confidence = 0.88
            except Exception as e:
                logger.warning(f"ML prediction blending failed: {e}")

        # Salary ranges: min (-15%), max (+18%)
        min_salary = round(final_median * 0.85)
        max_salary = round(final_median * 1.18)
        median_salary = round(final_median)

        lpa_formatted = None
        if currency == "INR":
            lpa_formatted = {
                "min_lpa": round(min_salary / 100000, 2),
                "max_lpa": round(max_salary / 100000, 2),
                "median_lpa": round(median_salary / 100000, 2)
            }

        return {
            "min_salary": min_salary,
            "max_salary": max_salary,
            "median_salary": median_salary,
            "currency": currency,
            "confidence": confidence,
            "matched_role": matched_role.title(),
            "lpa_breakdown": lpa_formatted,
            "factors": {
                "base_salary": round(base_sal),
                "experience_multiplier": round(exp_mult, 2),
                "location_multiplier": round(loc_mult, 2),
                "education_bonus_pct": round(edu_bonus * 100, 1),
                "skill_premium_pct": round(skill_bonus * 100, 1)
            }
        }
