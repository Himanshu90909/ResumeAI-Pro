"""
Skill Gap Analysis module for ResumeAI Pro.
Compares candidate skills against job requirements and recommends curated learning resources.
"""

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SkillGapAnalyzer:
    """Analyzes skill gaps between candidate resume and job requirements."""

    # Alias mapping for skill normalization
    ALIAS_MAP = {
        "py": "python",
        "python3": "python",
        "js": "javascript",
        "ts": "typescript",
        "react.js": "react",
        "reactjs": "react",
        "node.js": "node.js",
        "nodejs": "node.js",
        "express.js": "express.js",
        "expressjs": "express.js",
        "vue.js": "vue",
        "vuejs": "vue",
        "angularjs": "angular",
        "postgres": "postgresql",
        "mongo": "mongodb",
        "k8s": "kubernetes",
        "ml": "machine learning",
        "dl": "deep learning",
        "ai": "artificial intelligence",
        "aws": "amazon web services",
        "gcp": "google cloud platform",
        "tf": "tensorflow",
        "rest": "rest api",
        "restful": "rest api",
        "rest api": "rest api",
        "ci/cd": "ci/cd",
        "cicd": "ci/cd"
    }

    def __init__(self):
        """Initialize SkillGapAnalyzer with pre-built learning resources database."""
        self.resource_db = self._init_resource_database()

    def _init_resource_database(self) -> Dict[str, List[Dict[str, str]]]:
        """Hardcoded database of learning resources (Coursera, YouTube, Official Docs, etc.)."""
        return {
            "python": [
                {
                    "title": "Python for Everybody Specialization",
                    "type": "Course",
                    "platform": "Coursera",
                    "url": "https://www.coursera.org/specializations/python",
                    "estimated_time": "4 weeks",
                    "level": "Beginner to Intermediate"
                },
                {
                    "title": "Core Python Tutorials & Best Practices",
                    "type": "Video",
                    "platform": "YouTube (Corey Schafer)",
                    "url": "https://www.youtube.com/user/schafer5",
                    "estimated_time": "15 hours",
                    "level": "Intermediate"
                },
                {
                    "title": "Official Python Documentation & Tutorials",
                    "type": "Documentation",
                    "platform": "Official Docs",
                    "url": "https://docs.python.org/3/tutorial/",
                    "estimated_time": "10 hours",
                    "level": "All Levels"
                }
            ],
            "react": [
                {
                    "title": "React - The Complete Guide (incl. Hooks, React Router, Redux)",
                    "type": "Course",
                    "platform": "Udemy",
                    "url": "https://www.udemy.com/course/react-the-complete-guide-incl-redux/",
                    "estimated_time": "6 weeks",
                    "level": "Intermediate"
                },
                {
                    "title": "Official React Documentation (react.dev)",
                    "type": "Documentation",
                    "platform": "Official Docs",
                    "url": "https://react.dev/learn",
                    "estimated_time": "12 hours",
                    "level": "Beginner to Advanced"
                },
                {
                    "title": "React Full Course 2026",
                    "type": "Video",
                    "platform": "YouTube (freeCodeCamp)",
                    "url": "https://www.youtube.com/watch?v=bMknfKXIFA8",
                    "estimated_time": "12 hours",
                    "level": "Beginner"
                }
            ],
            "node.js": [
                {
                    "title": "Node.js Developer Course",
                    "type": "Course",
                    "platform": "Coursera",
                    "url": "https://www.coursera.org/learn/server-side-nodejs",
                    "estimated_time": "4 weeks",
                    "level": "Intermediate"
                },
                {
                    "title": "Node.js Official Documentation & Guides",
                    "type": "Documentation",
                    "platform": "Official Docs",
                    "url": "https://nodejs.org/en/docs/guides/",
                    "estimated_time": "8 hours",
                    "level": "Intermediate"
                }
            ],
            "docker": [
                {
                    "title": "Docker for Beginners & DevOps Engineers",
                    "type": "Course",
                    "platform": "Coursera / Udemy",
                    "url": "https://www.udemy.com/course/docker-easy/",
                    "estimated_time": "3 weeks",
                    "level": "Beginner to Intermediate"
                },
                {
                    "title": "Docker Official Orientation & Docs",
                    "type": "Documentation",
                    "platform": "Official Docs",
                    "url": "https://docs.docker.com/get-started/",
                    "estimated_time": "6 hours",
                    "level": "Beginner"
                },
                {
                    "title": "Docker Crash Course for Beginners",
                    "type": "Video",
                    "platform": "YouTube (TechWorld with Nana)",
                    "url": "https://www.youtube.com/watch?v=3c-iBn73dDE",
                    "estimated_time": "3 hours",
                    "level": "Beginner"
                }
            ],
            "kubernetes": [
                {
                    "title": "Architecting with Google Kubernetes Engine",
                    "type": "Course",
                    "platform": "Coursera",
                    "url": "https://www.coursera.org/specializations/google-kubernetes-engine",
                    "estimated_time": "5 weeks",
                    "level": "Advanced"
                },
                {
                    "title": "Kubernetes Official Tutorials & Documentation",
                    "type": "Documentation",
                    "platform": "Official Docs",
                    "url": "https://kubernetes.io/docs/tutorials/",
                    "estimated_time": "15 hours",
                    "level": "Intermediate to Advanced"
                }
            ],
            "amazon web services": [
                {
                    "title": "AWS Certified Solutions Architect Associate",
                    "type": "Course",
                    "platform": "Coursera / Stephane Maarek",
                    "url": "https://www.coursera.org/professional-certificates/aws-cloud-architecture",
                    "estimated_time": "8 weeks",
                    "level": "Intermediate"
                },
                {
                    "title": "AWS Fundamentals & Hands-on Labs",
                    "type": "Documentation",
                    "platform": "AWS Documentation",
                    "url": "https://aws.amazon.com/getting-started/",
                    "estimated_time": "20 hours",
                    "level": "All Levels"
                }
            ],
            "machine learning": [
                {
                    "title": "Machine Learning Specialization by Andrew Ng",
                    "type": "Course",
                    "platform": "Coursera (DeepLearning.AI)",
                    "url": "https://www.coursera.org/specializations/machine-learning-introduction",
                    "estimated_time": "8 weeks",
                    "level": "Beginner to Intermediate"
                },
                {
                    "title": "Machine Learning Course for Beginners",
                    "type": "Video",
                    "platform": "YouTube (freeCodeCamp)",
                    "url": "https://www.youtube.com/watch?v=i_LwzRVP7bg",
                    "estimated_time": "10 hours",
                    "level": "Beginner"
                },
                {
                    "title": "Scikit-Learn Official User Guide",
                    "type": "Documentation",
                    "platform": "Official Docs",
                    "url": "https://scikit-learn.org/stable/user_guide.html",
                    "estimated_time": "12 hours",
                    "level": "Intermediate"
                }
            ],
            "deep learning": [
                {
                    "title": "Deep Learning Specialization",
                    "type": "Course",
                    "platform": "Coursera",
                    "url": "https://www.coursera.org/specializations/deep-learning",
                    "estimated_time": "12 weeks",
                    "level": "Advanced"
                },
                {
                    "title": "PyTorch Official Tutorials",
                    "type": "Documentation",
                    "platform": "Official Docs",
                    "url": "https://pytorch.org/tutorials/",
                    "estimated_time": "15 hours",
                    "level": "Intermediate"
                }
            ],
            "sql": [
                {
                    "title": "SQL for Data Science",
                    "type": "Course",
                    "platform": "Coursera",
                    "url": "https://www.coursera.org/learn/sql-for-data-science",
                    "estimated_time": "4 weeks",
                    "level": "Beginner"
                },
                {
                    "title": "SQL Tutorial - Full Database Course for Beginners",
                    "type": "Video",
                    "platform": "YouTube (freeCodeCamp)",
                    "url": "https://www.youtube.com/watch?v=HXV3zeQKqGY",
                    "estimated_time": "4 hours",
                    "level": "Beginner"
                },
                {
                    "title": "PostgreSQL Official Documentation",
                    "type": "Documentation",
                    "platform": "Official Docs",
                    "url": "https://www.postgresql.org/docs/",
                    "estimated_time": "10 hours",
                    "level": "All Levels"
                }
            ],
            "postgresql": [
                {
                    "title": "PostgreSQL Tutorial & High Performance Query Optimization",
                    "type": "Documentation",
                    "platform": "Official Docs",
                    "url": "https://www.postgresqltutorial.com/",
                    "estimated_time": "10 hours",
                    "level": "Intermediate"
                }
            ],
            "system design": [
                {
                    "title": "Grokking the System Design Interview",
                    "type": "Course",
                    "platform": "Educative.io",
                    "url": "https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers",
                    "estimated_time": "4 weeks",
                    "level": "Intermediate to Advanced"
                },
                {
                    "title": "System Design Primer Repository",
                    "type": "Documentation",
                    "platform": "GitHub (donnemartin)",
                    "url": "https://github.com/donnemartin/system-design-primer",
                    "estimated_time": "20 hours",
                    "level": "Intermediate to Advanced"
                },
                {
                    "title": "System Design Course for Beginners",
                    "type": "Video",
                    "platform": "YouTube (ByteByteGo)",
                    "url": "https://www.youtube.com/@ByteByteGo",
                    "estimated_time": "10 hours",
                    "level": "Intermediate"
                }
            ],
            "git": [
                {
                    "title": "Git & GitHub Course for Beginners",
                    "type": "Video",
                    "platform": "YouTube (freeCodeCamp)",
                    "url": "https://www.youtube.com/watch?v=RGOj5yH7evk",
                    "estimated_time": "2 hours",
                    "level": "Beginner"
                },
                {
                    "title": "Pro Git Book & Official Documentation",
                    "type": "Documentation",
                    "platform": "Official Docs",
                    "url": "https://git-scm.com/book/en/v2",
                    "estimated_time": "8 hours",
                    "level": "Intermediate"
                }
            ],
            "java": [
                {
                    "title": "Java Programming and Software Engineering Fundamentals",
                    "type": "Course",
                    "platform": "Coursera",
                    "url": "https://www.coursera.org/specializations/java-programming",
                    "estimated_time": "6 weeks",
                    "level": "Beginner to Intermediate"
                }
            ],
            "spring boot": [
                {
                    "title": "Spring Boot Tutorial & Microservices Architecture",
                    "type": "Video",
                    "platform": "YouTube (Amigoscode)",
                    "url": "https://www.youtube.com/watch?v=9SGDpanrc8U",
                    "estimated_time": "6 hours",
                    "level": "Intermediate"
                }
            ]
        }

    def _normalize(self, skill: str) -> str:
        """Normalize a skill string for accurate comparison."""
        if not skill:
            return ""
        s = skill.strip().lower()
        # Clean extra punctuation
        s = re.sub(r'[\(\)\[\]\{\}]', '', s)
        s = s.strip()
        return self.ALIAS_MAP.get(s, s)

    def analyze(self, resume_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
        """
        Analyze skill match between resume skills and required job skills.

        Args:
            resume_skills: List of skills extracted from candidate resume.
            job_skills: List of skills required by job posting.

        Returns:
            Dictionary with matched_skills, missing_skills, extra_skills, match_percentage, and priority.
        """
        resume_skills = resume_skills or []
        job_skills = job_skills or []

        # Map normalized to original display names
        resume_norm_map = {self._normalize(s): s.strip() for s in resume_skills if s and s.strip()}
        job_norm_map = {self._normalize(s): s.strip() for s in job_skills if s and s.strip()}

        resume_norm_set = set(resume_norm_map.keys())
        job_norm_set = set(job_norm_map.keys())

        matched_norm = job_norm_set.intersection(resume_norm_set)
        missing_norm = job_norm_set.difference(resume_norm_set)
        extra_norm = resume_norm_set.difference(job_norm_set)

        matched_skills = [job_norm_map[n] for n in matched_norm]
        missing_skills = [job_norm_map[n] for n in missing_norm]
        extra_skills = [resume_norm_map[n] for n in extra_norm]

        # Priority ordering for missing skills based on position in job requirements & resource popularity
        priority_missing = sorted(
            missing_skills,
            key=lambda s: (
                0 if self._normalize(s) in self.resource_db else 1,
                job_skills.index(s) if s in job_skills else 99
            )
        )

        total_req = len(job_norm_set)
        if total_req == 0:
            match_percentage = 100.0
        else:
            match_percentage = round((len(matched_norm) / total_req) * 100.0, 2)

        return {
            "matched_skills": matched_skills,
            "missing_skills": priority_missing,
            "extra_skills": extra_skills,
            "match_percentage": match_percentage,
            "total_job_skills_count": total_req,
            "matched_count": len(matched_skills),
            "missing_count": len(missing_skills)
        }

    def recommend_resources(self, missing_skills: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """
        Recommend curated learning resources for each missing skill.

        Args:
            missing_skills: List of skill names that are missing.

        Returns:
            Dictionary mapping skill name to a list of resource dicts.
        """
        recommendations = {}
        missing_skills = missing_skills or []

        for skill in missing_skills:
            norm_skill = self._normalize(skill)
            
            # Check exact or partial match in resource database
            if norm_skill in self.resource_db:
                recommendations[skill] = self.resource_db[norm_skill]
            else:
                # Search partial matches in db
                matched_key = None
                for db_key in self.resource_db:
                    if db_key in norm_skill or norm_skill in db_key:
                        matched_key = db_key
                        break

                if matched_key:
                    recommendations[skill] = self.resource_db[matched_key]
                else:
                    # Dynamic generic resources
                    clean_query = skill.strip().replace(" ", "+")
                    recommendations[skill] = [
                        {
                            "title": f"Coursera Courses: {skill}",
                            "type": "Course",
                            "platform": "Coursera",
                            "url": f"https://www.coursera.org/search?query={clean_query}",
                            "estimated_time": "4 weeks",
                            "level": "All Levels"
                        },
                        {
                            "title": f"{skill} Tutorial for Beginners",
                            "type": "Video",
                            "platform": "YouTube",
                            "url": f"https://www.youtube.com/results?search_query={clean_query}+tutorial",
                            "estimated_time": "5 hours",
                            "level": "Beginner"
                        },
                        {
                            "title": f"{skill} Official Documentation / Reference",
                            "type": "Documentation",
                            "platform": "Official Docs",
                            "url": f"https://www.google.com/search?q={clean_query}+official+documentation",
                            "estimated_time": "10 hours",
                            "level": "All Levels"
                        }
                    ]

        return recommendations
