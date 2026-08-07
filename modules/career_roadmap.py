"""
AI Career Roadmap module for ResumeAI Pro.
Generates personalized step-by-step career roadmaps, portfolio project recommendations,
certifications, and online course suggestions.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try imports for AIEngine
try:
    from resumeai_pro.utils.ai_engine import AIEngine
except ImportError:
    try:
        from utils.ai_engine import AIEngine
    except ImportError:
        AIEngine = None


class CareerRoadmap:
    """Generates customized career roadmaps, project ideas, certifications, and course recommendations."""

    def __init__(self, ai_engine: Optional[Any] = None):
        """
        Initialize CareerRoadmap.

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

    def suggest_projects(self, target_role: str, current_skills: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Recommend tailored portfolio projects for a target role.

        Args:
            target_role: Target job title.
            current_skills: Optional list of candidate's current skills.

        Returns:
            List of project dictionaries.
        """
        role_lower = (target_role or "").lower()

        if "data scientist" in role_lower or "machine learning" in role_lower or "ai" in role_lower or "ml" in role_lower:
            return [
                {
                    "title": "Predictive Customer Churn & Lifetime Value Engine",
                    "description": "End-to-end ML pipeline with feature engineering, XGBoost modeling, model monitoring, and MLflow experiment tracking.",
                    "tech_stack": ["Python", "Scikit-Learn", "XGBoost", "MLflow", "Streamlit", "Docker"],
                    "difficulty": "Intermediate",
                    "key_features": ["Feature store", "SHAP model explainability", "Interactive dashboard"]
                },
                {
                    "title": "RAG-Powered AI Document Assistant",
                    "description": "Retrieval-Augmented Generation application indexing PDF documents with vector database and LLM response generation.",
                    "tech_stack": ["Python", "LangChain/LlamaIndex", "FAISS/Qdrant", "OpenAI/Ollama", "FastAPI"],
                    "difficulty": "Advanced",
                    "key_features": ["Semantic search", "Source citation", "Streaming API responses"]
                },
                {
                    "title": "Real-time Anomaly Detection on Streaming Data",
                    "description": "Streaming data pipeline ingesting financial transactions and flagging fraudulent transactions using Isolation Forests.",
                    "tech_stack": ["Python", "Kafka", "PySpark", "Docker", "PostgreSQL"],
                    "difficulty": "Advanced",
                    "key_features": ["Stream processing", "Alerting trigger", "Live dashboard"]
                }
            ]
        elif "devops" in role_lower or "cloud" in role_lower or "infrastructure" in role_lower:
            return [
                {
                    "title": "Automated Multi-Environment GitOps Kubernetes Deployment",
                    "description": "Infrastructure as Code (Terraform) provisioning AWS EKS cluster managed via ArgoCD GitOps pipeline.",
                    "tech_stack": ["Terraform", "AWS EKS", "ArgoCD", "Docker", "Helm", "GitHub Actions"],
                    "difficulty": "Advanced",
                    "key_features": ["Automated rollback", "Prometheus/Grafana monitoring", "Ingress TLS setup"]
                },
                {
                    "title": "CI/CD Pipeline Security & Compliance Automation",
                    "description": "DevSecOps pipeline incorporating static code analysis, vulnerability scanning, and container security compliance.",
                    "tech_stack": ["GitHub Actions", "SonarQube", "Trivy", "Docker", "Python"],
                    "difficulty": "Intermediate",
                    "key_features": ["Automated security blocking", "Slack notifications", "Coverage reports"]
                }
            ]
        else: # Software Engineer / Full Stack / Backend / Frontend default
            return [
                {
                    "title": "Scalable Collaborative Task & Workflow Management Platform",
                    "description": "Full-stack web application featuring real-time updates, RESTful APIs, role-based authorization, and automated testing.",
                    "tech_stack": ["React/TypeScript", "Node.js/Express", "PostgreSQL", "Redis", "Docker"],
                    "difficulty": "Intermediate",
                    "key_features": ["WebSocket real-time collaboration", "JWT authentication", "Redis caching"]
                },
                {
                    "title": "Distributed URL Shortener & Analytics Microservice",
                    "description": "High-throughput microservice system capable of shortening URLs with analytics tracking and geo-location metrics.",
                    "tech_stack": ["Python/FastAPI or Go", "PostgreSQL", "Redis", "Docker", "Prometheus"],
                    "difficulty": "Intermediate",
                    "key_features": ["Base62 encoding", "Rate limiting", "Geo-analytics dashboard"]
                },
                {
                    "title": "E-Commerce Microservices Platform with Distributed Transactions",
                    "description": "Event-driven microservice architecture with order management, payment processing, and inventory service.",
                    "tech_stack": ["Node.js/Python", "Kafka/RabbitMQ", "MongoDB", "Docker Compose"],
                    "difficulty": "Advanced",
                    "key_features": ["Saga pattern for transactions", "Event streaming", "Centralized logging"]
                }
            ]

    def suggest_certifications(self, target_role: str) -> List[Dict[str, str]]:
        """
        Recommend industry certifications for target role.

        Args:
            target_role: Target job title.

        Returns:
            List of certification dictionaries.
        """
        role_lower = (target_role or "").lower()

        if "cloud" in role_lower or "devops" in role_lower or "aws" in role_lower:
            return [
                {
                    "name": "AWS Certified Solutions Architect – Associate",
                    "provider": "Amazon Web Services",
                    "level": "Associate",
                    "url": "https://aws.amazon.com/certification/certified-solutions-architect-associate/"
                },
                {
                    "name": "Certified Kubernetes Administrator (CKA)",
                    "provider": "Cloud Native Computing Foundation (CNCF)",
                    "level": "Intermediate / Advanced",
                    "url": "https://www.cncf.io/certification/cka/"
                },
                {
                    "name": "HashiCorp Certified: Terraform Associate",
                    "provider": "HashiCorp",
                    "level": "Associate",
                    "url": "https://www.hashicorp.com/certification/terraform-associate"
                }
            ]
        elif "data" in role_lower or "machine learning" in role_lower or "ai" in role_lower:
            return [
                {
                    "name": "AWS Certified Machine Learning – Specialty",
                    "provider": "Amazon Web Services",
                    "level": "Specialty",
                    "url": "https://aws.amazon.com/certification/certified-machine-learning-specialty/"
                },
                {
                    "name": "TensorFlow Developer Certificate / Google Professional Data Engineer",
                    "provider": "Google Cloud",
                    "level": "Professional",
                    "url": "https://cloud.google.com/learn/certification/data-engineer"
                },
                {
                    "name": "Databricks Certified Machine Learning Professional",
                    "provider": "Databricks",
                    "level": "Professional",
                    "url": "https://www.databricks.com/learn/certification"
                }
            ]
        else:
            return [
                {
                    "name": "AWS Certified Developer – Associate",
                    "provider": "Amazon Web Services",
                    "level": "Associate",
                    "url": "https://aws.amazon.com/certification/certified-developer-associate/"
                },
                {
                    "name": "Oracle Certified Professional: Java SE Software Developer",
                    "provider": "Oracle",
                    "level": "Professional",
                    "url": "https://education.oracle.com/java-se-17-developer/pexam_1Z0-829"
                },
                {
                    "name": "Meta Front-End / Back-End Developer Professional Certificate",
                    "provider": "Meta / Coursera",
                    "level": "Professional",
                    "url": "https://www.coursera.org/professional-certificates/meta-back-end-developer"
                }
            ]

    def suggest_courses(self, target_role: str) -> List[Dict[str, str]]:
        """
        Recommend online courses for target role.

        Args:
            target_role: Target job title.

        Returns:
            List of course dictionaries.
        """
        role_lower = (target_role or "").lower()

        if "data scientist" in role_lower or "machine learning" in role_lower or "ai" in role_lower:
            return [
                {
                    "title": "Machine Learning Specialization",
                    "platform": "Coursera",
                    "instructor_or_org": "Andrew Ng (DeepLearning.AI)",
                    "level": "Beginner to Intermediate",
                    "url": "https://www.coursera.org/specializations/machine-learning-introduction"
                },
                {
                    "title": "Deep Learning Specialization",
                    "platform": "Coursera",
                    "instructor_or_org": "DeepLearning.AI",
                    "level": "Intermediate to Advanced",
                    "url": "https://www.coursera.org/specializations/deep-learning"
                }
            ]
        elif "devops" in role_lower or "cloud" in role_lower:
            return [
                {
                    "title": "Docker and Kubernetes: The Complete Guide",
                    "platform": "Udemy",
                    "instructor_or_org": "Stephen Grider",
                    "level": "Intermediate",
                    "url": "https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/"
                },
                {
                    "title": "DevOps Engineering on AWS",
                    "platform": "Coursera",
                    "instructor_or_org": "AWS",
                    "level": "Intermediate",
                    "url": "https://www.coursera.org/specializations/aws-devops"
                }
            ]
        else:
            return [
                {
                    "title": "Full Stack Web Development with React and Node",
                    "platform": "Coursera / edX",
                    "instructor_or_org": "HKUST / Meta",
                    "level": "Intermediate",
                    "url": "https://www.coursera.org/specializations/full-stack-react"
                },
                {
                    "title": "Grokking Modern System Design for Engineers",
                    "platform": "Educative.io",
                    "instructor_or_org": "Educative",
                    "level": "Intermediate to Advanced",
                    "url": "https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers"
                }
            ]

    def _generate_rule_based_roadmap(
        self,
        current_skills: List[str],
        target_role: str,
        timeline_weeks: int = 12
    ) -> List[Dict[str, Any]]:
        """Fallback rule-based weekly roadmap generator."""
        current_skills_str = ", ".join(current_skills) if current_skills else "general programming basics"
        target_role = target_role or "Software Engineer"
        weeks_count = max(4, min(timeline_weeks, 24))

        # Block allocation logic across weeks
        roadmap = []
        for w in range(1, weeks_count + 1):
            progress_pct = w / weeks_count

            if progress_pct <= 0.25:
                focus = "Foundations & Core Skill Gap Bridging"
                skills = ["Core Language Fundamentals", "Data Structures", "Version Control (Git)"]
                mini_proj = "Build a CLI Tool or small utility implementing data processing."
                milestone = "Master core concepts and set up development workflow."
            elif progress_pct <= 0.50:
                focus = f"Framework Mastery & API Development for {target_role}"
                skills = ["RESTful APIs / GraphQL", "Database Integration (SQL/NoSQL)", "Authentication & Security"]
                mini_proj = "Build a full REST API back-end with database persistence and user auth."
                milestone = "Successfully deploy a functional CRUD back-end / front-end web service."
            elif progress_pct <= 0.75:
                focus = "System Architecture, Caching & Performance Optimization"
                skills = ["System Design Patterns", "Redis / Caching", "Docker Containerization", "Unit & Integration Testing"]
                mini_proj = "Containerize service with Docker and add Redis caching for optimal performance."
                milestone = "Pass integration tests and run containerized application locally."
            else:
                focus = f"Capstone Portfolio Project & {target_role} Interview Prep"
                skills = ["Cloud Deployment (AWS/GCP)", "CI/CD Pipelines", "Mock Technical Interviews", "Resume Optimization"]
                mini_proj = f"Publish Capstone Project on GitHub with live URL demonstration for {target_role}."
                milestone = f"Complete production-ready portfolio and start applying for {target_role} positions."

            roadmap.append({
                "week": w,
                "focus_area": focus,
                "skills_to_learn": skills,
                "resources": [
                    {
                        "title": f"Week {w} Guide: {focus}",
                        "platform": "Documentation & Video Tutorials",
                        "url": "https://developer.mozilla.org" if "front" in target_role.lower() else "https://docs.python.org/3/"
                    }
                ],
                "mini_project": mini_proj,
                "milestone": milestone
            })

        return roadmap

    def generate(
        self,
        current_skills: List[str],
        target_role: str,
        timeline_weeks: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Generate a personalized weekly career roadmap towards a target role.

        Args:
            current_skills: List of candidate's existing skills.
            target_role: Target career position.
            timeline_weeks: Total timeline duration in weeks (default 12).

        Returns:
            List of weekly roadmap dictionaries containing week, focus_area, skills_to_learn, resources, mini_project, milestone.
        """
        target_role = target_role or "Software Engineer"
        timeline_weeks = max(4, min(timeline_weeks, 24))

        if self.ai_engine and hasattr(self.ai_engine, "is_available") and self.ai_engine.is_available():
            try:
                system_prompt = (
                    "You are a career advisor and engineering lead. "
                    "Generate a detailed, weekly career roadmap for a candidate transitioning to a target role. "
                    "Return strictly valid JSON object with a key 'weeks' containing a list of week objects."
                )
                prompt = f"""
Candidate Current Skills: {', '.join(current_skills if current_skills else ['Basic Coding'])}
Target Role: {target_role}
Timeline: {timeline_weeks} Weeks

Return a JSON object with key "weeks" containing exactly {timeline_weeks} items.
Each item in "weeks" MUST have keys:
- "week": int
- "focus_area": string
- "skills_to_learn": list of strings
- "resources": list of objects with "title", "platform", "url"
- "mini_project": string
- "milestone": string
"""
                json_res = self.ai_engine.generate_json(prompt, system_prompt=system_prompt)
                if isinstance(json_res, dict) and "weeks" in json_res and isinstance(json_res["weeks"], list):
                    weeks_list = json_res["weeks"]
                    if len(weeks_list) > 0:
                        formatted_roadmap = []
                        for item in weeks_list:
                            formatted_roadmap.append({
                                "week": item.get("week", len(formatted_roadmap) + 1),
                                "focus_area": item.get("focus_area", "Core Engineering"),
                                "skills_to_learn": item.get("skills_to_learn", []),
                                "resources": item.get("resources", []),
                                "mini_project": item.get("mini_project", "Build a small prototype."),
                                "milestone": item.get("milestone", "Complete weekly deliverables.")
                            })
                        return formatted_roadmap
            except Exception as e:
                logger.warning(f"AI roadmap generation failed, falling back to rule-based roadmap: {e}")

        # Fallback to rule-based roadmap
        return self._generate_rule_based_roadmap(current_skills, target_role, timeline_weeks)
