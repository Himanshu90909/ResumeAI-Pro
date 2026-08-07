"""
AI Interview Generator module for ResumeAI Pro.
Generates tailored interview questions across multiple categories and difficulty levels.
"""

import logging
import random
import re
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


class InterviewGenerator:
    """Generates interview questions, suggested answers, and evaluation criteria."""

    def __init__(self, ai_engine: Optional[Any] = None):
        """
        Initialize the InterviewGenerator.

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

        self._question_bank = self._init_question_bank()

    def _init_question_bank(self) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        """Initialize curated bank of interview questions categorized by type and difficulty."""
        return {
            "hr": {
                "easy": [
                    {
                        "question": "Tell me about yourself and your professional background.",
                        "suggested_answer": "Provide an impactful 2-minute summary covering your background, core technical strengths, key achievements, and why you are passionate about this role.",
                        "evaluation_criteria": "Clarity of expression, structured delivery, relevance to role, enthusiasm, and conciseness."
                    },
                    {
                        "question": "What are your top strengths and key areas of development?",
                        "suggested_answer": "Highlight 2-3 technical/interpersonal strengths with brief examples. For weakness, choose a genuine area you are actively improving.",
                        "evaluation_criteria": "Self-awareness, honesty, actionable growth mindset, professional maturity."
                    },
                    {
                        "question": "Why are you interested in joining our company?",
                        "suggested_answer": "Mention specific details about the company's product, engineering culture, recent news, or mission that genuinely excite you.",
                        "evaluation_criteria": "Company research, alignment with organization values, enthusiasm."
                    }
                ],
                "medium": [
                    {
                        "question": "Why are you looking to leave your current role or transition at this stage of your career?",
                        "suggested_answer": "Frame your answer positively around seeking new challenges, growing technical ownership, and aligning with the company's domain.",
                        "evaluation_criteria": "Positive phrasing, career growth orientation, absence of bad-mouthing past employers."
                    },
                    {
                        "question": "Describe your ideal workplace culture and team environment.",
                        "suggested_answer": "Discuss collaborative engineering practices, open communication, code reviews, continuous learning, and ownership.",
                        "evaluation_criteria": "Team fit, adaptability, understanding of collaborative software development."
                    },
                    {
                        "question": "How do you prioritize tasks when faced with multiple competing deadlines?",
                        "suggested_answer": "Explain frameworks like Eisenhower matrix or priority impact vs effort, clear communication with stakeholders, and managing expectations.",
                        "evaluation_criteria": "Time management, stakeholder management, pragmatic prioritization."
                    }
                ],
                "hard": [
                    {
                        "question": "Where do you see yourself in 5 years, and how does this role fit into your long-term career strategy?",
                        "suggested_answer": "Map out career progression towards technical lead or staff engineer role while demonstrating commitment to growing within the company.",
                        "evaluation_criteria": "Strategic vision, realistic career progression, commitment to retention."
                    },
                    {
                        "question": "Tell me about a time when you strongly disagreed with a company direction or management decision. How did you handle it?",
                        "suggested_answer": "Use the STAR method. Focus on data-driven feedback, professional dialogue, and backing the team decision once finalized ('disagree and commit').",
                        "evaluation_criteria": "Conflict resolution, data-driven argumentation, teamwork and alignment."
                    }
                ]
            },
            "technical": {
                "easy": [
                    {
                        "question": "What is the difference between synchronous and asynchronous execution?",
                        "suggested_answer": "Synchronous operations block execution until complete. Asynchronous operations run non-blockingly using event loops, promises, or async/await.",
                        "evaluation_criteria": "Core understanding of concurrency concepts, blocking vs non-blocking I/O."
                    },
                    {
                        "question": "Explain the concept of Object-Oriented Programming (OOP) and its 4 core pillars.",
                        "suggested_answer": "Encapsulation, Abstraction, Inheritance, and Polymorphism. Give brief real-world examples for each.",
                        "evaluation_criteria": "Clear definition of OOP principles and real-world application."
                    },
                    {
                        "question": "What is the difference between SQL (relational) and NoSQL (non-relational) databases?",
                        "suggested_answer": "SQL uses structured tabular schema, ACID compliance, relational joins (e.g. Postgres). NoSQL uses flexible schema (document, key-value, graph), dynamic scaling (e.g. MongoDB).",
                        "evaluation_criteria": "Understanding data models, trade-offs between consistency and horizontal scaling."
                    }
                ],
                "medium": [
                    {
                        "question": "Explain how RESTful APIs differ from GraphQL and gRPC.",
                        "suggested_answer": "REST relies on standard HTTP verbs and resources. GraphQL allows client-driven query fetching without over/under-fetching. gRPC uses HTTP/2 and Protocol Buffers for high-performance RPC.",
                        "evaluation_criteria": "Knowledge of modern API architectures, network protocols, payload optimization."
                    },
                    {
                        "question": "How do indexes work in relational databases, and what are the trade-offs of indexing?",
                        "suggested_answer": "Indexes (typically B-Trees or Hash indexes) speed up SELECT queries from O(N) to O(log N). Trade-off: increased storage space and slower INSERT/UPDATE/DELETE performance.",
                        "evaluation_criteria": "Database internal mechanisms, query performance tuning, write amplification trade-offs."
                    },
                    {
                        "question": "What is Garbage Collection in modern programming languages and how does reference counting differ from mark-and-sweep?",
                        "suggested_answer": "Garbage collection manages memory automatically. Reference counting tracks active pointers (fails on cyclic refs). Mark-and-sweep traverses object graph from roots to free unreachable memory.",
                        "evaluation_criteria": "Memory management concepts, understanding execution engine overhead."
                    }
                ],
                "hard": [
                    {
                        "question": "Explain the CAP theorem and the PACELC extension in distributed system design.",
                        "suggested_answer": "CAP: Consistency, Availability, Partition Tolerance (choose 2 of 3 during network partition). PACELC extends this: If Partition (P), choose Availability (A) vs Consistency (C); Else (E), choose Latency (L) vs Consistency (C).",
                        "evaluation_criteria": "Deep understanding of distributed systems trade-offs, network fault tolerance, database consistency models."
                    },
                    {
                        "question": "How would you design a rate limiter for a high-throughput public API system?",
                        "suggested_answer": "Discuss algorithms like Token Bucket, Leaky Bucket, Fixed Window, and Sliding Window Log. Mention distributed implementation using Redis cluster with atomic Lua scripts or Token Bucket algorithms.",
                        "evaluation_criteria": "System design skills, scalability, concurrency, caching layer choice, edge cases."
                    }
                ]
            },
            "behavioral": {
                "easy": [
                    {
                        "question": "Describe a project you worked on that you are particularly proud of.",
                        "suggested_answer": "Use STAR format (Situation, Task, Action, Result). State the goal, your specific contribution, key challenge, and measurable outcome.",
                        "evaluation_criteria": "STAR method structure, clarity of contribution, measurable impact."
                    },
                    {
                        "question": "How do you handle receiving critical feedback on your code or performance?",
                        "suggested_answer": "Explain that feedback is an opportunity to learn. Describe active listening, asking clarifying questions, and implementing changes constructively.",
                        "evaluation_criteria": "Receptivity, humility, continuous improvement mindset."
                    }
                ],
                "medium": [
                    {
                        "question": "Tell me about a time when you had to meet a tight deadline under challenging constraints.",
                        "suggested_answer": "Detail how you scoped down non-critical requirements, communicated risks early to team members, prioritized core deliverables, and successfully shipped.",
                        "evaluation_criteria": "Pragmatism, communication under pressure, scope management."
                    },
                    {
                        "question": "Give an example of a mistake or bug you introduced into production and how you handled it.",
                        "suggested_answer": "Explain immediate remediation (rollback or quick hotfix), transparent communication post-mortem, root cause analysis, and implementing preventative testing/monitoring.",
                        "evaluation_criteria": "Accountability, blameless post-mortem mindset, prevention focus."
                    }
                ],
                "hard": [
                    {
                        "question": "Describe a scenario where you led a technical project with ambiguous requirements and conflicting stakeholder expectations.",
                        "suggested_answer": "Outline how you gathered requirements, created prototype RFCs, facilitated design alignment, broke down work into iterative sprints, and navigated trade-offs.",
                        "evaluation_criteria": "Leadership, dealing with ambiguity, technical consensus building, stakeholder alignment."
                    }
                ]
            },
            "coding": {
                "easy": [
                    {
                        "question": "Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target` (Two Sum).",
                        "suggested_answer": "Use a Hash Map storing value to index mapping. Iterate through array; for each number `x`, check if `target - x` exists in map. Time Complexity: O(N), Space: O(N).",
                        "evaluation_criteria": "Hash map usage, optimal time complexity, handling edge cases."
                    },
                    {
                        "question": "Write a function to check if a string is a valid palindrome, ignoring non-alphanumeric characters and case.",
                        "suggested_answer": "Use two pointers (left and right) moving inward, skipping non-alphanumeric chars and comparing lowercase equivalents.",
                        "evaluation_criteria": "Two-pointer approach, string manipulation, edge cases."
                    }
                ],
                "medium": [
                    {
                        "question": "Implement an LRU (Least Recently Used) Cache class with `get(key)` and `put(key, value)` operating in O(1) time complexity.",
                        "suggested_answer": "Combine a Doubly Linked List (for O(1) node insertion/removal) with a Hash Map (for O(1) key lookup pointing to list node).",
                        "evaluation_criteria": "Data structure composition, O(1) operations, pointer management."
                    },
                    {
                        "question": "Given an array of intervals, merge all overlapping intervals.",
                        "suggested_answer": "Sort intervals by start time. Iterate through sorted intervals; if current interval overlaps with last merged interval, update end time; else append current interval.",
                        "evaluation_criteria": "Sorting strategy, interval comparison logic, edge cases."
                    }
                ],
                "hard": [
                    {
                        "question": "Given two sorted arrays `nums1` and `nums2` of size `m` and `n`, find the median of the two sorted arrays in O(log(min(m,n))) time complexity.",
                        "suggested_answer": "Binary search on the partition index of the smaller array so that total elements in left partition equal right partition, ensuring max_left <= min_right.",
                        "evaluation_criteria": "Advanced binary search, handling partition edge cases, optimal logarithmic time."
                    }
                ]
            },
            "project_based": {
                "easy": [
                    {
                        "question": "Walk me through the overall architecture of a key project from your resume.",
                        "suggested_answer": "Explain frontend, backend services, database selection, third-party integrations, deployment setup, and data flow.",
                        "evaluation_criteria": "Architectural clarity, end-to-end understanding, ability to explain technology choices."
                    }
                ],
                "medium": [
                    {
                        "question": "What was the most challenging technical decision or architectural trade-off in your recent project?",
                        "suggested_answer": "Describe options considered, evaluated parameters (latency, cost, complexity, maintainability), why you chose the final solution, and outcome.",
                        "evaluation_criteria": "Analytical trade-off evaluation, practical decision-making, technical depth."
                    }
                ],
                "hard": [
                    {
                        "question": "If your primary system experienced a 100x spike in concurrent users tomorrow, what would fail first, and how would you re-architect it?",
                        "suggested_answer": "Identify database connection bottlenecks, slow unindexed queries, synchronous external calls. Detail caching strategy (Redis), horizontal scaling, messaging queues (Kafka/RabbitMQ), and database read replicas.",
                        "evaluation_criteria": "System scalability awareness, bottleneck identification, architectural resilience."
                    }
                ]
            }
        }

    def generate_questions(
        self,
        resume_text: str = "",
        job_description: str = "",
        question_type: str = "all",
        difficulty: str = "medium",
        count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate interview questions based on resume, job description, category, and difficulty.

        Args:
            resume_text: Raw resume text.
            job_description: Raw job description text.
            question_type: 'hr', 'technical', 'behavioral', 'coding', 'project_based', or 'all'.
            difficulty: 'easy', 'medium', or 'hard'.
            count: Number of questions to return (default 10).

        Returns:
            List of dictionaries containing question, type, difficulty, suggested_answer, evaluation_criteria.
        """
        question_type = (question_type or "all").lower().strip()
        difficulty = (difficulty or "medium").lower().strip()
        if difficulty not in ["easy", "medium", "hard"]:
            difficulty = "medium"
        count = max(1, min(count, 30))

        # Attempt AI generation if AIEngine is available
        if self.ai_engine and hasattr(self.ai_engine, "is_available") and self.ai_engine.is_available():
            try:
                system_prompt = (
                    "You are a principal technical interviewer and HR specialist. "
                    "Generate custom, realistic interview questions, suggested answers, and evaluation criteria based on "
                    "the applicant's resume and job description. Return JSON strictly in the requested format."
                )
                prompt = f"""
Resume Text:
{resume_text[:2000]}

Job Description:
{job_description[:2000]}

Requested Category: {question_type}
Requested Difficulty: {difficulty}
Number of Questions: {count}

Respond with a JSON object containing a key "questions" which is a list of exactly {count} question objects.
Each question object MUST have the following keys:
- "question": string
- "type": string (one of: hr, technical, behavioral, coding, project_based)
- "difficulty": string (easy, medium, hard)
- "suggested_answer": string
- "evaluation_criteria": string
"""
                json_res = self.ai_engine.generate_json(prompt, system_prompt=system_prompt)
                if isinstance(json_res, dict) and "questions" in json_res and isinstance(json_res["questions"], list):
                    q_list = json_res["questions"]
                    if len(q_list) > 0:
                        formatted_questions = []
                        for q in q_list[:count]:
                            formatted_questions.append({
                                "question": q.get("question", "Describe your experience."),
                                "type": q.get("type", question_type if question_type != "all" else "technical"),
                                "difficulty": q.get("difficulty", difficulty),
                                "suggested_answer": q.get("suggested_answer", "Focus on structured execution and results."),
                                "evaluation_criteria": q.get("evaluation_criteria", "Clarity and technical depth.")
                            })
                        return formatted_questions
            except Exception as e:
                logger.warning(f"AI question generation failed, using curated question bank fallback: {e}")

        # Fallback to curated question bank
        return self._generate_fallback_questions(resume_text, job_description, question_type, difficulty, count)

    def _generate_fallback_questions(
        self,
        resume_text: str,
        job_description: str,
        question_type: str,
        difficulty: str,
        count: int
    ) -> List[Dict[str, Any]]:
        """Fallback question generator pulling from curated banks and dynamic keyword customization."""
        all_types = ["hr", "technical", "behavioral", "coding", "project_based"]
        selected_types = all_types if question_type == "all" else [question_type] if question_type in all_types else all_types

        candidates: List[Dict[str, Any]] = []

        # Collect matching questions from bank
        for q_type in selected_types:
            type_bank = self._question_bank.get(q_type, {})
            # Try specified difficulty, then fallback difficulties
            diff_levels = [difficulty] + [d for d in ["medium", "easy", "hard"] if d != difficulty]
            for d in diff_levels:
                q_list = type_bank.get(d, [])
                for item in q_list:
                    candidates.append({
                        "question": item["question"],
                        "type": q_type,
                        "difficulty": d,
                        "suggested_answer": item["suggested_answer"],
                        "evaluation_criteria": item["evaluation_criteria"]
                    })

        # Dynamically inject keyword-tailored technical questions if keywords present in resume/JD
        combined_text = (resume_text + " " + job_description).lower()
        extracted_techs = []
        tech_keywords = [
            ("python", "Python", "Explain GIL (Global Interpreter Lock) in Python and how it affects multithreading vs multiprocessing."),
            ("react", "React", "What are React Hooks? Explain the purpose of useEffect, useMemo, and useCallback."),
            ("aws", "AWS", "How would you design a highly available, fault-tolerant infrastructure using AWS services (S3, EC2, RDS, ALB)?"),
            ("docker", "Docker", "Explain containerization vs virtualization, multi-stage Docker builds, and security best practices."),
            ("sql", "SQL", "How do you analyze and optimize a slow SQL query using EXPLAIN ANALYZE?"),
            ("machine learning", "Machine Learning", "Explain bias-variance tradeoff and strategies to prevent model overfitting.")
        ]
        for key, name, custom_q in tech_keywords:
            if key in combined_text:
                extracted_techs.append((name, custom_q))

        for tech_name, custom_q in extracted_techs[:3]:
            candidates.insert(0, {
                "question": custom_q,
                "type": "technical",
                "difficulty": difficulty,
                "suggested_answer": f"Demonstrate deep working knowledge of {tech_name}, best practices, and performance considerations.",
                "evaluation_criteria": f"Depth of technical understanding in {tech_name} and practical application."
            })

        # Ensure unique questions
        unique_questions = []
        seen_q = set()
        for c in candidates:
            if c["question"] not in seen_q:
                seen_q.add(c["question"])
                unique_questions.append(c)

        # Fill up if count is greater than available candidates by duplicating with variations or sampling
        result = []
        if unique_questions:
            while len(result) < count:
                item = unique_questions[len(result) % len(unique_questions)]
                result.append(dict(item))
        else:
            # Absolute baseline default
            result.append({
                "question": "Describe a challenging problem you solved recently.",
                "type": "behavioral",
                "difficulty": difficulty,
                "suggested_answer": "Use STAR format to describe the problem, your action, and measurable outcome.",
                "evaluation_criteria": "Problem solving ability and communication clarity."
            })

        return result[:count]
