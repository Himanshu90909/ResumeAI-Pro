"""
NLP Utilities for ResumeAI Pro.
Provides text processing, skill extraction, section parser, readability metrics,
contact extraction, and semantic analysis with spaCy integration and fallback mechanisms.
"""

import re
import os
import json
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try loading spaCy
_nlp = None
try:
    import spacy
    try:
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        try:
            _nlp = spacy.blank("en")
        except Exception:
            _nlp = None
except ImportError:
    _nlp = None


def count_words(text: str) -> int:
    """
    Count total words in a given text.
    
    Args:
        text: Input string.
        
    Returns:
        Total word count as an integer.
    """
    if not text or not text.strip():
        return 0
    return len(re.findall(r'\b\w+\b', text))


def count_sentences(text: str) -> int:
    """
    Count total sentences in text using spaCy if available, with regex fallback.
    
    Args:
        text: Input string.
        
    Returns:
        Total sentence count.
    """
    if not text or not text.strip():
        return 0

    if _nlp is not None:
        try:
            doc = _nlp(text)
            if doc.has_annotation("SENT_START"):
                sents = list(doc.sents)
                if sents:
                    return len(sents)
        except Exception:
            pass

    # Regex fallback
    sentences = re.split(r'[.!?]+(?:\s+|\n+|$)', text)
    valid_sentences = [s.strip() for s in sentences if s.strip()]
    return max(1, len(valid_sentences)) if text.strip() else 0


def count_syllables(word: str) -> int:
    """
    Estimate the number of syllables in an English word.
    
    Args:
        word: A single word.
        
    Returns:
        Estimated syllable count.
    """
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
    word = re.sub(r'^y', '', word)
    syllables = len(re.findall(r'[aeiouy]{1,2}', word))
    return max(1, syllables)


def compute_readability(text: str) -> float:
    """
    Compute the Flesch Reading Ease score for text.
    Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    
    Args:
        text: Resume or article text.
        
    Returns:
        Flesch reading ease score between 0.0 and 100.0.
    """
    if not text or not text.strip():
        return 0.0

    words = re.findall(r'\b[a-zA-Z]+\b', text)
    num_words = len(words)
    if num_words == 0:
        return 0.0

    num_sentences = count_sentences(text)
    if num_sentences == 0:
        num_sentences = 1

    total_syllables = sum(count_syllables(w) for w in words)

    asl = num_words / num_sentences
    asw = total_syllables / num_words

    score = 206.835 - (1.015 * asl) - (84.6 * asw)
    return round(max(0.0, min(100.0, score)), 2)


def extract_skills(
    text: str, 
    skills_db: Optional[Union[Dict[str, List[str]], str]] = None
) -> Dict[str, List[str]]:
    """
    Extract technical and soft skills from text using keyword matching against skills database.
    
    Args:
        text: Input resume text.
        skills_db: Dictionary of skills categories or path to JSON database file.
        
    Returns:
        Dict with keys for each category (e.g., 'programming_languages', 'frameworks') containing matched skills lists,
        and an 'all_skills' list containing all unique extracted skills.
    """
    if not text:
        return {"all_skills": []}

    skills_data = {}
    if skills_db is None:
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "skills_database.json")
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                skills_data = json.load(f)
    elif isinstance(skills_db, str):
        if os.path.exists(skills_db):
            with open(skills_db, "r", encoding="utf-8") as f:
                skills_data = json.load(f)
    elif isinstance(skills_db, dict):
        skills_data = skills_db

    extracted: Dict[str, List[str]] = {}
    all_matched = set()

    for category, skills in skills_data.items():
        matched_category = []
        for skill in skills:
            escaped_skill = re.escape(skill)
            starts_word = skill[0].isalnum()
            ends_word = skill[-1].isalnum()

            prefix = r'\b' if starts_word else r'(?:^|[\s,;()|/\-:])'
            suffix = r'\b' if ends_word else r'(?:$|[\s,;()|/\-:])'

            pattern = r'(?i)' + prefix + escaped_skill + suffix
            if re.search(pattern, text):
                matched_category.append(skill)
                all_matched.add(skill)

        extracted[category] = matched_category

    extracted["all_skills"] = sorted(list(all_matched))
    return extracted


def extract_sections(text: str) -> Dict[str, str]:
    """
    Detect and group text into standard resume sections (experience, education, projects, skills, certifications, summary).
    
    Args:
        text: Full resume text.
        
    Returns:
        Dict mapping section names to text content.
    """
    if not text:
        return {}

    section_keywords = {
        "summary": ["summary", "professional summary", "executive summary", "objective", "career objective", "about me", "profile"],
        "experience": ["experience", "work experience", "employment history", "work history", "professional experience", "career history"],
        "education": ["education", "academic background", "academic qualifications", "educational background"],
        "skills": ["skills", "technical skills", "core competencies", "technologies", "skills & tools", "areas of expertise"],
        "projects": ["projects", "personal projects", "key projects", "academic projects", "portfolio"],
        "certifications": ["certifications", "licenses & certifications", "certificates", "courses", "credentials"]
    }

    lines = text.splitlines()
    header_indices = []

    for idx, line in enumerate(lines):
        clean_line = line.strip().lower()
        clean_heading = re.sub(r'^[#*=\-\s]+|[#*=\-:\s]+$', '', clean_line)
        for canonical, synonyms in section_keywords.items():
            if clean_heading in synonyms:
                header_indices.append((idx, canonical, line))
                break

    sections: Dict[str, str] = {}
    if not header_indices:
        sections["other"] = text
        return sections

    if header_indices[0][0] > 0:
        header_text = "\n".join(lines[:header_indices[0][0]]).strip()
        if header_text:
            sections["header"] = header_text

    for i, (line_idx, canonical, raw_line) in enumerate(header_indices):
        start_line = line_idx + 1
        end_line = header_indices[i + 1][0] if i + 1 < len(header_indices) else len(lines)
        content = "\n".join(lines[start_line:end_line]).strip()
        if canonical in sections:
            sections[canonical] += "\n\n" + content
        else:
            sections[canonical] = content

    return sections


def extract_contact(text: str) -> Dict[str, Optional[str]]:
    """
    Extract contact details: email, phone, LinkedIn, GitHub, portfolio/website, location.
    
    Args:
        text: Input text (e.g. resume text or header).
        
    Returns:
        Dict with contact keys and extracted string or None.
    """
    contact_info: Dict[str, Optional[str]] = {
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "website": None,
        "location": None
    }
    if not text:
        return contact_info

    # Email
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_match = re.search(email_pattern, text)
    if email_match:
        contact_info["email"] = email_match.group(0).lower()

    # Phone
    phone_pattern = r'(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'
    phone_match = re.search(phone_pattern, text)
    if phone_match:
        contact_info["phone"] = phone_match.group(0).strip()

    # LinkedIn
    linkedin_pattern = r'(?:https?:\/\/)?(?:www\.)?linkedin\.com\/in\/[a-zA-Z0-9_-]+\/?'
    linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
    if linkedin_match:
        contact_info["linkedin"] = linkedin_match.group(0).rstrip('/')

    # GitHub
    github_pattern = r'(?:https?:\/\/)?(?:www\.)?github\.com\/[a-zA-Z0-9_-]+\/?'
    github_match = re.search(github_pattern, text, re.IGNORECASE)
    if github_match:
        contact_info["github"] = github_match.group(0).rstrip('/')

    # Website
    website_pattern = r'(?:https?:\/\/)?(?:www\.)?[a-zA-Z0-9-]+\.(?:com|org|io|net|dev|me|co|app|tech|edu|gov)(?:\/[^\s]*)?'
    for match in re.finditer(website_pattern, text, re.IGNORECASE):
        url = match.group(0)
        if "linkedin.com" not in url.lower() and "github.com" not in url.lower():
            contact_info["website"] = url.rstrip('/')
            break

    # Location
    location_pattern = r'\b([A-Z][a-zA-B\s]{2,25},\s*(?:[A-Z]{2}|[A-Z][a-zA-B\s]{2,15}))\b'
    loc_match = re.search(location_pattern, text[:500])
    if loc_match:
        contact_info["location"] = loc_match.group(0).strip()

    return contact_info


def extract_education(text: str) -> List[Dict[str, str]]:
    """
    Extract education entries (degree, institution, dates, GPA).
    
    Args:
        text: Full resume text.
        
    Returns:
        List of dicts containing extracted education fields.
    """
    education_entries = []
    sections = extract_sections(text)
    edu_text = sections.get("education", text)

    degree_patterns = [
        r'(?i)\b(?:Bachelor|Master|Doctor|Ph\.D\.|PhD|B\.S\.|M\.S\.|B\.A\.|M\.A\.|B\.Tech|M\.Tech|B\.E\.|M\.E\.|BBA|MBA|Associate|Diploma)\b[^\n,]*',
        r'(?i)\b(?:B\.S|M\.S|B\.A|M\.A|BS|MS|BA|MA|BTech|MTech)\b\s+in\s+[^\n,]+'
    ]
    date_pattern = r'\b(?:19|20)\d{2}\s*(?:--?|to|until|\s)\s*(?:(?:19|20)\d{2}|Present|Current|Expected)?\b|\b(?:19|20)\d{2}\b'
    gpa_pattern = r'(?i)\b(?:GPA|CGPA)[:\s]*([0-4]\.\d{1,2}(?:\s*\/\s*4\.0)?|[0-9]\.\d{1,2}(?:\s*\/\s*10(?:\.0)?)?)\b'
    inst_pattern = r'(?i)\b(?:University|College|Institute|Academy|School)\b[^\n,]*'

    blocks = [b.strip() for b in re.split(r'\n\s*\n', edu_text) if b.strip()]

    for block in blocks:
        entry = {
            "degree": "",
            "institution": "",
            "dates": "",
            "gpa": ""
        }
        for dp in degree_patterns:
            deg_match = re.search(dp, block)
            if deg_match:
                entry["degree"] = deg_match.group(0).strip()
                break

        inst_match = re.search(inst_pattern, block)
        if inst_match:
            entry["institution"] = inst_match.group(0).strip()

        date_match = re.search(date_pattern, block)
        if date_match:
            entry["dates"] = date_match.group(0).strip()

        gpa_match = re.search(gpa_pattern, block)
        if gpa_match:
            entry["gpa"] = gpa_match.group(1).strip() if len(gpa_match.groups()) > 0 else gpa_match.group(0).strip()

        if entry["degree"] or entry["institution"]:
            education_entries.append(entry)

    return education_entries


def extract_experience(text: str) -> List[Dict[str, Any]]:
    """
    Extract work experience details (job title, company, dates, bullet points).
    
    Args:
        text: Full resume text.
        
    Returns:
        List of dicts representing work experience items.
    """
    exp_entries = []
    sections = extract_sections(text)
    exp_text = sections.get("experience", text)

    title_patterns = [
        r'(?i)\b(?:Software|Senior|Junior|Lead|Principal|Staff|Full Stack|Frontend|Backend|Data|Machine Learning|DevOps|Cloud|Product|Project|QA|System|Security)\s+(?:Engineer|Developer|Scientist|Architect|Manager|Analyst|Consultant|Intern)\b',
        r'(?i)\b(?:Software Engineer|Data Scientist|Product Manager|DevOps Engineer|Project Manager|Business Analyst|UX Designer)\b'
    ]
    date_pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s*(?:--?|to|\s)\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|Present|Current)?\b|\b(?:19|20)\d{2}\s*(?:--?|to|\s)\s*(?:(?:19|20)\d{2}|Present|Current)?\b'

    blocks = [b.strip() for b in re.split(r'\n\s*\n', exp_text) if b.strip()]

    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue

        title = ""
        company = ""
        dates = ""
        bullets = []

        for tp in title_patterns:
            title_match = re.search(tp, block)
            if title_match:
                title = title_match.group(0).strip()
                break

        date_match = re.search(date_pattern, block)
        if date_match:
            dates = date_match.group(0).strip()

        for line in lines:
            line_str = line.strip()
            if line_str.startswith(('•', '-', '*', '')):
                bullets.append(re.sub(r'^[•\-\*\s]+', '', line_str))
            elif not title and any(term in line_str.lower() for term in ['engineer', 'developer', 'manager', 'lead', 'intern', 'analyst', 'architect']):
                title = line_str
            elif not company and not any(term in line_str.lower() for term in ['engineer', 'developer', 'manager', 'lead', 'intern', 'analyst', 'architect']) and len(line_str) < 60:
                if not re.search(date_pattern, line_str):
                    company = line_str

        if title or company or bullets:
            exp_entries.append({
                "title": title or "Position",
                "company": company or "Company",
                "dates": dates,
                "bullets": bullets
            })

    return exp_entries


def extract_keywords(text: str, top_n: int = 20) -> List[Tuple[str, float]]:
    """
    Extract top TF-IDF keywords from input text.
    
    Args:
        text: Input string.
        top_n: Number of top keywords to extract.
        
    Returns:
        List of tuples: (keyword, float_score) sorted by score descending.
    """
    if not text or not text.strip():
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=top_n * 2
        )
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        return [(kw, round(float(score), 4)) for kw, score in keyword_scores[:top_n]]
    except Exception:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stopwords = {'and', 'the', 'for', 'with', 'that', 'this', 'from', 'have', 'were', 'been', 'their', 'which', 'about'}
        filtered = [w for w in words if w not in stopwords]
        from collections import Counter
        counts = Counter(filtered).most_common(top_n)
        max_c = counts[0][1] if counts else 1
        return [(kw, round(count / max_c, 4)) for kw, count in counts]


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two text snippets using TF-IDF vectors.
    
    Args:
        text1: First text snippet.
        text2: Second text snippet.
        
    Returns:
        Cosine similarity score as float in range [0.0, 1.0].
    """
    if not text1 or not text2:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(sim), 4)
    except Exception:
        set1 = set(re.findall(r'\b\w+\b', text1.lower()))
        set2 = set(re.findall(r'\b\w+\b', text2.lower()))
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return round(float(intersection / union), 4)
