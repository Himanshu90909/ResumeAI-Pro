"""
PDF Generation utilities for ResumeAI Pro.
Provides template-based PDF resume creation and detailed ATS report generation using ReportLab.
"""

from io import BytesIO
import re
from typing import Dict, List, Any, Optional, Union
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


def _get_theme_colors(template_name: str) -> Dict[str, Any]:
    """
    Get color palette and font settings for requested resume template theme.
    """
    name = (template_name or "modern").lower().strip()

    if name == "harvard":
        return {
            "font_main": "Times-Roman",
            "font_bold": "Times-Bold",
            "font_italic": "Times-Italic",
            "primary": colors.HexColor("#1B365D"),  # Harvard Navy
            "secondary": colors.HexColor("#333333"),
            "text": colors.HexColor("#111111"),
            "line": colors.HexColor("#1B365D"),
            "align_header": TA_CENTER,
            "header_uppercase": True
        }
    elif name == "minimal":
        return {
            "font_main": "Helvetica",
            "font_bold": "Helvetica-Bold",
            "font_italic": "Helvetica-Oblique",
            "primary": colors.HexColor("#2D3748"),  # Dark Charcoal
            "secondary": colors.HexColor("#64748B"),  # Slate Gray
            "text": colors.HexColor("#1F2937"),
            "line": colors.HexColor("#E2E8F0"),
            "align_header": TA_LEFT,
            "header_uppercase": False
        }
    elif name == "developer":
        return {
            "font_main": "Helvetica",
            "font_bold": "Helvetica-Bold",
            "font_italic": "Helvetica-Oblique",
            "primary": colors.HexColor("#0D9488"),  # Teal
            "secondary": colors.HexColor("#0F172A"),  # Slate Black
            "text": colors.HexColor("#0F172A"),
            "line": colors.HexColor("#0D9488"),
            "align_header": TA_LEFT,
            "header_uppercase": True
        }
    else:  # modern (default)
        return {
            "font_main": "Helvetica",
            "font_bold": "Helvetica-Bold",
            "font_italic": "Helvetica-Oblique",
            "primary": colors.HexColor("#2563EB"),  # Royal Blue
            "secondary": colors.HexColor("#1E293B"),  # Dark Blue-Gray
            "text": colors.HexColor("#1E293B"),
            "line": colors.HexColor("#2563EB"),
            "align_header": TA_LEFT,
            "header_uppercase": True
        }


def generate_resume_pdf(resume_data: Dict[str, Any], template_name: str = "modern") -> bytes:
    """
    Generate a styled PDF resume byte stream using ReportLab.
    
    Args:
        resume_data: Dictionary containing contact, summary, experience, education, skills, projects, certifications.
        template_name: One of 'harvard', 'modern', 'minimal', 'developer'.
        
    Returns:
        PDF bytes buffer.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    theme = _get_theme_colors(template_name)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'ResumeTitle',
        parent=styles['Normal'],
        fontName=theme['font_bold'],
        fontSize=20 if template_name == "harvard" else 22,
        leading=24,
        textColor=theme['primary'],
        alignment=theme['align_header']
    )

    contact_style = ParagraphStyle(
        'ResumeContact',
        parent=styles['Normal'],
        fontName=theme['font_main'],
        fontSize=9.5,
        leading=13,
        textColor=theme['secondary'],
        alignment=theme['align_header']
    )

    section_heading_style = ParagraphStyle(
        'ResumeSectionHeading',
        parent=styles['Normal'],
        fontName=theme['font_bold'],
        fontSize=12,
        leading=15,
        textColor=theme['primary'],
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'ResumeBody',
        parent=styles['Normal'],
        fontName=theme['font_main'],
        fontSize=10,
        leading=13,
        textColor=theme['text']
    )

    bullet_style = ParagraphStyle(
        'ResumeBullet',
        parent=styles['Normal'],
        fontName=theme['font_main'],
        fontSize=9.5,
        leading=13,
        textColor=theme['text'],
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2
    )

    bold_text_style = ParagraphStyle(
        'ResumeBoldText',
        parent=styles['Normal'],
        fontName=theme['font_bold'],
        fontSize=10,
        leading=13,
        textColor=theme['text']
    )

    italic_text_style = ParagraphStyle(
        'ResumeItalicText',
        parent=styles['Normal'],
        fontName=theme['font_italic'],
        fontSize=9.5,
        leading=13,
        textColor=theme['secondary']
    )

    story = []

    # --- Header / Contact Block ---
    contact_data = resume_data.get("contact", {})
    name = contact_data.get("name", "John Doe")
    email = contact_data.get("email", "")
    phone = contact_data.get("phone", "")
    linkedin = contact_data.get("linkedin", "")
    github = contact_data.get("github", "")
    location = contact_data.get("location", "")
    website = contact_data.get("website", "")

    story.append(Paragraph(name, title_style))
    story.append(Spacer(1, 4))

    contact_parts = [p for p in [email, phone, location, linkedin, github, website] if p]
    contact_line = "  |  ".join(contact_parts)
    if contact_line:
        story.append(Paragraph(contact_line, contact_style))
        story.append(Spacer(1, 6))

    story.append(HRFlowable(width="100%", thickness=1.5, color=theme['line'], spaceBefore=2, spaceAfter=8))

    # Helper function for section headings
    def add_section_heading(title: str):
        heading_text = title.upper() if theme['header_uppercase'] else title
        story.append(Paragraph(heading_text, section_heading_style))
        story.append(HRFlowable(width="100%", thickness=0.75, color=theme['line'], spaceBefore=2, spaceAfter=6))

    # --- Professional Summary ---
    summary = resume_data.get("summary", "")
    if summary:
        add_section_heading("Professional Summary")
        story.append(Paragraph(summary, body_style))
        story.append(Spacer(1, 6))

    # --- Technical Skills ---
    skills = resume_data.get("skills", [])
    if skills:
        add_section_heading("Skills & Expertise")
        if isinstance(skills, dict):
            skill_lines = []
            for cat, s_list in skills.items():
                if cat != "all_skills" and s_list:
                    cat_name = cat.replace('_', ' ').title()
                    skill_lines.append(f"<b>{cat_name}:</b> {', '.join(s_list)}")
            if skill_lines:
                story.append(Paragraph("<br/>".join(skill_lines), body_style))
        elif isinstance(skills, list):
            skills_str = ", ".join(str(s) for s in skills)
            story.append(Paragraph(f"<b>Technical Skills:</b> {skills_str}", body_style))
        story.append(Spacer(1, 6))

    # --- Work Experience ---
    experience = resume_data.get("experience", [])
    if experience:
        add_section_heading("Work Experience")
        for item in experience:
            item_story = []
            title = item.get("title", "Position")
            company = item.get("company", "Company")
            dates = item.get("dates", "")
            bullets = item.get("bullets", [])

            # Header table for Experience (Title + Company on left, Dates on right)
            left_text = f"<b>{title}</b> -- <i>{company}</i>"
            right_text = f"<i>{dates}</i>"
            
            table_data = [[
                Paragraph(left_text, body_style),
                Paragraph(right_text, ParagraphStyle('RightAlign', parent=body_style, alignment=TA_RIGHT))
            ]]
            table = Table(table_data, colWidths=[380, 160])
            table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
            ]))
            item_story.append(table)

            for bullet in bullets:
                if bullet.strip():
                    item_story.append(Paragraph(f"• {bullet.strip()}", bullet_style))

            item_story.append(Spacer(1, 4))
            story.append(KeepTogether(item_story))

        story.append(Spacer(1, 4))

    # --- Projects ---
    projects = resume_data.get("projects", [])
    if projects:
        add_section_heading("Key Projects")
        for proj in projects:
            proj_story = []
            p_name = proj.get("name", "Project")
            p_desc = proj.get("description", "")
            p_tech = proj.get("technologies", [])
            p_bullets = proj.get("bullets", [])
            p_link = proj.get("link", "")

            tech_str = f" [{', '.join(p_tech)}]" if p_tech else ""
            link_str = f" (<i>{p_link}</i>)" if p_link else ""
            header_str = f"<b>{p_name}</b>{tech_str}{link_str}"

            proj_story.append(Paragraph(header_str, body_style))
            if p_desc:
                proj_story.append(Paragraph(p_desc, italic_text_style))

            for bullet in p_bullets:
                if bullet.strip():
                    proj_story.append(Paragraph(f"• {bullet.strip()}", bullet_style))

            proj_story.append(Spacer(1, 4))
            story.append(KeepTogether(proj_story))

        story.append(Spacer(1, 4))

    # --- Education ---
    education = resume_data.get("education", [])
    if education:
        add_section_heading("Education")
        for edu in education:
            degree = edu.get("degree", "Degree")
            institution = edu.get("institution", "Institution")
            dates = edu.get("dates", "")
            gpa = edu.get("gpa", "")

            gpa_str = f" | GPA: {gpa}" if gpa else ""
            left_text = f"<b>{degree}</b>, {institution}{gpa_str}"
            right_text = f"<i>{dates}</i>"

            table_data = [[
                Paragraph(left_text, body_style),
                Paragraph(right_text, ParagraphStyle('RightAlignEdu', parent=body_style, alignment=TA_RIGHT))
            ]]
            table = Table(table_data, colWidths=[380, 160])
            table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
            ]))
            story.append(table)
            story.append(Spacer(1, 2))

        story.append(Spacer(1, 4))

    # --- Certifications ---
    certifications = resume_data.get("certifications", [])
    if certifications:
        add_section_heading("Certifications")
        cert_items = []
        for cert in certifications:
            if isinstance(cert, str):
                cert_items.append(f"• {cert}")
            elif isinstance(cert, dict):
                c_name = cert.get("name", "")
                c_issuer = cert.get("issuer", "")
                c_date = cert.get("date", "")
                details = f" -- {c_issuer}" if c_issuer else ""
                date_str = f" ({c_date})" if c_date else ""
                cert_items.append(f"• <b>{c_name}</b>{details}{date_str}")

        for item in cert_items:
            story.append(Paragraph(item, bullet_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def generate_report_pdf(analysis_data: Dict[str, Any]) -> bytes:
    """
    Generate an ATS Analysis Report PDF byte stream using ReportLab.
    
    Args:
        analysis_data: Dictionary containing candidate_name, overall_score, readability_score,
                       skills_matched, skills_missing, section_scores, suggestions, bullet_improvements.
                       
    Returns:
        PDF bytes buffer.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    primary_color = colors.HexColor("#1E3A8A")  # Dark Blue
    secondary_color = colors.HexColor("#3B82F6")  # Bright Blue
    bg_light = colors.HexColor("#F1F5F9")
    dark_text = colors.HexColor("#0F172A")

    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=primary_color,
        alignment=TA_LEFT
    )

    subtitle_style = ParagraphStyle(
        'ReportSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        alignment=TA_LEFT
    )

    section_heading = ParagraphStyle(
        'ReportSectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=dark_text
    )

    bullet_style = ParagraphStyle(
        'ReportBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=dark_text,
        leftIndent=12
    )

    story = []

    # Title Banner
    candidate_name = analysis_data.get("candidate_name", "Candidate")
    story.append(Paragraph("ResumeAI Pro -- ATS Analysis Report", title_style))
    story.append(Paragraph(f"Prepared for: <b>{candidate_name}</b>", subtitle_style))
    story.append(Spacer(1, 10))

    # Overall ATS Score Card Table
    overall_score = analysis_data.get("overall_score", 75)
    readability = analysis_data.get("readability_score", 65.0)
    skills_matched_cnt = len(analysis_data.get("skills_matched", []))
    skills_missing_cnt = len(analysis_data.get("skills_missing", []))

    score_color = "#16A34A" if overall_score >= 80 else ("#D97706" if overall_score >= 60 else "#DC2626")

    score_card_data = [
        [
            Paragraph(f"<font size=28 color='{score_color}'><b>{overall_score}/100</b></font><br/><font size=9 color='#64748B'>Overall ATS Score</font>", ParagraphStyle('ScoreCenter', parent=body_style, alignment=TA_CENTER)),
            Paragraph(f"<font size=18 color='#1E293B'><b>{readability:.1f}</b></font><br/><font size=9 color='#64748B'>Readability Ease</font>", ParagraphStyle('ScoreCenter2', parent=body_style, alignment=TA_CENTER)),
            Paragraph(f"<font size=18 color='#16A34A'><b>{skills_matched_cnt}</b></font><br/><font size=9 color='#64748B'>Skills Matched</font>", ParagraphStyle('ScoreCenter3', parent=body_style, alignment=TA_CENTER)),
            Paragraph(f"<font size=18 color='#DC2626'><b>{skills_missing_cnt}</b></font><br/><font size=9 color='#64748B'>Skills Missing</font>", ParagraphStyle('ScoreCenter4', parent=body_style, alignment=TA_CENTER)),
        ]
    ]

    card_table = Table(score_card_data, colWidths=[135, 135, 135, 135])
    card_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 12))

    # --- Section Breakdown Table ---
    section_scores = analysis_data.get("section_scores", {
        "Contact Info": 90,
        "Professional Summary": 75,
        "Work Experience": 80,
        "Education": 95,
        "Skills Match": 70
    })

    if section_scores:
        story.append(Paragraph("Section Completeness & Impact Breakdown", section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=primary_color, spaceBefore=2, spaceAfter=6))

        breakdown_data = [["Section", "Score", "Status", "Evaluation"]]
        for sec, sc in section_scores.items():
            status = "Strong" if sc >= 80 else ("Needs Work" if sc >= 60 else "Critical Gap")
            color_hex = "#16A34A" if sc >= 80 else ("#D97706" if sc >= 60 else "#DC2626")
            eval_text = "Optimized with high keyword match." if sc >= 80 else "Add more quantifiable metrics and action verbs."
            breakdown_data.append([
                sec,
                f"{sc}%",
                Paragraph(f"<font color='{color_hex}'><b>{status}</b></font>", body_style),
                eval_text
            ])

        sec_table = Table(breakdown_data, colWidths=[130, 60, 90, 260])
        sec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(sec_table)
        story.append(Spacer(1, 10))

    # --- Skills Match Table ---
    matched = analysis_data.get("skills_matched", [])
    missing = analysis_data.get("skills_missing", [])

    story.append(Paragraph("Skills Analysis & Gap Analysis", section_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=primary_color, spaceBefore=2, spaceAfter=6))

    matched_str = ", ".join(matched) if matched else "None detected"
    missing_str = ", ".join(missing) if missing else "None -- Great match!"

    skills_table_data = [
        [
            Paragraph("<font color='#16A34A'><b>Matched Skills Found:</b></font>", body_style),
            Paragraph(matched_str, body_style)
        ],
        [
            Paragraph("<font color='#DC2626'><b>Recommended Skills to Add:</b></font>", body_style),
            Paragraph(missing_str, body_style)
        ]
    ]

    sk_table = Table(skills_table_data, colWidths=[160, 380])
    sk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), bg_light),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(sk_table)
    story.append(Spacer(1, 10))

    # --- Key Recommendations ---
    suggestions = analysis_data.get("suggestions", [
        "Include more quantifiable achievements with percentages, dollar amounts, or team size.",
        "Add missing target job skills to your Skills section.",
        "Ensure work experience bullet points begin with strong active verbs."
    ])

    if suggestions:
        story.append(Paragraph("Actionable Recommendations", section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=primary_color, spaceBefore=2, spaceAfter=6))
        for sug in suggestions:
            story.append(Paragraph(f"• {sug}", bullet_style))
        story.append(Spacer(1, 10))

    # --- Bullet Point Improvements ---
    bullet_improvements = analysis_data.get("bullet_improvements", [])
    if bullet_improvements:
        story.append(Paragraph("Suggested Bullet Point Improvements", section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=primary_color, spaceBefore=2, spaceAfter=6))

        for item in bullet_improvements:
            orig = item.get("original", "")
            imp = item.get("improved", "")
            if orig and imp:
                b_data = [
                    [Paragraph("<font color='#DC2626'><b>Before:</b></font> " + orig, body_style)],
                    [Paragraph("<font color='#16A34A'><b>After (AI Improved):</b></font> " + imp, body_style)]
                ]
                b_table = Table(b_data, colWidths=[540])
                b_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#FEF2F2")),
                    ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#F0FDF4")),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(b_table)
                story.append(Spacer(1, 4))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
