class ResumeTemplateGenerator:
    def __init__(self):
        self.templates = ["Modern Minimalist", "Executive Slate", "Creative Tech", "Academic Standard"]

    def get_templates(self) -> list:
        return self.templates

    def render_html_template(self, resume_data: dict, template_style: str = "Modern Minimalist") -> str:
        p_info = resume_data.get("personal_info", {})
        name = p_info.get("name", "Jane Doe")
        email = p_info.get("email", "jane.doe@example.com")
        phone = p_info.get("phone", "+1 (555) 019-2834")
        summary = resume_data.get("summary", "Software Engineer with background in building robust web applications.")

        skills = resume_data.get("skills", {})
        if isinstance(skills, dict):
            skill_list = skills.get("all", ["Python", "JavaScript", "SQL", "React", "Docker"])
        else:
            skill_list = skills if isinstance(skills, list) else ["Python", "React"]

        experience = resume_data.get("experience", [
            "Senior Engineer | Acme Corp (2022-Present) - Spearheaded core web platforms.",
            "Software Developer | Tech Solutions (2020-2022) - Built scalable cloud APIs."
        ])

        education = resume_data.get("education", [
            "B.S. Computer Science | State University (2016-2020)"
        ])

        skills_pills = "".join([f"<span style='background:#e0e7ff; color:#3730a3; padding:4px 10px; border-radius:12px; margin:2px; font-size:12px; display:inline-block;'>{s}</span>" for s in skill_list])
        
        exp_html = ""
        for item in experience:
            exp_html += f"<li style='margin-bottom:8px;'>{item}</li>"

        edu_html = ""
        for item in education:
            edu_html += f"<li style='margin-bottom:6px;'>{item}</li>"

        if template_style == "Executive Slate":
            bg_color = "#f8fafc"
            accent_color = "#0f172a"
            border_color = "#334155"
        elif template_style == "Creative Tech":
            bg_color = "#faf5ff"
            accent_color = "#7e22ce"
            border_color = "#a855f7"
        elif template_style == "Academic Standard":
            bg_color = "#ffffff"
            accent_color = "#1e3a8a"
            border_color = "#1e40af"
        else: # Modern Minimalist
            bg_color = "#ffffff"
            accent_color = "#2563eb"
            border_color = "#3b82f6"

        html_content = f"""
        <div style="background-color: {bg_color}; padding: 30px; border-radius: 12px; border: 1px solid #e2e8f0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #1e293b;">
            <div style="border-bottom: 3px solid {accent_color}; padding-bottom: 12px; margin-bottom: 20px;">
                <h1 style="margin: 0; color: {accent_color}; font-size: 26px;">{name}</h1>
                <p style="margin: 4px 0 0 0; color: #64748b; font-size: 14px;">{email} &bull; {phone} &bull; San Francisco, CA</p>
            </div>
            
            <div style="margin-bottom: 20px;">
                <h3 style="color: {accent_color}; margin-bottom: 6px; text-transform: uppercase; font-size: 14px; letter-spacing: 1px;">Summary</h3>
                <p style="font-size: 14px; line-height: 1.5; color: #334155;">{summary}</p>
            </div>

            <div style="margin-bottom: 20px;">
                <h3 style="color: {accent_color}; margin-bottom: 8px; text-transform: uppercase; font-size: 14px; letter-spacing: 1px;">Core Competencies & Skills</h3>
                <div>{skills_pills}</div>
            </div>

            <div style="margin-bottom: 20px;">
                <h3 style="color: {accent_color}; margin-bottom: 8px; text-transform: uppercase; font-size: 14px; letter-spacing: 1px;">Work Experience</h3>
                <ul style="padding-left: 20px; font-size: 14px; color: #334155; line-height: 1.5;">
                    {exp_html}
                </ul>
            </div>

            <div>
                <h3 style="color: {accent_color}; margin-bottom: 8px; text-transform: uppercase; font-size: 14px; letter-spacing: 1px;">Education</h3>
                <ul style="padding-left: 20px; font-size: 14px; color: #334155; line-height: 1.5;">
                    {edu_html}
                </ul>
            </div>
        </div>
        """
        return html_content
