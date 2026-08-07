"""Student Dashboard Module for ResumeAI Pro.

Provides progress tracking, application analytics, skill development monitoring,
resume version comparisons, and career readiness summary charts for students.
"""

import sys
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


class StudentDashboard:
    """Student career analytics and progress tracking dashboard service."""

    def __init__(self) -> None:
        self._local_state: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Session State Persistence Helpers
    # -------------------------------------------------------------------------

    def _is_streamlit_context(self) -> bool:
        """Checks if code is running inside an active Streamlit runtime context."""
        if "streamlit" not in sys.modules:
            return False
        try:
            import streamlit as st
            return hasattr(st, "runtime") and hasattr(st.runtime, "exists") and st.runtime.exists()
        except Exception:
            return False

    def _get_from_session(self, key: str, fallback: Any) -> Any:
        """Retrieves key from Streamlit session_state or local dictionary."""
        if self._is_streamlit_context():
            try:
                import streamlit as st
                return st.session_state.get(key, fallback)
            except Exception:
                pass
        return self._local_state.get(key, fallback)

    def _save_to_session(self, key: str, value: Any) -> None:
        """Saves key to Streamlit session_state or local dictionary."""
        self._local_state[key] = value
        if self._is_streamlit_context():
            try:
                import streamlit as st
                st.session_state[key] = value
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # 1. ATS History Progression
    # -------------------------------------------------------------------------

    def track_ats_history(
        self, ats_scores_list: Optional[List[Dict[str, Any]]] = None
    ) -> go.Figure:
        """Tracks ATS score progression over time.

        Args:
            ats_scores_list: List of dicts, e.g. [
                {"date": "2026-01-01", "score": 62, "version": "v1.0", "target_role": "Backend Dev"},
                {"date": "2026-01-15", "score": 75, "version": "v1.1", "target_role": "Backend Dev"},
                {"date": "2026-02-01", "score": 88, "version": "v2.0", "target_role": "Backend Dev"},
            ]

        Returns:
            Plotly Figure line chart showing ATS score progression over time.
        """
        if ats_scores_list is None:
            ats_scores_list = self._get_from_session("ats_history", [])

        if not ats_scores_list:
            ats_scores_list = [
                {"date": "2026-01-01", "score": 55, "version": "v1.0", "target_role": "General"},
                {"date": "2026-01-15", "score": 68, "version": "v1.1", "target_role": "Software Engineer"},
                {"date": "2026-02-01", "score": 82, "version": "v2.0", "target_role": "Backend Engineer"},
            ]

        self._save_to_session("ats_history", ats_scores_list)

        df = pd.DataFrame(ats_scores_list)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df = df.sort_values(by="date").reset_index(drop=True)

        fig = go.Figure()

        # Target benchmark line
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=[80] * len(df),
            mode="lines",
            name="Target ATS Benchmark (80%)",
            line=dict(color="#10b981", width=2, dash="dash"),
            hovertemplate="Target Benchmark: 80%<extra></extra>"
        ))

        # ATS Score line
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["score"],
            mode="lines+markers+text",
            name="ATS Match Score",
            text=df["score"].astype(str) + "%",
            textposition="top center",
            line=dict(color="#0284c7", width=3),
            marker=dict(size=10, color="#0284c7", symbol="circle"),
            hovertemplate="<b>Date:</b> %{x}<br><b>Score:</b> %{y}%<br><extra></extra>"
        ))

        fig.update_layout(
            title=dict(text="ATS Optimization Score Progression", font=dict(size=18, family="Arial")),
            xaxis=dict(title="Date", showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title="ATS Score (%)", range=[0, 105], showgrid=True, gridcolor="#f1f5f9"),
            template="plotly_white",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        return fig

    # -------------------------------------------------------------------------
    # 2. Application Tracker
    # -------------------------------------------------------------------------

    def track_applications(
        self, applications_list: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Tracks job application pipeline metrics and charts.

        Args:
            applications_list: List of dicts, e.g. [
                {"company": "Google", "role": "SWE", "status": "Applied", "date": "2026-01-10"},
                {"company": "Meta", "role": "Backend", "status": "Interview", "date": "2026-01-15"},
                {"company": "Amazon", "role": "SDE I", "status": "Offer", "date": "2026-02-01"},
            ]

        Returns:
            Dict containing metrics, status_chart (Plotly), timeline_chart (Plotly), and applications_df.
        """
        if applications_list is None:
            applications_list = self._get_from_session("applications_list", [])

        if not applications_list:
            applications_list = [
                {"company": "Tech Corp", "role": "Software Engineer", "status": "Applied", "date": "2026-01-05"},
                {"company": "Innovate Inc", "role": "Backend Developer", "status": "Interview", "date": "2026-01-12"},
                {"company": "Cloud Systems", "role": "DevOps Engineer", "status": "Applied", "date": "2026-01-20"},
                {"company": "Data Dynamics", "role": "Data Engineer", "status": "Offer", "date": "2026-02-01"},
                {"company": "StartupX", "role": "Full Stack Dev", "status": "Rejected", "date": "2026-02-03"},
            ]

        self._save_to_session("applications_list", applications_list)

        df = pd.DataFrame(applications_list)

        total_apps = len(df)
        status_counts = df["status"].value_counts().to_dict() if "status" in df.columns else {}

        interviews = status_counts.get("Interview", 0) + status_counts.get("Interviewing", 0)
        offers = status_counts.get("Offer", 0) + status_counts.get("Offered", 0)
        rejections = status_counts.get("Rejected", 0)

        responded = interviews + offers + rejections
        response_rate = round((responded / total_apps * 100.0), 1) if total_apps > 0 else 0.0
        offer_rate = round((offers / total_apps * 100.0), 1) if total_apps > 0 else 0.0

        colors = {
            "Applied": "#0284c7",
            "Interview": "#f59e0b",
            "Interviewing": "#f59e0b",
            "Offer": "#10b981",
            "Offered": "#10b981",
            "Rejected": "#ef4444"
        }

        status_df = pd.DataFrame(list(status_counts.items()), columns=["Status", "Count"])
        status_chart = px.pie(
            status_df,
            names="Status",
            values="Count",
            hole=0.4,
            title="Application Status Breakdown",
            color="Status",
            color_discrete_map=colors
        )
        status_chart.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=50, b=20))

        if "date" in df.columns:
            df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            timeline_df = df.groupby(["date_str", "status"]).size().reset_index(name="count")
            timeline_chart = px.bar(
                timeline_df,
                x="date_str",
                y="count",
                color="status",
                title="Application Activity Timeline",
                color_discrete_map=colors,
                labels={"date_str": "Date", "count": "Applications"}
            )
            timeline_chart.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=50, b=20))
        else:
            timeline_chart = go.Figure()

        metrics = {
            "total_applications": total_apps,
            "active_interviews": interviews,
            "offers_received": offers,
            "response_rate_pct": response_rate,
            "offer_rate_pct": offer_rate
        }

        return {
            "metrics": metrics,
            "status_chart": status_chart,
            "timeline_chart": timeline_chart,
            "applications_df": df
        }

    # -------------------------------------------------------------------------
    # 3. Skill Growth & Proficiency Tracking
    # -------------------------------------------------------------------------

    def track_skill_growth(
        self, skills_history: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> go.Figure:
        """Tracks skill growth and proficiency levels as a Radar / Spider Chart.

        Args:
            skills_history: Dict of skill proficiencies {"Python": 85, "SQL": 70, ...}
                or List of skill snapshot dicts over time.

        Returns:
            Plotly Figure Radar chart.
        """
        if skills_history is None:
            skills_history = self._get_from_session("skills_history", {})

        if not skills_history:
            skills_dict = {
                "Python": 85,
                "Data Structures": 75,
                "System Design": 60,
                "SQL & Databases": 80,
                "Docker / DevOps": 65,
                "Web Frameworks": 78
            }
        elif isinstance(skills_history, dict):
            skills_dict = skills_history
        elif isinstance(skills_history, list) and len(skills_history) > 0:
            latest = skills_history[-1]
            skills_dict = latest.get("skills", latest) if isinstance(latest, dict) else {}
        else:
            skills_dict = {}

        self._save_to_session("skills_history", skills_dict)

        categories = list(skills_dict.keys())
        values = [float(v) for v in skills_dict.values()]

        if not categories:
            categories = ["Python", "SQL", "Git", "Problem Solving"]
            values = [70, 60, 80, 75]

        r_vals = values + [values[0]]
        theta_vals = categories + [categories[0]]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=theta_vals,
            fill="toself",
            name="Current Proficiency",
            fillcolor="rgba(2, 132, 199, 0.25)",
            line=dict(color="#0284c7", width=2)
        ))

        fig.add_trace(go.Scatterpolar(
            r=[90] * len(r_vals),
            theta=theta_vals,
            mode="lines",
            name="Target Benchmark (90%)",
            line=dict(color="#10b981", width=1.5, dash="dash")
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title=dict(text="Skill Proficiency & Radar Matrix", font=dict(size=18)),
            template="plotly_white",
            height=420,
            margin=dict(l=50, r=50, t=60, b=50)
        )

        return fig

    # -------------------------------------------------------------------------
    # 4. Resume Versions Comparison
    # -------------------------------------------------------------------------

    def resume_versions_comparison(
        self, versions_list: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Compares multiple resume versions across metrics.

        Args:
            versions_list: List of dicts, e.g. [
                {"version": "v1.0", "ats_score": 60, "skill_count": 8, "word_count": 350, "date": "2026-01-01"},
                {"version": "v2.0", "ats_score": 85, "skill_count": 15, "word_count": 480, "date": "2026-02-01"}
            ]

        Returns:
            Dict containing comparison_df and comparison_chart (Plotly Figure).
        """
        if versions_list is None:
            versions_list = self._get_from_session("resume_versions", [])

        if not versions_list:
            versions_list = [
                {"version": "v1.0 (Draft)", "date": "2026-01-05", "ats_score": 58, "skill_count": 8, "word_count": 320, "key_changes": "Initial basic resume"},
                {"version": "v1.1 (Standard)", "date": "2026-01-20", "ats_score": 72, "skill_count": 12, "word_count": 420, "key_changes": "Added project metrics"},
                {"version": "v2.0 (Targeted)", "date": "2026-02-05", "ats_score": 88, "skill_count": 18, "word_count": 510, "key_changes": "Added ATS tech keywords"},
            ]

        self._save_to_session("resume_versions", versions_list)

        df = pd.DataFrame(versions_list)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df["version"],
            y=df["ats_score"],
            name="ATS Score (%)",
            marker_color="#0284c7",
            text=df["ats_score"].astype(str) + "%",
            textposition="auto"
        ))

        fig.add_trace(go.Bar(
            x=df["version"],
            y=df["skill_count"],
            name="Skills Identified",
            marker_color="#f59e0b",
            text=df["skill_count"].astype(str),
            textposition="auto"
        ))

        fig.update_layout(
            barmode="group",
            title="Resume Iteration Benchmark Comparison",
            xaxis_title="Resume Version",
            yaxis_title="Score / Count",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=50, b=40)
        )

        return {
            "comparison_df": df,
            "comparison_chart": fig
        }

    # -------------------------------------------------------------------------
    # 5. Overall Progress Summary
    # -------------------------------------------------------------------------

    def generate_summary(
        self, user_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generates overall student career progression summary.

        Args:
            user_data: Dict containing candidate overall statistics or None to load from state.

        Returns:
            Dict containing progress metrics, readiness index score, and recommended next steps.
        """
        if user_data is None:
            user_data = self._get_from_session("user_profile_summary", {})

        ats_history = self._get_from_session("ats_history", [])
        apps_data = self.track_applications()

        current_ats = ats_history[-1]["score"] if ats_history else 75
        prev_ats = ats_history[0]["score"] if len(ats_history) > 1 else current_ats
        score_diff = current_ats - prev_ats

        app_metrics = apps_data["metrics"]

        app_weight = min(30.0, (app_metrics["total_applications"] / 10.0) * 30.0)
        resp_weight = min(30.0, app_metrics["response_rate_pct"] * 0.3)
        ats_weight = (current_ats / 100.0) * 40.0
        readiness_index = round(min(100.0, ats_weight + app_weight + resp_weight), 1)

        recommendations = []
        if current_ats < 80:
            recommendations.append("Optimize resume keywords using Harvard or Developer templates to hit 80%+ ATS match score.")
        if app_metrics["total_applications"] < 5:
            recommendations.append("Increase job application volume to at least 10 active applications.")
        if app_metrics["active_interviews"] == 0:
            recommendations.append("Tailor projects section with measurable quantitative metrics to improve interview conversion.")
        if not recommendations:
            recommendations.append("Excellent progress! Focus on mock interview prep and system design practice.")

        return {
            "current_ats_score": current_ats,
            "ats_score_change": score_diff,
            "readiness_index": readiness_index,
            "applications_summary": app_metrics,
            "recommendations": recommendations,
            "summary_text": (
                f"Career Readiness Index: {readiness_index}/100. "
                f"Current ATS Score is {current_ats}% ({'+' if score_diff >= 0 else ''}{score_diff}% change). "
                f"Total Applications: {app_metrics['total_applications']}."
            )
        }
