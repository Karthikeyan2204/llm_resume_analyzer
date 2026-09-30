"""
app1.py

Streamlit front-end for the AI Resume Analyzer.

Implements the "user-friendly interface" from the project brief: users
upload a resume (and optionally paste a job description), the app runs the
full pipeline (parse -> BERT classification -> Groq LLM feedback) and
displays the analyzed insights and feedback so job seekers can tailor their
resumes effectively.

Run with:
    streamlit run app1.py
"""
import os
import tempfile

import streamlit as st

from Resume_analyzer1 import ResumeAnalyzer

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="centered")

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and get instant AI feedback to help you land your dream job.")

with st.sidebar:
    st.header("Settings")
    groq_key_input = st.text_input(
        "Groq API key (optional)",
        type="password",
        help="Leave blank to use GROQ_API_KEY from .env1",
    )

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
job_description = st.text_area(
    "Paste a job description (optional)", height=120
)

if st.button("Analyze Resume", type="primary"):
    if not uploaded_file:
        st.warning("Please upload a resume file first.")
    else:
        with st.spinner("Analyzing your resume..."):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            result = None
            error_message = None
            try:
                analyzer = ResumeAnalyzer(groq_api_key=groq_key_input or None)
                result = analyzer.analyze_resume(tmp_path, job_description=job_description)
            except Exception as exc:
                error_message = str(exc)
            finally:
                os.remove(tmp_path)

        if error_message:
            st.error(f"Something went wrong: {error_message}")

        if result:
            st.divider()

            # Predicted role + confidence score
            st.subheader("📌 Predicted Job Category")
            st.info(
                f"**{result['predicted_category']}** — "
                f"{result['confidence'] * 100:.1f}% confidence"
            )

            # Confidence chart across all 25 categories
            st.subheader("📊 Confidence Score by Category")
            st.bar_chart(result["category_scores"])

            st.divider()
            st.subheader("💪 Strengths & Weaknesses")
            st.markdown(result["ai_feedback"])

            st.divider()
            st.subheader("📈 Missing Skills")
            st.markdown(result["skill_gap_analysis"])

            if result["job_match_analysis"]:
                st.divider()
                st.subheader("🎯 Job Match Analysis")
                st.markdown(result["job_match_analysis"])