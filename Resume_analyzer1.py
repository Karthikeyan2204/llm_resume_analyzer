"""
resume_analyzer1.py

Ties everything together into the "personal resume advisor" described in
the project brief, using LangChain for the LLM layer:

  1. resume_parser1.ResumeParser  -- pulls raw text + contact/skills/education
     out of the uploaded resume (PDF / DOCX / TXT).
  2. bert_classifier1.BertResumeClassifier -- classifies the resume into a
     job category using BERT.
  3. langchain_groq.ChatGroq -- the LLM, wired up via LangChain Expression
     Language (LCEL) chains: `prompt | llm | StrOutputParser()`. Prompts
     come from prompts1.py.

This is the module app1.py (the Streamlit UI) imports and calls.
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

import Prompts
from Bert_classifier1 import BertResumeClassifier
from Resume_parser1 import ResumeParser

# Load variables from .env1 (falls back silently to plain .env / real env vars
# if .env1 is not present).
load_dotenv(".env1")
load_dotenv()  # also pick up a standard .env if present

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class ResumeAnalyzer:
    """High-level pipeline: parse -> classify (BERT) -> LangChain/Groq feedback."""

    def __init__(self, groq_api_key=None, groq_model=None, bert_model_dir="bert_model"):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
            raise ValueError(
                "No Groq API key found. Set GROQ_API_KEY in .env1, or pass "
                "groq_api_key=... when creating ResumeAnalyzer."
            )

        self.groq_model = groq_model or DEFAULT_GROQ_MODEL

        # ------------------------------------------------------------
        # LangChain LLM + chains (LCEL: prompt | llm | output_parser)
        # ------------------------------------------------------------
        self.llm = ChatGroq(
    model=self.groq_model,
    groq_api_key=self.groq_api_key,
    temperature=0.3,
    max_tokens=400,
)
        output_parser = StrOutputParser()

        self.review_chain = Prompts.RESUME_REVIEW_PROMPT | self.llm | output_parser
        self.skill_gap_chain = Prompts.SKILL_GAP_PROMPT | self.llm | output_parser
        self.job_match_chain = Prompts.JOB_MATCH_PROMPT | self.llm | output_parser

        # ------------------------------------------------------------
        # BERT classifier (fine-tuned model if present, else zero-shot)
        # ------------------------------------------------------------
        self.classifier = BertResumeClassifier(model_dir=bert_model_dir)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def analyze_resume(self, file_path, job_description=None):
        """Run the full pipeline on a resume file and return a results dict.

        Args:
            file_path: path to a .pdf, .docx, or .txt resume.
            job_description: optional plain-text job description. If
                provided, an extra job-match / tailoring section is
                generated via `job_match_chain`.

        Returns:
            dict with contact info, skills, education, predicted category,
            confidence, per-category scores, AI feedback, skill-gap
            analysis, optional job-match analysis, and the raw resume text.
        """
        # 1. Parse the resume
        parser = ResumeParser(file_path)
        summary = parser.get_summary()
        resume_text = summary["raw_text"]

        if not resume_text.strip():
            raise ValueError(
                "No text could be extracted from this resume. If it is a "
                "scanned/image-based PDF, try uploading a text-based PDF or DOCX."
            )

        # 2. Classify into a job category with BERT
        classification = self.classifier.predict(resume_text)

        # 3. General AI feedback on the resume (LangChain chain)
        ai_feedback = self.review_chain.invoke(
            {
                "resume_text": resume_text,
                "predicted_category": classification["category"],
                "confidence": Prompts.format_confidence(classification["confidence"]),
            }
        )

        # 4. Skill-gap analysis for the predicted category (LangChain chain)
        skill_gap_analysis = self.skill_gap_chain.invoke(
            {
                "predicted_category": classification["category"],
                "extracted_skills": Prompts.format_skills(summary["skills"]),
            }
        )

        # 5. Optional: tailor advice to a specific job description (LangChain chain)
        job_match_analysis = None
        if job_description and job_description.strip():
            job_match_analysis = self.job_match_chain.invoke(
                {
                    "resume_text": resume_text,
                    "job_description": job_description,
                }
            )

        return {
            
            "skills": summary["skills"],
            "education": summary["education"],
            "word_count": summary["word_count"],
            "predicted_category": classification["category"],
            "confidence": classification["confidence"],
            "category_scores": classification["all_scores"],
            "classifier_mode": self.classifier.mode,
            "ai_feedback": ai_feedback,
            "skill_gap_analysis": skill_gap_analysis,
            "job_match_analysis": job_match_analysis,
            
        }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python resume_analyzer1.py <path-to-resume> [path-to-job-description.txt]")
    else:
        resume_path = sys.argv[1]
        jd_text = None
        if len(sys.argv) > 2:
            with open(sys.argv[2], "r", encoding="utf-8") as f:
                jd_text = f.read()

        analyzer = ResumeAnalyzer()
        result = analyzer.analyze_resume(resume_path, job_description=jd_text)

        # Trim raw_text for console readability
        printable = dict(result)
        printable["raw_text"] = printable["raw_text"][:500] + "..."
        print(json.dumps(printable, indent=2))