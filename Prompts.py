

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a resume advisor. Give short, concise, and direct feedback.
Use bullet points only. Maximum 4-5 bullets per section. No long explanations."""

RESUME_REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Resume text:
{resume_text}

Predicted job category: "{predicted_category}" (confidence: {confidence}%)

Give SHORT feedback with exactly:
- **Strengths** (3 bullets max)
- **Weaknesses** (3 bullets max)

Keep each bullet under 15 words. Be direct and specific."""),
])

SKILL_GAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Job category: "{predicted_category}"
Skills already in resume: {extracted_skills}

List only the TOP 5 missing skills for this role.
One line per skill. No explanations, just the skill name and why it matters in under 10 words."""),
])

JOB_MATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Resume:
{resume_text}

Job Description:
{job_description}

Give a SHORT analysis with:
- **Match Score**: X/100
- **Top 3 Matching Skills** (bullets)
- **Top 3 Gaps** (bullets)
- **Key Suggestion** (1 line)

Keep everything brief."""),
])

def format_confidence(confidence):
    pct = confidence * 100 if confidence <= 1 else confidence
    return round(pct, 1)

def format_skills(extracted_skills):
    return ", ".join(extracted_skills) if extracted_skills else "None detected"
