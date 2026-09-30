**README.md**

```markdown
# 📄 AI Resume Analyzer

An AI-powered resume analyzer that classifies your resume into a job category using BERT and provides short, actionable feedback using a LangChain + Groq LLM pipeline.

---

## 🚀 Features

- 📂 Supports PDF, DOCX, and TXT resume formats
- 🤖 BERT-based job category classification (25 categories)
- 💪 Short Strengths & Weaknesses analysis
- 📈 Top 5 missing skills for your predicted role
- 🎯 Optional Job Description match analysis
- 📊 Confidence score chart across all job categories
- ⚡ Fast responses via Groq API (Llama 3.3 70B)

---

## 🗂️ Project Structure

```
resume-organiser/
├── .env1                  # API key and model config
├── app1.py                # Streamlit UI
├── prompts1.py            # LangChain prompt templates
├── resume_analyzer1.py    # LangChain LCEL chains orchestrator
├── resume_parser1.py      # Resume text + info extractor
├── bert_classifier1.py    # BERT job category classifier
└── requirements.txt       # Dependencies
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your Groq API key

Edit `.env1`:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Get a free key at [https://console.groq.com/keys](https://console.groq.com/keys)

### 3. Run the app

```bash
streamlit run app1.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 How It Works

```
Upload Resume (PDF / DOCX / TXT)
        ↓
resume_parser1.py  →  Extract raw text
        ↓
bert_classifier1.py  →  Predict job category (BERT)
        ↓
LangChain LCEL Chains (prompts1.py + ChatGroq)
   ├── review_chain       →  Strengths & Weaknesses
   ├── skill_gap_chain    →  Top 5 Missing Skills
   └── job_match_chain    →  Job Match (if JD provided)
        ↓
Streamlit UI  →  Display results
```

---

## 📊 Output

| Section | What You Get |
|---|---|
| Predicted Category | Job role + confidence % |
| Confidence Chart | Bar chart across 25 categories |
| Strengths & Weaknesses | 3 bullets each, concise |
| Missing Skills | Top 5 skills for your role |
| Job Match *(optional)* | Score, gaps, key suggestion |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Streamlit | Web UI |
| LangChain + langchain-groq | LLM orchestration (LCEL chains) |
| Groq API (Llama 3.3 70B) | LLM for feedback generation |
| BERT (bert-base-uncased) | Resume classification |
| pdfplumber | PDF text extraction |
| python-docx | DOCX text extraction |
| python-dotenv | API key management |

---

## 🔧 Optional: Fine-tune BERT

To improve classification accuracy, fine-tune BERT on a labelled resume dataset (e.g. [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)):

```python
from bert_classifier1 import BertResumeClassifier

classifier = BertResumeClassifier()
classifier.fine_tune(
    csv_path="UpdatedResumeDataSet.csv",
    text_column="Resume",
    label_column="Category",
    output_dir="bert_model",
    epochs=3,
)
```

After training, the model is auto-detected and used on the next run.

---

## 📝 Notes

- Scanned / image-only PDFs won't work — use text-based PDFs or DOCX
- Without fine-tuning, BERT runs in zero-shot mode (embedding similarity)
- The Groq API key can also be entered directly in the Streamlit sidebar

---

## 👤 Author

**Karthikeyan E**
B.Tech Artificial Intelligence and Data Science — 2026
```
