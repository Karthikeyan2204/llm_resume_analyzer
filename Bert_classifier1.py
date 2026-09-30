"""
bert_classifier1.py

BERT-based job-category classifier for resumes, as called for in the
project brief ("Implement classification techniques to categorize resumes
based on job-relevant skills and experience").

Two modes are supported:

1. "finetuned" -- if a fine-tuned BERT sequence-classification model is
   found at `model_dir` (e.g. produced by `fine_tune()` below, trained on a
   labelled resume dataset such as the Kaggle "Resume Dataset"), it is
   loaded with `AutoModelForSequenceClassification` and used directly.

2. "zero_shot" -- if no fine-tuned model is present, the classifier falls
   back to a zero-shot approach: it embeds the resume text and a short
   description of each job category with a base BERT model (mean-pooled
   token embeddings) and picks the category with the highest cosine
   similarity. This means the project works out-of-the-box even before
   you have a labelled training set.

Either way, `predict()` returns the same shape of result so
`resume_analyzer1.py` doesn't need to care which mode is active.
"""

import os

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


# 25 job categories based on the commonly-used "Resume Dataset" (Kaggle),
# matching the kind of "various job categories" mentioned in the project idea.
JOB_CATEGORIES = [
    "Advocate", "Arts", "Automation Testing", "Blockchain", "Business Analyst",
    "Civil Engineer", "Data Science", "Database", "DevOps Engineer",
    "DotNet Developer", "ETL Developer", "Electrical Engineering", "HR",
    "Hadoop", "Health and fitness", "Java Developer", "Mechanical Engineer",
    "Network Security Engineer", "Operations Manager", "PMO",
    "Python Developer", "SAP Developer", "Sales", "Testing", "Web Designing",
]


class BertResumeClassifier:
    """Classifies resume text into one of `categories` using BERT."""

    def __init__(self, model_dir="bert_model", categories=None, base_model_name="bert-base-uncased"):
        self.categories = categories or JOB_CATEGORIES
        self.model_dir = model_dir
        self.base_model_name = base_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_dir and os.path.isdir(model_dir) and os.listdir(model_dir):
            # A fine-tuned classification head exists -- use it directly.
            self.mode = "finetuned"
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device).eval()
        else:
            # No fine-tuned model yet -- fall back to zero-shot similarity.
            self.mode = "zero_shot"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = AutoModel.from_pretrained(base_model_name)
            self.model.to(self.device).eval()
            self._category_embeddings = self._embed_categories()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _embed_text(self, text):
        """Mean-pooled BERT embedding for `text` (used in zero-shot mode)."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled.squeeze(0).cpu().numpy()

    def _embed_categories(self):
        embeddings = {}
        for category in self.categories:
            description = (
                f"This resume belongs to the {category} job category. "
                f"It contains skills, tools, projects and experience relevant to {category}."
            )
            embeddings[category] = self._embed_text(description)
        return embeddings

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, resume_text):
        """Return {"category": str, "confidence": float, "all_scores": {cat: score}}."""
        if self.mode == "finetuned":
            return self._predict_finetuned(resume_text)
        return self._predict_zero_shot(resume_text)

    def _predict_finetuned(self, resume_text):
        inputs = self.tokenizer(
            resume_text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
        top_idx = int(np.argmax(probs))

        id2label = getattr(self.model.config, "id2label", None) or {}
        all_scores = {}
        for i, prob in enumerate(probs):
            label = id2label.get(i, id2label.get(str(i)))
            if label is None:
                label = self.categories[i] if i < len(self.categories) else str(i)
            all_scores[label] = float(prob)

        top_label = id2label.get(top_idx, id2label.get(str(top_idx)))
        if top_label is None:
            top_label = self.categories[top_idx] if top_idx < len(self.categories) else str(top_idx)

        return {
            "category": top_label,
            "confidence": float(probs[top_idx]),
            "all_scores": all_scores,
        }

    def _predict_zero_shot(self, resume_text):
        text_emb = self._embed_text(resume_text)

        raw_scores = {}
        for category, cat_emb in self._category_embeddings.items():
            cos_sim = float(
                np.dot(text_emb, cat_emb)
                / (np.linalg.norm(text_emb) * np.linalg.norm(cat_emb) + 1e-9)
            )
            raw_scores[category] = cos_sim

        # Sharpen the (fairly close) cosine similarities into a pseudo-probability
        # distribution so the UI can display a meaningful "confidence".
        values = np.array(list(raw_scores.values()))
        scaled = (values - values.max()) * 25  # temperature scaling
        exp_values = np.exp(scaled)
        probs = exp_values / exp_values.sum()

        all_scores = dict(zip(raw_scores.keys(), probs.tolist()))
        top_category = max(all_scores, key=all_scores.get)

        return {
            "category": top_category,
            "confidence": all_scores[top_category],
            "all_scores": all_scores,
        }

    # ------------------------------------------------------------------
    # Optional: fine-tune on a labelled dataset
    # ------------------------------------------------------------------
    def fine_tune(self, csv_path, text_column="Resume", label_column="Category",
                   output_dir="bert_model", epochs=3, batch_size=8, learning_rate=2e-5):
        """Fine-tune a fresh BERT classification head on a labelled CSV dataset.

        The CSV should have at least two columns: one with the raw resume
        text (`text_column`) and one with the job category label
        (`label_column`), e.g. the Kaggle "Resume Dataset"
        (https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset).

        After training, the model + tokenizer are saved to `output_dir`.
        Re-instantiate `BertResumeClassifier(model_dir=output_dir)`
        afterwards to use the fine-tuned model.
        """
        import pandas as pd
        from datasets import Dataset
        from sklearn.preprocessing import LabelEncoder
        from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

        df = pd.read_csv(csv_path)
        df = df[[text_column, label_column]].dropna()

        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(df[label_column])
        labels = sorted(set(df["label"]))
        id2label = {int(i): label_encoder.inverse_transform([i])[0] for i in labels}
        label2id = {v: k for k, v in id2label.items()}

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=len(id2label),
            id2label={k: str(v) for k, v in id2label.items()},
            label2id={str(k): v for k, v in label2id.items()},
        )

        dataset = Dataset.from_pandas(df[[text_column, "label"]])

        def tokenize(batch):
            return tokenizer(batch[text_column], truncation=True, max_length=512)

        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.train_test_split(test_size=0.1, seed=42)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=50,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
        )

        trainer.train()

        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Fine-tuned model saved to '{output_dir}'.")


if __name__ == "__main__":
    # Quick smoke test using the zero-shot fallback (no fine-tuned model needed)
    sample_resume = (
        "Experienced software engineer skilled in Python, Django, REST APIs, "
        "PostgreSQL, Docker and AWS. Built and deployed microservices and "
        "led a team of 3 developers."
    )
    classifier = BertResumeClassifier(model_dir="bert_model_does_not_exist")
    result = classifier.predict(sample_resume)
    print(f"Mode: {classifier.mode}")
    print(f"Predicted category: {result['category']} ({result['confidence'] * 100:.1f}%)")