# AI Resume Matcher

> TF-IDF + cosine similarity pipeline that scores and ranks resumes against a job description — because hiring shouldn't require reading 200 PDFs manually.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?logo=scikitlearn)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-data-150458?logo=pandas)](https://pandas.pydata.org)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-4CAF50)](https://www.nltk.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets)

---

## What This Does

Takes a job description as input, vectorizes it alongside a dataset of real resumes using TF-IDF, and ranks candidates by cosine similarity score. Returns the top 10 matches with their scores.

The core idea: resumes and job descriptions are bags of domain-specific vocabulary. TF-IDF weights rare, meaningful terms higher than common ones — so "LangChain" in both the JD and resume scores more than "experience" appearing everywhere.

---

## How It Works

```
Load Dataset (HuggingFace → Pandas DataFrame)
    → Text Preprocessing
        - Lowercase
        - Remove punctuation / stopwords
        - Lemmatization (NLTK WordNetLemmatizer)
    → TF-IDF Vectorization (sklearn TfidfVectorizer)
        - Fit on full resume corpus
        - Transform job description + all resumes
    → Cosine Similarity
        - Score each resume against JD vector
    → Rank top-10 candidates
    → Visualize score distribution (Matplotlib)
```

---

## Dataset

**[Sachinkelenjaguri/Resume_dataset](https://huggingface.co/datasets/Sachinkelenjaguri/Resume_dataset)** from HuggingFace

Loaded directly via `datasets` library — no manual download needed. Contains labeled resume text across multiple job categories.

The `Faker` library is used to generate synthetic candidate names and metadata for demo purposes, keeping PII out of the pipeline.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data handling | Pandas, NumPy |
| Text preprocessing | NLTK (stopwords, lemmatization) |
| Vectorization | Scikit-learn TfidfVectorizer |
| Similarity scoring | Scikit-learn cosine_similarity |
| Dataset | HuggingFace `datasets` |
| Synthetic data | Faker |
| Visualization | Matplotlib |

---

## Setup

```bash
git clone https://github.com/jibz33on/AI-Resume-Matcher
cd AI-Resume-Matcher

pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet
```

---

## Usage

```python
from matcher import ResumeMatcher

matcher = ResumeMatcher()
matcher.load_dataset()  # pulls from HuggingFace

job_description = """
We're looking for an AI Engineer with experience in LangChain, RAG pipelines,
Python, and deploying LLM-based systems to production.
"""

top_matches = matcher.rank(job_description, top_k=10)
print(top_matches[['candidate_name', 'similarity_score', 'resume_preview']])
```

Output:

```
   candidate_name  similarity_score  resume_preview
0    Alex Chen          0.847        "5 years Python, built RAG system..."
1    Jordan Kim         0.812        "LangChain, FastAPI, OpenAI APIs..."
...
```

---

## Why TF-IDF (and When It's Not Enough)

TF-IDF works well for keyword-heavy matching — job descriptions and resumes tend to use consistent domain vocabulary. It's fast, interpretable, and needs no GPU.

The limitation: it's lexical, not semantic. "Retrieval-augmented generation" and "RAG" score as different terms. For production-grade matching, you'd layer in a bi-encoder (e.g., `sentence-transformers/all-MiniLM-L6-v2`) for semantic similarity on top of the TF-IDF shortlist. This repo is the foundation for that.

---

## Author

**Jibin Kunjumon** — AI Engineer  
[GitHub](https://github.com/jibz33on) · [LinkedIn](https://linkedin.com/in/jibin-kunjumon)
