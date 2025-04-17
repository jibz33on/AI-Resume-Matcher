
# 🧠 AI-Powered Resume Matcher using TF-IDF

This project is an end-to-end implementation of an **AI-powered resume matching system**, designed to help recruiters automatically rank candidate resumes based on how well they match a given job description.

## 🚀 Overview

The system uses **TF-IDF vectorization** and **cosine similarity** to compare resumes with a recruiter’s job description (JD). It outputs the **Top 10 most relevant candidates**, complete with match scores, names, and email addresses.

---

## 📁 Dataset

- Source: [Sachinkelenjaguri/Resume_dataset](https://huggingface.co/datasets/Sachinkelenjaguri/Resume_dataset)
- Format: Resumes labeled by job categories
- Cleaned and enriched with:
  - Resume ID
  - Fake Name and Email (via Faker)
  - Resume Length
  - Preprocessed Resume Text

---

## 🛠️ Features

✅ Load and explore real resume data  
✅ Preprocess resumes (cleaning, lemmatization, stopword removal)  
✅ Add fake metadata for realism  
✅ Vectorize resumes using **TF-IDF**  
✅ Input dynamic job descriptions  
✅ Match resumes to JD using **cosine similarity**  
✅ Return Top 10 candidates with match scores  
✅ Export results as CSV  
✅ Visualize scores with bar plots

---

## 📊 Example Output

```
| Resume ID | Candidate Name   | Email               | Match_Score (%) |
|-----------|------------------|---------------------|-----------------|
| R0182     | John Smith       | john@example.com    | 82.45           |
| R0451     | Jane Doe         | jane@example.com    | 78.23           |
```

---

## 🧪 Technologies Used

- Python 🐍
- Pandas, NumPy
- Matplotlib
- NLTK (Text cleaning)
- Scikit-learn (TF-IDF & cosine similarity)
- Faker (fake names & emails)
- Hugging Face Datasets

---

## 💡 How to Use

1. Clone the repo or open the `.ipynb` notebook in Colab
2. Install dependencies:
    ```
    pip install datasets faker nltk
    ```
3. Run all cells
4. Change the `job_description` input to test different roles
5. View the Top 10 ranked resumes and their match scores

---

## 📦 Future Enhancements

- Dynamic resume upload & comparison
- Streamlit / Django-based frontend
- Semantic search with BERT / Sentence Transformers
- Skill extraction and filtering

---

## 🤝 Credits

Project built by **Jibin Kunjumon** 👨‍💻  
Built as part of my AI learning journey, with assistance from AI tools.   
Dataset by [Sachinkelenjaguri on Hugging Face](https://huggingface.co/datasets/Sachinkelenjaguri/Resume_dataset)

---

## 📜 License

This project is for educational and portfolio purposes only.
