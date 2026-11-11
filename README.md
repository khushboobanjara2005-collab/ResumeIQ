# ResumeIQ — Rule-Based NLP Resume Keyword Matcher

ResumeIQ is a local, session-based Streamlit application that extracts keywords
from a resume and compares them with a job description. It is a rule-based NLP
keyword matcher using classical AI techniques—not a generative AI system. It
uses tokenization, stopword removal, TF-IDF, and cosine similarity, with no paid
or external AI services.

## Features

- Analyze pasted resume text or upload a `.txt` or text-based `.pdf` file.
- Normalize and tokenize text, remove English stopwords, and rank keywords with
  TF-IDF.
- Detect useful multiword phrases such as Machine Learning, Data Analysis, Web
  Development, Artificial Intelligence, and Database Management.
- Compare resume and job-description keywords, with a separate cosine similarity
  metric.
- Compare multiple `.txt` or `.pdf` resumes with one job description.
- Save comparison snapshots in a History page and manage a persistent shortlist.
- Review recent activity and download a CSV results report.
- Keep resume and job-description text in memory; SQLite stores comparison
  metrics, keyword lists, file names, fingerprints, and shortlist selections.

## Technologies

Python, Streamlit, NLTK, scikit-learn, Pandas, Matplotlib, and PyMuPDF.

## Installation and launch (Windows PowerShell)

Run these commands from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The application is available at the local URL printed by Streamlit. To run it
without activating the environment, use:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## How the NLP workflow works

1. Input is lowercased, punctuation and whitespace are normalized, and NLTK's
   regular-expression tokenizer splits it into tokens. The tokenizer does not
   require downloaded NLTK tokenizer datasets.
2. English stopwords are removed. If the NLTK stopwords corpus is not installed,
   the application falls back to scikit-learn's local English stopword list
   without a setup step.
3. Exact occurrences of known skills and phrases are retained first. TF-IDF
   ranks additional terms across sentence-like segments to fill the remaining
   keyword slots.
4. For a job match, keyword coverage is
   `matched extracted job keywords / total extracted job keywords * 100`.
   Coverage is zero when no job keywords are extracted. TF-IDF cosine similarity
   is a distinct metric and is not an ATS score.

## How aliases and the skills file work

The app reads skill-name variants from `data/aliases.json` and applies them to
both resume and job-description text before extracting and matching keywords.
For example, `reactjs` and `react.js` become `react`, while `k8s` becomes
`kubernetes`. Alias matching uses whole-word boundaries, so `ml` does not match
inside `html`.

The curated vocabulary is in `data/skills.json`, grouped into programming
languages, web, databases, cloud/devops, data/AI, tools, and soft skills.
Detected entries from this file are listed before terms ranked by TF-IDF.
Generic job wording is kept separately in `data/stopwords_extra.json`, making
the vocabularies easy to inspect and update without changing the extraction
logic.

## Evaluation results

Run the evaluation from the project folder with
`python .\evaluation\evaluate.py`. It compares ten short fictional resume/job
pairs against manually written `expected_matched` labels and updates
`evaluation/results.md`.

Precision, recall, and F1 compare extracted matched keywords with the
hand-written `expected_matched` labels. Results describe this small fictional
sample only.

| Pair | Role | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| backend-python | Backend Python Developer | 1.000 | 0.833 | 0.909 |
| data-analyst | Data Analyst | 1.000 | 1.000 | 1.000 |
| frontend-engineer | Frontend Engineer | 1.000 | 1.000 | 1.000 |
| mobile-developer | Mobile Developer | 1.000 | 1.000 | 1.000 |
| cloud-engineer | Cloud Engineer | 1.000 | 1.000 | 1.000 |
| customer-support | Customer Support Associate | 1.000 | 1.000 | 1.000 |
| marketing-coordinator | Marketing Coordinator | 1.000 | 1.000 | 1.000 |
| java-engineer | Java Engineer | 1.000 | 1.000 | 1.000 |
| machine-learning-intern | Machine Learning Intern | 1.000 | 1.000 | 1.000 |
| hr-assistant | Human Resources Assistant | 1.000 | 1.000 | 1.000 |
| **Average** | **10 pairs** | **1.000** | **0.983** | **0.991** |

## Comparison history and privacy

Saved comparison snapshots are stored locally in
`data/resumeiq_history.sqlite3`. Set the `RESUMEIQ_DATABASE` environment variable
before launching Streamlit to use a different SQLite file. History does not
store resume or job-description contents; deleting a history entry also removes
its shortlist selection.

## Limitations

This is a simple keyword-based educational tool. Its alias list only covers
known, explicitly configured variants; it does not understand context,
unlisted synonyms, or whether a resume claim is accurate. It cannot assess
proficiency or predict hiring success, and its keyword coverage is not an ATS
score. The evaluation uses ten short fictional examples with labels written by
hand, so the results are not a measure of real-world performance. PDF
extraction works best with selectable text; scanned PDFs require OCR, which is
not included. Uploads are limited to 5 MB and PDFs to 10 pages. Analysis text
remains session-only; comparison metadata is saved locally in SQLite and is not
encrypted by the application.
