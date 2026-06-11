# Instructional Correlation Metric
### AI-assisted Exam–Syllabus Alignment with Explainable Results

> A BITS Pilani Study Project (BITS F266) by **Rashi Tripathi** under the supervision of **Prof. Balasubramanian Malayappan**

Given a **course handout PDF** and an **exam paper PDF**, this tool automatically maps every exam question to its most relevant syllabus topic, computes a weighted relevance score, and generates a detailed PDF report — in under 45 seconds, versus ~90 minutes manually.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RashiTripathi-06/Instructional-Correlation-Metric-/blob/main/instructional_correlation_metric.ipynb)

---

## Motivation

Most faculty rely on personal judgment to check whether exam questions cover all syllabus topics fairly. Traditional keyword search is inadequate — a question about *"designing state machines"* should map to *"Finite State Machines"* even without shared keywords. This project automates that process using semantic NLP, with no domain-specific training required.

---

## Pipeline — 14 Steps

```
 Step 1: Topics          Step 2: Questions        Step 3: TF-IDF enrichment
  (from handout)          (from exam paper)         (lexical phase, pre-model)
       │                        │                          │
       └────────────────────────┴──────────────────────────┘
                                │
 Step 4: Enrich topics    Step 5: Enrich questions    Step 6: Abbreviations
  (TF-IDF + Wikipedia       (TF-IDF + Wikipedia         (dynamic abbrev dict
   + flan-t5 if avail.)      + flan-t5 if avail.)        from both PDFs)
       │
 Step 7: Load models      Step 7.5: Semantic enrichment  Step 8: Score matrix
  (SBERT bi-encoders        (bi-encoder topic↔question     (5 signals, weighted
   + NLI cross-encoder)      context bridging)              ensemble)
       │
 Step 9: Mapping          Step 10-11: Summary            Step 12: Visualisations
  (argmax per question)    & Insights                     (heatmap + coverage)
       │
 Step 13: Export          Step 14: PDF Report
  (JSON + CSV)             (10-section, reportlab)
```

### PDF Parsing (Steps 1–2)
- **Primary:** `pdfplumber` — spatial table extraction, preserves layout
- **Fallback 1:** `PyPDF2` — plain text extraction
- **Fallback 2:** `camelot-py` — Lattice mode (grid lines) then Stream mode (whitespace)

Input validation: MD5 hash check to catch identical uploads; regex signal scoring to detect if files are swapped.

### Topic Extraction
3-tier column detection strategy:
1. **Header keywords** — "topic", "syllabus", "coverage", "chapter"…
2. **Content scoring** — indicator words like "introduction to", "analysis of"
3. **Row fallback** — capitalized noun phrases above a minimum word count

Context enrichment runs in two phases:
- **Phase 1 (lexical):** TF-IDF overlap → find top similar questions per topic → extract noun phrases → append to topic context (runs before models load)
- **Phase 2 (semantic):** After bi-encoders load (Step 7.5), re-enriches topic contexts using cosine similarity to bridge exam↔syllabus vocabulary gap

Average topic string grows from ~8 words to ~22 words. The enriched string is used only for scoring; the original title appears in the report.

### Enrichment Pipeline (Steps 4–5)
Each topic and question goes through a cascade:
1. **TF-IDF self-enrichment** — top discriminating n-gram terms appended
2. **Wikipedia** — fetches a 300-character summary via REST API if a matching article is found
3. **`flan-t5-small` (optional)** — if available:
   - *Topics:* prompted for 5 domain-specific keywords
   - *Questions (Pass 1):* prompted for topic name + concept tested
   - *Questions (Pass 2):* prompted for the solving method/technique used (bridges implicit-topic questions)

LLM enrichment is optional — the pipeline falls back gracefully to TF-IDF + Wikipedia only.

### Ensemble Scoring (Step 8)

```
S(qi, tj) = 0.40 × S_mpnet  +  0.20 × S_bm25  +  0.15 × S_minilm
          + 0.10 × S_tfidf  +  0.10 × S_ce  +  0.05 × S_hint
```

| Signal | Model | Weight |
|---|---|---|
| Bi-encoder (MPNet) | `all-mpnet-base-v2` (768-d) | 40% |
| BM25 sparse retrieval | Custom BM25 (`k1=1.5`, `b=0.75`) | 20% |
| Bi-encoder (MiniLM) | `paraphrase-MiniLM-L6-v2` (384-d) | 15% |
| TF-IDF cosine | `sklearn TfidfVectorizer`, n-gram (1–3) | 10% |
| NLI cross-encoder | `nli-MiniLM2-L6-H768` (applied on ambiguous pairs only) | 10% |
| Hint n-grams | Self-derived phrase matching | 5% |

All bi-encoder vectors are L2-normalised before cosine computation.

**Primary assignment:** `f(qi) = argmax S(qi, tj)`

**Ambiguity flag:** if top-1 vs top-2 score gap < 0.04, question is annotated `*` and capped at MEDIUM confidence.

### Confidence Tiers (Dynamic, Step 8)

Thresholds are set dynamically from the actual score distribution of each run:

| Band | Threshold |
|---|---|
|  HIGH | ≥ 75th percentile of top scores (floor: 0.35) |
|  MEDIUM | ≥ 50th percentile of top scores (floor: 0.25) |
|  LOW | ≥ 25th percentile × 0.5 (floor: 0.08) |

### Output (Steps 13–14)

| Output | Tool | Contents |
|---|---|---|
| `Exam_Coverage_Report.pdf` | `reportlab` | 10-section analysis report |
| `report_fig1–5.png` | `matplotlib` + `seaborn` | Heatmap, bar charts, pie, score timeline |
| `results_v34.json` + `results_v34.csv` | `pandas` | Machine-readable question–topic mappings |

---

## Report Sections

1. Executive Summary — coverage rate, confidence breakdown
2. Topic Coverage Overview — primary / secondary-only / uncovered
3. Marks Distribution per Topic
4. Relevance Heatmap (questions × topics matrix)
5. Confidence & Marks Distribution
6. Question-level Marks & Relevance Scores
7. Top-6 Topics Deep Dive
8. Detailed Question–Topic Mapping Table
9. Coverage Gaps & Study Recommendations
10. Methodology

---

## Results (Digital Design Course)

Tested on a 40-lecture, 13-topic syllabus against a 120-mark exam (39 question units).

| Metric | Value | Assessment |
|---|---|---|
| Topic Extraction Accuracy | 100% | Excellent |
| Question Extraction Completeness | 97.4% (38/39) | Excellent |
| Marks Attribution Accuracy | 100% | Excellent |
| Primary Mapping Precision | 87.2% (34/39) | Good |
| Top-2 Mapping Recall | 94.9% (37/39) | Excellent |
| Primary Syllabus Coverage | 92.3% (12/13) | Good |
| Over-Emphasis Detection Accuracy | 100% | Excellent |
| Runtime | ~45 seconds | vs ~90 min manually |

---

## Requirements

- Python 3.8+
- Google Colab (recommended)

**Key libraries:**
```
sentence-transformers   transformers      huggingface_hub
rank-bm25               pdfplumber        camelot-py
PyPDF2                  scikit-learn      numpy
pandas                  matplotlib        seaborn
reportlab               requests
```

---

## Usage

1. Click **Open in Colab** above
2. Run **Cell 1** (install dependencies) → restart runtime when prompted
3. Run **Cell 2** (main pipeline) — upload your handout PDF and exam PDF when prompted
4. Run **Cell 3** (report generation) — PDF auto-downloads

**Inputs:** Course handout PDF (syllabus/topics table) + Exam paper PDF (questions and marks)

**Outputs:** `Exam_Coverage_Report.pdf` · `report_fig1–5.png` · `results_v34.json` · `results_v34.csv`

---

## Known Limitations

- **Image-embedded questions** (circuit diagrams, K-maps, timing diagrams) receive LOW confidence — the system processes text only
- **Single-topic assignment** — questions spanning multiple topics are flagged as ambiguous but cannot be fractionally allocated across topics
- **Domain vocabulary gap** — SBERT models are not fine-tuned for electronics/hardware terminology, causing occasional mismatches on highly technical terms
- **Cognitive difficulty** is not assessed — marks weight and cognitive demand are treated independently
- **Non-standard formats** — BITS Pilani does not use a uniform handout/exam format; parsing may need adjustment for other institutions

---

## Project Structure

```
Instructional-Correlation-Metric/
├── instructional_correlation_metric.ipynb   # Main Colab notebook
├── README.md
└── .gitignore
```

---

## Academic Context

- **Institution:** BITS Pilani, Hyderabad Campus
- **Course:** BITS F266 — Study Project
- **Supervisor:** Prof. Balasubramanian Malayappan
- **Submitted:** March 2026

---

## Author

**Rashi Tripathi** (2024A3PS0412H) — [GitHub](https://github.com/RashiTripathi-06)
