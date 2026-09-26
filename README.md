# Amazon ML Challenge 2026 — Business Entity Resolution

An end-to-end **Machine Learning solution for the Amazon ML Challenge 2026**, focused on **Business Entity Resolution** across multiple noisy and independent data sources.

## 📌 Problem Statement

In large-scale commercial platforms, business information is collected from multiple independent sources. The same real-world business may appear differently across these sources due to variations in names, addresses, abbreviations, typos, missing information, and formatting.

The objective of this challenge is to identify which records from **Source 2 and Source 3** correspond to each business entity in **Source 1**, where Source 1 acts as the deduplicated reference source.

A Source 1 entity can have:

* No matching records
* One matching record
* Multiple matching records

The solution therefore requires an effective combination of **candidate generation, entity matching, feature engineering, and machine learning**.

---

## 🎯 Objective

Build an ML pipeline that can accurately match business records across three independent sources while minimizing incorrect entity merges.

The solution focuses on:

* Data preprocessing and cleaning
* Business name normalization
* Address normalization
* Candidate generation / blocking
* Similarity feature extraction
* Machine learning-based matching
* Validation and threshold optimization
* Final entity matching

---

## 📂 Dataset

The challenge provides business records from three sources:

| Source       | Description                         |
| ------------ | ----------------------------------- |
| **Source 1** | Deduplicated reference entities     |
| **Source 2** | Business records requiring matching |
| **Source 3** | Business records requiring matching |

Each record contains:

* `entity_id`
* `business_name`
* `business_address`
* `country`

The training dataset contains ground-truth matching labels, while the test dataset does not.

### Dataset Structure

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
│
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All files use **TSV format** and must be read using a tab separator.

```python
import pandas as pd

df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

---

## 🔍 Challenges in Entity Resolution

The dataset contains several types of noise and inconsistencies.

### Business Name Variations

Examples include:

* Abbreviations: `Corp` vs `Corporation`
* Legal suffixes: `Ltd` vs `Limited`
* Punctuation differences
* Typos
* Word-order changes
* DBA / trade names
* Transliteration differences

### Address Variations

Addresses may contain:

* Abbreviations such as `Rd` / `Road`
* Missing PIN codes or states
* Different formatting
* Reordered address components
* Landmark-based descriptions
* Transliteration variations

These variations make exact string matching unreliable.

---

## 🧠 Approach

Our pipeline follows a multi-stage entity-resolution architecture:

```text
Raw Data
   ↓
Data Cleaning & Normalization
   ↓
Feature Engineering
   ↓
Candidate Generation / Blocking
   ↓
Similarity Feature Extraction
   ↓
ML Matching Model
   ↓
Threshold Optimization
   ↓
Final Entity Matches
   ↓
TSV Submission
```

### 1. Data Preprocessing

The first stage cleans and standardizes business records.

Operations include:

* Handling missing values
* Lowercasing text
* Removing unnecessary punctuation
* Standardizing whitespace
* Normalizing common abbreviations
* Cleaning business names and addresses
* Preserving country information as an open-set string label

The pipeline does not assume that the country values are limited to the training countries because the test set may contain additional countries such as France.

---

### 2. Candidate Generation / Blocking

Comparing every Source 1 record against every Source 2 and Source 3 record would be computationally expensive.

Therefore, a **blocking / candidate-generation stage** is used to identify plausible matches before applying the final matching model.

Potential blocking signals include:

* Normalized business name
* Name tokens
* Address components
* Country
* Character-level similarity
* Shared informative tokens

The candidate-generation stage is important because it determines the maximum possible recall of the final system.

---

### 3. Similarity Features

For each candidate pair, similarity features can be generated from business names and addresses.

Examples include:

* Jaccard similarity
* Levenshtein similarity
* TF-IDF cosine similarity
* Token overlap
* Character n-gram similarity
* Address component similarity
* Country consistency

These features allow the model to distinguish genuine business matches from unrelated records.

---

### 4. Machine Learning Matching

The generated candidate-pair features are provided to a machine learning model.

The model predicts whether a candidate Source 2 / Source 3 record corresponds to the Source 1 entity.

The matching stage is designed to prioritize **precision**, since incorrectly merging two different businesses is more costly than missing a potential match.

---

### 5. Validation & Threshold Optimization

Since the test set has no publicly available ground truth, a validation split is created from the training data.

The matching threshold is tuned using the challenge evaluation metric:

**F₀.₅ Score**

```text
F₀.₅ = (1.25 × Precision × Recall)
       / (0.25 × Precision + Recall)
```

F₀.₅ gives greater importance to precision than recall.

Singleton entities are also important: correctly predicting that an entity has **no matches** receives full credit for that entity, while incorrectly assigning a match results in a penalty.

---

## 📊 Evaluation Metric

The solution is evaluated using a **macro-averaged F₀.₅ score** across Source 1 entities.

```text
                 Precision × Recall
F₀.₅ = 1.25 × ─────────────────────────
              0.25 × Precision + Recall
```

The metric emphasizes precision, making the reduction of false-positive matches an important part of the solution.

---

## 📤 Output

The final pipeline generates two files:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

### `matching_results.tsv`

Contains the final predicted matches:

```text
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812
S1-00002             S3-00004
S1-00003
```

Every Source 1 entity must have exactly one row. If there are no matches, the `matched_entity_ids` field remains empty.

### `candidate_pairs.tsv`

Contains the candidate records generated during the final blocking stage before the matching model makes its predictions.

Every final match must be present in this candidate set.

---

## 📁 Project Structure

```text
business-entity-resolution/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── src/
│   ├── preprocessing.py
│   ├── blocking.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── inference.py
│   └── utils.py
│
├── notebooks/
│   ├── EDA.ipynb
│   └── experiments.ipynb
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd business-entity-resolution
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### Train

```bash
python src/train.py
```

### Generate Predictions

```bash
python src/inference.py
```

### Validate Submission

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

The official validator checks the output format and ensures that the generated files satisfy the challenge requirements.

---

## ⚠️ Challenge Constraints

The solution follows the competition requirements:

* Only Source 2 and Source 3 IDs can be predicted as matches.
* Every Source 1 test entity must appear in the output.
* Duplicate IDs are not allowed.
* Final matches must be a subset of generated candidates.
* The model must comply with the specified **MIT/Apache 2.0 licensing requirement** and parameter limit.
* External databases, APIs, geocoding services, business-registration databases, and internet-based entity lookups are prohibited.

---

## 📈 Future Improvements

Potential improvements to the pipeline include:

* Advanced multilingual text normalization
* Better address component extraction
* Character n-gram embeddings
* Ensemble similarity models
* Improved blocking strategies
* Hard-negative mining
* Threshold calibration
* Model-based pairwise classification
* More robust handling of singleton entities

---

## 🏆 Challenge

**Amazon ML Challenge 2026**

**Problem:** Business Entity Resolution

The objective is to build an accurate and scalable system capable of linking noisy business records across independent data sources while maintaining high precision.

---

## 👥 Team

**Team:** `<ERROR101>`

**Members:**

* `<Seema Birajdar>`
* `<Harini Thirunagari>`
* `<Bhoomi Patil>`
* `<Madhura Mane>`

---

## 📄 License

This repository contains code developed for the Amazon ML Challenge 2026 and follows the licensing requirements applicable to the models and libraries used in the solution.
