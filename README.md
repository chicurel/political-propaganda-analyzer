# Political Propaganda NLP Pipeline (Spanish Case)

## Overview

This repository contains a Python-based NLP pipeline for processing Spanish political propaganda content.

The pipeline performs:

1. OCR extraction from campaign material (using Mistral OCR)
2. Text cleaning and normalization
3. Sentence segmentation
4. Sentence translation if text is not in Spanish or Enlgish
5. Sentence-level classification using LLMs 

The objective of this project is to analyze political propaganda at the sentence level in a structured, modular, and reproducible way.

---

## Project Structure

```
│
├── cat_code_individual.ipynb # Main pipeline notebook (Spanish case)
├── data/ # Input propaganda material (images or PDFs)
├── outputs/ # Processed text and sentence outputs
└── README.md
```

---

## Pipeline Steps

### 1. OCR Extraction

- **Input:** Image or scanned political campaign material  
- **Tool:** Mistral OCR  
- **Output:** Raw extracted Spanish text  

The OCR step converts visual political material into machine-readable text for downstream NLP analysis.

---

### 2. Text Cleaning

The raw OCR text is processed to:

- Remove noise and artifacts
- Normalize whitespace
- Clean encoding issues
- Standardize punctuation

This step improves text quality and ensures consistent sentence segmentation.

---

### 3. Sentence Segmentation

The cleaned text is split into individual sentences.

This enables:

- Sentence-level analysis
- Fine-grained discourse study
- Future classification of propaganda techniques

---

### 4. Sentence Classification

Functionality:

- Use a Large Language Model (GPT-based model)
- Classify each sentence according to propaganda techniques
- Store structured outputs (CSV)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

If running via Jupyter Notebook:

```bash
pip install notebook
jupyter notebook
```

## Requirements

- Python 3.9+
- Mistral OCR access
- NLP libraries (e.g., spaCy, nltk, or similar)
- OpenAI API access
