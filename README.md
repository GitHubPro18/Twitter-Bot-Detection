
# Twitter Bot Detection

A deep learning-based project to detect spambots on Twitter using advanced NLP models such as Bi-LSTM, Bi-GRU, DistilBERT, DistilRoBERTa, and XLNet. This project was developed as part of my end-semester Deep Learning Lab course at Manipal Institute of Technology. It leverages the Cresci-2017 dataset to classify accounts as spambots or humans based purely on tweet content, avoiding reliance on user profiles or network structures.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architectures](#model-architectures)
- [Performance Metrics](#performance-metrics)
- [Deployment](#deployment)
- [Results](#results)

---

## Problem Statement

This project focuses on detecting spambots on Twitter by analyzing tweet content. It avoids reliance on handcrafted features or user profiles, providing an efficient and scalable solution to combat malicious online behavior.

---

## Dataset

The **Cresci-2017 dataset** is used, featuring:
- **3,474 human accounts** (~8M tweets)
- **1,455 spambot accounts** (~3M tweets)



---

## Exploratory Data Analysis

EDA revealed linguistic patterns distinguishing spambots from humans:
- Spambots frequently use exaggerated language and external links.
- Human tweets focus on personal interactions.

Word clouds and statistical summaries were used for insights.

---

## Preprocessing Pipeline

Steps include:
1. **Tokenization**: Using NLTK and model-specific tokenizers.
2. **Embedding**: Pre-trained GloVe embeddings for word vectors.
3. **Cleaning**: Removal of special characters, URLs, and standardization.
4. **Padding**: Fixed-length sequences for model compatibility.

---

## Model Architectures

The following models were implemented:
- **Bi-LSTM**: Captures long-term dependencies in sequences.
- **Bi-GRU**: Lightweight alternative to Bi-LSTM.
- **DistilBERT**: Efficient transformer with ~97% of BERT's accuracy.
- **DistilRoBERTa**: Robust contextual understanding with speed optimization.
- **XLNet**: Bidirectional context through autoregressive pretraining.

---

## Performance Metrics

Evaluation metrics include:
- **Precision, Recall, F1 Score**
- **Accuracy**
- **Matthews Correlation Coefficient (MCC)**

---

## Deployment

The final model was deployed using **Streamlit**, enabling real-time bot detection via a user-friendly web interface.

---

## Results

| Model           | Training Accuracy | Testing Accuracy | Precision | Recall | F1 Score |
|------------------|-------------------|------------------|-----------|--------|----------|
| Bi-LSTM          | 92.02%           | 92.22%           | 94.72%    | 89.02% | 91.78%   |
| Bi-GRU           | 91.52%           | 93.05%           | 94.97%    | 90.84% | 92.86%   |
| DistilBERT       | 98.18%           | 96.36%           | 98.57%    | 94.40% | 96.44%   |
| DistilRoBERTa    | 97.80%           | 96.34%           | 96.97%    | 95.74% | 96.35%   |
| XLNet            | 49.96%           | 50.00%           | N/A       | N/A    | N/A      |

---
