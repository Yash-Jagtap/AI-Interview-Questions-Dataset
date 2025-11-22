# AI-Interview-Questions-Dataset
Curated dataset of AI/ML, Data Science, System Design, and DSA interview questions for fine-tuning LLMs and interview preparation

## Overview
This repository contains a curated collection of interview questions across multiple technical domains:
- **AI/ML**: Machine Learning algorithms, deep learning, NLP, computer vision
- **Data Science**: Statistics, data analysis, feature engineering, model evaluation
- **System Design**: Scalability, distributed systems, architecture patterns
- **DSA**: Data structures and algorithms fundamentals

## Purpose
- Fine-tune open-source LLMs for interview preparation assistance
- Practice and rehearse technical interview questions
- Build an intelligent interview preparation system
- Showcase ML engineering capabilities

## Dataset Format
The dataset is available in parquet format:
- `train-00000-of-00001.parquet` - Training dataset
- `eval-00000-of-00001.parquet` - Evaluation dataset

## Source
Dataset sourced from HuggingFace: [K-areem/AI-Interview-Questions](https://huggingface.co/datasets/K-areem/AI-Interview-Questions)

## Usage
```python
import pandas as pd

# Load the dataset
train_df = pd.read_parquet('train-00000-of-00001.parquet')
eval_df = pd.read_parquet('eval-00000-of-00001.parquet')

# Explore the data
print(train_df.head())
```

## Goals
1. Fine-tune an open-source LLM on this dataset
2. Create an interview preparation assistant
3. Demonstrate proficiency in GenAI and model fine-tuning
4. Build a portfolio project for SDE2/GenAI roles

## Tech Stack
- Python
- Pandas/PyArrow for data processing
- Transformers/Ollama for model fine-tuning
- RAG systems for enhanced responses

## License
MIT License - See LICENSE file for details

## Author
Yash Jagtap - Software Developer with 2 years of experience
- Working on backend and GenAI projects
- Building towards SDE2-level roles
- Focus: RAG systems, MCP integration, LLM applications
