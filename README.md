# Sentiment API

A FastAPI backend that predicts the sentiment of product reviews using two methods:

1. *Machine learning model*: TF-IDF + Logistic Regression, trained on Amazon food reviews
2. *VADER*: a rule-based sentiment analyzer from NLTK

This API is the backend for my [sentiment-dashboard](https://github.com/murugagowsi/sentiment-dashboard) project.

## Features

- POST /predict returns the ML prediction, its confidence, class probabilities, and the VADER prediction and scores
- Interactive API docs at /docs

## Tech Stack

- Python 3.13
- FastAPI, Uvicorn, Pydantic
- scikit-learn 1.6.1, joblib
- NLTK (VADER)

## Setup

```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload

## Example

Request:
```json
{"text": "This product is amazing, I love it!"}