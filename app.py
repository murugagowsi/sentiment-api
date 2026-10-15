from fastapi import FastAPI
from pydantic import BaseModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk, joblib

nltk.download("vader_lexicon", quiet=True)

app = FastAPI(title="Sentiment API")
sia = SentimentIntensityAnalyzer()
model = joblib.load("model.pkl")
vec = joblib.load("vectorizer.pkl")

class Review(BaseModel):
    text: str

def vader_label(c: float) -> str:
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"

@app.get("/")
def home():
    return {"message": "Sentiment API is running"}

@app.post("/predict")
def predict(review: Review):
    v = sia.polarity_scores(review.text)
    X = vec.transform([review.text])
    probs = model.predict_proba(X)[0]
    classes = list(model.classes_)
    return {
        "text": review.text,
        "ml_prediction": model.predict(X)[0],
        "ml_confidence": round(float(max(probs)), 3),
        "ml_probabilities": {c: round(float(p), 3) for c, p in zip(classes, probs)},
        "vader_prediction": vader_label(v["compound"]),
        "vader_scores": v,
    }