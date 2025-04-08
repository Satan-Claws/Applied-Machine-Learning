import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model():
    try:
        # Load the model and vectorizer from local files
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def score(text: str, model, vectorizer, threshold: float) -> tuple[bool, float]:
    X = vectorizer.transform([text])
    propensity = model.predict_proba(X)[0][1]
    prediction = propensity > threshold
    return prediction, propensity 