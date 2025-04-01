import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_and_save():
    # Load data
    train = pd.read_csv('../Assignment2/data/train.csv')
    
    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train['text'])
    y_train = train['label']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Model and vectorizer saved successfully")

if __name__ == '__main__':
    train_and_save() 