import pytest
import os
import time
import requests
from score import load_model, score

print("Starting test setup...")

OBVIOUS_SPAM = "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
OBVIOUS_HAM = "Hi, how are you? Just checking in to see if you're free for lunch tomorrow."

@pytest.fixture
def model_fixture():
    print("Loading model...")
    try:
        model, vectorizer = load_model()
        print("Model loaded successfully")
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def test_score(model_fixture):
    print("Running score test...")
    model, vectorizer = model_fixture
    prediction, propensity = score("Test message", model, vectorizer, 0.5)
    assert 0 <= propensity <= 1
    
    prediction_zero, _ = score("Test message", model, vectorizer, 0.0)
    prediction_one, _ = score("Test message", model, vectorizer, 1.0)
    assert prediction_zero == True
    assert prediction_one == False
    
    spam_pred, spam_prop = score(OBVIOUS_SPAM, model, vectorizer, 0.5)
    ham_pred, ham_prop = score(OBVIOUS_HAM, model, vectorizer, 0.5)
    assert spam_pred == True
    assert ham_pred == False
    assert spam_prop > 0.5
    assert ham_prop < 0.5
    print("Score test completed")

def test_flask():
    print("Starting Flask test...")
    os.system("python app.py &")
    time.sleep(2)
    try:
        response = requests.post(
            "http://localhost:5001/score",
            json={"text": "Test message", "threshold": 0.5}
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "propensity" in data
    finally:
        os.system("pkill -f 'python app.py'")
    print("Flask test completed") 