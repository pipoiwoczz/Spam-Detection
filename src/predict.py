import joblib
from preprocess import clean_text

clf = joblib.load('model/spam_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

def predict_spam(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    prediction = clf.predict(vect)
    return "Spam" if prediction[0] == 1 else "Not Spam"
