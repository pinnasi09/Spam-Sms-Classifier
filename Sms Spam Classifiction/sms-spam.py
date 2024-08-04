import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import tkinter as tk
from tkinter import scrolledtext, messagebox

# Load and preprocess data
def load_and_preprocess_data():
    data = pd.read_csv('spam.csv', encoding='latin1')  # Changed encoding
    X = data['message']
    y = data['label']
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(X)
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, count_vect, tfidf_transformer

# Train and save the model
def train_and_save_model(X_train, y_train, count_vect, tfidf_transformer):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    joblib.dump(model, 'spam_classifier_model.pkl')
    joblib.dump(count_vect, 'count_vectorizer.pkl')
    joblib.dump(tfidf_transformer, 'tfidf_transformer.pkl')

# Evaluate the model
def evaluate_model(X_test, y_test):
    model = joblib.load('spam_classifier_model.pkl')
    count_vect = joblib.load('count_vectorizer.pkl')
    tfidf_transformer = joblib.load('tfidf_transformer.pkl')
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Load model and vectorizer
def load_model_and_vectorizer():
    model = joblib.load('spam_classifier_model.pkl')
    count_vect = joblib.load('count_vectorizer.pkl')
    tfidf_transformer = joblib.load('tfidf_transformer.pkl')
    return model, count_vect, tfidf_transformer

# Predict SMS
def predict_sms(message, model, count_vect, tfidf_transformer):
    message_transformed = tfidf_transformer.transform(count_vect.transform([message]))
    prediction = model.predict(message_transformed)
    return prediction[0]

# GUI setup
def setup_gui():
    def on_predict():
        msg = entry.get("1.0", tk.END).strip()
        if not msg:
            messagebox.showwarning("Input Error", "Please enter a message.")
            return
        model, count_vect, tfidf_transformer = load_model_and_vectorizer()
        prediction = predict_sms(msg, model, count_vect, tfidf_transformer)
        result_var.set(f"Prediction: {prediction}")

    window = tk.Tk()
    window.title("SMS Spam Classifier")

    tk.Label(window, text="Enter SMS message:").pack(pady=5)
    entry = scrolledtext.ScrolledText(window, width=50, height=10)
    entry.pack(pady=5)
    
    tk.Button(window, text="Classify", command=on_predict).pack(pady=10)

    result_var = tk.StringVar()
    tk.Label(window, textvariable=result_var).pack(pady=5)
    
    window.mainloop()

# Main workflow
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, count_vect, tfidf_transformer = load_and_preprocess_data()
    train_and_save_model(X_train, y_train, count_vect, tfidf_transformer)
    evaluate_model(X_test, y_test)
    setup_gui()
