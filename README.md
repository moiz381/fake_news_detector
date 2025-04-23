# fake_news_detector

📰 Fake News Detection System
📌 Overview
This project is a machine learning-based system that detects whether a news article is fake or real. It leverages Natural Language Processing (NLP) techniques and a Random Forest Classifier, packaged with an intuitive Streamlit web application for real-time user interaction.

🚀 Features
Real-time prediction of news authenticity

Streamlit web app for user interaction

TF-IDF vectorization for feature extraction

Model confidence score display

Sample news articles for quick testing

Clean UI with detailed prediction feedback

🧠 Machine Learning Pipeline
Data Preprocessing:

Removal of stopwords

Text normalization and lemmatization

Cleaning with regular expressions

Feature Extraction:

Used TF-IDF Vectorizer to convert text into numeric features

Model Training:

Trained a Random Forest classifier on labeled True.csv and Fake.csv datasets

Web App Interface:

Built using Streamlit

Users can input custom text or select from sample articles

Displays prediction and confidence score

🧪 Model Performance
Achieved 99.76% accuracy on the test set

Excellent performance in distinguishing real and fake news

Balanced classification with an almost perfect confusion matrix

⚙️ Technologies Used
Python

Scikit-learn

Pandas & NumPy

NLTK

TF-IDF Vectorizer

Random Forest Classifier

Streamlit (for the web UI)
