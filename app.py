import streamlit as st
import joblib
import requests
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Setup NLTK
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.download('wordnet', download_dir=nltk_data_path)
    nltk.download('omw-1.4', download_dir=nltk_data_path)

# --- Function to load Joblib model from Google Drive
@st.cache_data
def load_joblib_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(open("/tmp/tmp_model.pkl", "wb").write(response.content))
    else:
        st.error("❌ Failed to download model or vectorizer from Google Drive.")
        return None

# --- Replace these with your actual Google Drive file IDs
model_file_id = "1Tskp7Q0SNw2jIiJsKIA_q71HT7sOgugg"
vectorizer_file_id = "1hdtfuvnZDz9125eC_VVugdTz-QXz6f_J"

# --- Load model and vectorizer
model = load_joblib_from_drive(model_file_id)
vectorizer = load_joblib_from_drive(vectorizer_file_id)

# --- Preprocessing function
def preprocess_input(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- UI
st.title("🕵️ Fake News Detector")
st.write("Paste a news article below, or try an example. The app will predict whether it's **real** or **fake**.")

sample_news = {
    "🔴 Fake News: Hillary sold weapons to ISIS": """FBI Director James Comey held a press conference today confirming that Hillary Clinton sold weapons to the Islamic State during her tenure as Secretary of State.""",
    "🟢 Real News: Pentagon accepts transgender recruits": """The Pentagon said on Friday it will allow transgender individuals to enlist in the military beginning Jan. 1, after a federal judge ruled that the military must accept them.""",
    "🔘 (Custom Input)": ""
}
selection = st.selectbox("Choose a sample or write your own:", list(sample_news.keys()))
text_input = sample_news[selection]

if selection == "🔘 (Custom Input)":
    user_text_input = st.text_area("✍️ Enter news article text here:")
    if user_text_input.strip():
        text_input = user_text_input
else:
    st.text_area("✍️ Selected news article:", text_input, height=150)

st.info("""
📌 **Note:** For best results, provide the **full article text** or a detailed summary.  
Short inputs like headlines may result in less accurate predictions.
""")

if st.button("🔍 Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess_input(text_input)
        vect_text = vectorizer.transform([cleaned])

        prediction = model.predict(vect_text)[0]
        confidence = model.predict_proba(vect_text)[0]
        result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        prob = confidence[prediction] * 100

        st.subheader(f"{result}")
        st.write(f"**Confidence:** {prob:.2f}%")

st.markdown("""
---
<div style='text-align: center; font-size: 14px; color: grey;'>
    © 2025 Fake News Detector | Created with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
