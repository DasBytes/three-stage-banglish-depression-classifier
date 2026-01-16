import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
import pickle
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.sparse import hstack

# --- Page Config ---
st.set_page_config(page_title="Banglish Depression Detection", layout="wide")

# --- Helper Functions ---
def clean_text(text):
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\u0980-\u09FF ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def count_lexicon(text):
    pos = ['valo', 'bhalo', 'happy', 'alhamdulillah', 'nice', 'moja', 'sundor']
    neg = ['kharap', 'na', 'tired', 'stress', 'sad', 'suicide', 'khub', 'dukho', 'niras']
    p_count = sum(text.count(w) for w in pos)
    n_count = sum(text.count(w) for w in neg)
    return len(text.split()), p_count, n_count

# --- Load Models (Cached for Performance) ---
@st.cache_resource
def load_all_models():
    # 1. Logistic Regression Components
    with open('models/logistic_model.pkl', 'rb') as f: lr_model = pickle.load(f)
    with open('models/logistic_tfidf_vectorizer.pkl', 'rb') as f: lr_vec = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    
    # 2. Random Forest Components
    with open('models/rf_model.pkl', 'rb') as f: rf_model = pickle.load(f)
    with open('models/rf_tfidf_vectorizer.pkl', 'rb') as f: rf_vec = pickle.load(f)
    
    # 3. MLP Components
    mlp_model = tf.keras.models.load_model('models/mlp_model.h5')
    with open('models/mlp_label_encoder.pkl', 'rb') as f: mlp_le = pickle.load(f)
    
    # 4. LSTM Components
    lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
    with open('models/lstm_label_encoder.pkl', 'rb') as f: lstm_le = pickle.load(f)
    with open('models/word_index.json', 'r') as f: w_idx = json.load(f)
    
    return lr_model, lr_vec, scaler, rf_model, rf_vec, mlp_model, mlp_le, lstm_model, lstm_le, w_idx

# Load everything
(lr_m, lr_v, sc, rf_m, rf_v, mlp_m, mlp_le, lstm_m, lstm_le, w_idx) = load_all_models()

# --- UI ---
st.title("🧠 Banglish Depression Detection")
st.markdown("Predict sentiment from Banglish (Bengali written in English script) using 4 different ML/DL approaches.")

model_choice = st.sidebar.selectbox(
    "Choose Model Architecture", 
    ("Logistic Regression", "Random Forest", "MLP (FastText)", "LSTM (Sequential)")
)

user_input = st.text_area("Enter Banglish text here:", placeholder="Amar khub kharap lagche...")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        
        if model_choice == "Logistic Regression":
            tfidf = lr_v.transform([cleaned])
            stats = sc.transform([count_lexicon(cleaned)])
            combined = hstack([tfidf, stats])
            pred = lr_m.predict(combined)[0]
            prob = np.max(lr_m.predict_proba(combined))

        elif model_choice == "Random Forest":
            vec = rf_v.transform([cleaned])
            pred = rf_m.predict(vec)[0]
            prob = np.max(rf_m.predict_proba(vec))

        elif model_choice == "MLP (FastText)":
            # Note: For permanent deploy, you'd usually pre-calculate or use a smaller embedding 
            # This logic assumes the MLP model input is the averaged vector.
            # (Requires loading FastText or a mapping file)
            st.info("MLP requires FastText vectors to be computed.")
            # Dummy logic placeholder - replace with your sentence_to_vec function
            pred, prob = "Depression", 0.85 

        elif model_choice == "LSTM (Sequential)":
            tokens = cleaned.split()
            seq = [w_idx.get(t, 0) for t in tokens]
            padded = pad_sequences([seq], maxlen=50) # use your saved max_len
            res = lstm_m.predict(padded)
            idx = np.argmax(res)
            pred = lstm_le.inverse_transform([idx])[0]
            prob = res[0][idx]

        # --- Display Results ---
        col1, col2 = st.columns(2)
        col1.metric("Predicted Category", pred)
        col2.metric("Confidence Score", f"{prob*100:.2f}%")
        
        if pred.lower() == "depression":
            st.error("The system detected signs of depression. Please reach out to someone you trust.")
        else:
            st.success("The system detected a non-depressive sentiment.")
