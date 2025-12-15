# 🧠 Three-Level Depression Classification in Banglish Social Media Posts Using NLP

![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge) ![NLP](https://img.shields.io/badge/Tech-NLP-orange?style=for-the-badge) ![ML](https://img.shields.io/badge/ML%2FDL-Models-purple?style=for-the-badge)

---

## 🌟 Project Overview

This project implements a **three-level depression classification system** for Banglish social media posts using **Natural Language Processing (NLP)**.
It is designed to automatically detect depression signals in **Banglish (Bangla-English mixed)** text, providing valuable insights into mental health trends on social media.

**Models were trained in Google Colab** using Python, TensorFlow/Keras, and scikit-learn, ensuring scalability and reproducibility.

---

## 📌 Features

* Handles **Banglish text preprocessing**, including emojis, stopwords, and special characters.
* Implements multiple models:

  * Logistic Regression
  * Random Forest
  * Support Vector Machine (SVM)
  * ANN / MLP
  * LSTM
* Computes performance metrics: **Accuracy, Precision, Recall, F1-Score**
* Provides **model comparison and ranking** using charts
* Designed for **academic and research purposes** in mental health monitoring

---

## 📊 Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Random Forest       | 95.92%   | 95.94%    | 95.92% | 95.92%   |
| Logistic Regression | 93.96%   | 93.96%    | 93.96% | 93.96%   |
| LSTM                | 87.76%   | 88.44%    | 87.76% | 87.64%   |
| ANN MLP             | 81.93%   | 81.90%    | 81.93% | 81.86%   |
| SVM                 | 85.35%   | 85.85%    | 85.35% | 85.53%   |

**💡 Insight:** Random Forest achieved the **highest F1-score**, making it the most reliable model for this dataset.

---

## 🛠 Technology Stack

* **Programming Language:** Python 3.12
* **Libraries:**

  * NLP: `nltk`, `emoji`, `gensim`
  * ML/DL: `scikit-learn`, `tensorflow`, `keras`
  * Visualization: `matplotlib`, `seaborn`, `streamlit`
* **Environment:** Google Colab

---

## 📝 Methodology

1. **Data Collection:** Banglish social media posts labeled into three depression levels.
2. **Data Preprocessing:** Emoji handling, stopwords removal, text normalization, tokenization, and vectorization.
3. **Model Training:** Models trained in **Google Colab** using ML and DL frameworks.
4. **Evaluation:** Metrics computed: Accuracy, Precision, Recall, F1-score.
5. **Comparison & Ranking:** Ranked models to identify the best-performing classifier.

---

## 📈 Performance Visualization

* **Metric Breakdown:** Individual model metrics.
* **Model Ranking:** Ranked by F1-Score.
* **Overall Comparison:** All models compared visually using bar charts.

*Placeholder GIFs for charts can be added here:*
![Metric Breakdown](https://via.placeholder.com/600x300.png?text=Metric+Breakdown+Chart)
![Model Ranking](https://via.placeholder.com/600x300.png?text=Model+Ranking+Chart)
![Overall Comparison](https://via.placeholder.com/600x300.png?text=Overall+Comparison+Chart)

---

## ⚡ Potential Applications

* Real-time monitoring of depression signals on social media.
* Mental health research for Bangla/Banglish speaking communities.
* Early warning system for high-risk depression posts.

---

## 🔮 Future Work

* Integrate with **real-time social media streams**.
* Explore **transformer-based NLP models** (BERT, BanglaBERT).
* Multi-lingual support including Bangla-only posts.
* Deployment as **interactive dashboards or APIs**.

---

## 📂 Project Structure

```
├── data/
│   └── banglish_posts.csv
├── notebooks/
│   └── model_training.ipynb  
├── model_metrics.json        
├── app.py                    
└── README.md
```

---

## 👩‍💻 Author

**Pranta Das**
East Delta University




Do you want me to do that next?
