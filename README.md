📌 Project Overview

Social media platforms have become a primary outlet for expressing emotions. In Bangladesh, many users express their feelings using "Banglish" (Bengali written in English script). This project aims to develop a Natural Language Processing (NLP) system capable of detecting and classifying depression levels in Banglish social media posts.

The system analyzes text data to categorize posts into three distinct levels of mental health status, helping researchers and support organizations identify potential issues early.

🎯 Objectives

Build a Labeled Dataset: Collect and categorize Banglish social media posts.

Text Preprocessing: Clean and normalize informal Banglish text (handling mixed language, emojis, and slang).

Classification: Train a machine learning model to accurately predict depression levels.

📂 Dataset

The dataset consists of social media posts labeled into three categories:

No Depression: Casual conversations, general updates, or positive content.

Mild Depression: Expressions of sadness, loneliness, or mild distress.

Severe Depression: Content indicating deep hopelessness, self-harm, or suicidal ideation.

Size: Approximately 5,000+ labeled posts.

⚙️ System Workflow

The project follows a standard NLP pipeline:

Input: Raw Banglish text.

Preprocessing:

Lowercasing

URL & HTML tag removal

Punctuation & Emoji removal

Extra whitespace removal

Feature Extraction: Converting text into numerical vectors (e.g., TF-IDF).

Model Training: Training a classifier on the processed data.

Output: Predicted Class (No Depression / Mild / Severe).

🛠️ Technologies & Libraries

Language: Python

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

NLP: NLTK (Natural Language Toolkit)

Machine Learning: Scikit-Learn

🚀 How to Run

Clone the Repository or download the project files.

Upload the Dataset: Ensure Banglish depression dataset.csv is available in the working directory.

Run the Script:

If using Google Colab, upload the notebook and run the cells sequentially.

The script will ask you to upload the CSV file if it is not found.

Live Prediction:

After training is complete, a generic input box will appear.

Type any Banglish sentence to see the predicted depression level.
