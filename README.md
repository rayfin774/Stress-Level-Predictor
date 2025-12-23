# Stress Level Predictor

A machine learning–based web application that predicts a user's **stress level (Low / Medium / High)** using lifestyle factors such as sleep hours, work hours, physical activity, and screen time.

---

## Key Features
- Stress level prediction using **Random Forest**
- Interactive web interface built with **Streamlit**
- Color-coded stress level output with basic insights

---

## Tech Stack
- Python, Pandas
- Scikit-learn (Random Forest)
- Streamlit

---

## How It Works
User inputs lifestyle factors, which are processed by a trained Random Forest classifier to predict stress level.

---

## How to Run
```bash
git clone https://github.com/shreeshtjagga/stress-level-predictor.git
cd stress-level-predictor
pip install -r requirements.txt
streamlit run app.py
