# VCT Valorant Match Predictor & Analytics 🏆

An interactive web analytics portal designed to predict professional esports match outcomes using Machine Learning. This project leverages an XGBoost model trained on historical Valorant Champions Tour (VCT) dataset spanning from 2021 to 2026. 

This project was built as part of a Machine Learning portfolio to demonstrate end-to-end data processing, model training, and web application deployment.

## 🚀 Features
* **Match Outcome Prediction:** Calculates the winning probability between two teams based on selected maps and agent compositions.
* **Agent Meta & Synergy Dashboard:** Provides visual analytics of agent pick rates and synergies across different professional maps.
* **Interactive UI:** A user-friendly web interface built with Streamlit for seamless data exploration.

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Web Framework:** Streamlit
* **Model Serialization:** Joblib

## 📂 Repository Structure
* `app.py`: The main Streamlit web application script.
* `data_prep.ipynb`: Jupyter Notebook containing the end-to-end pipeline for data cleaning, feature engineering, and model training.
* `saved_models/`: Directory containing the trained XGBoost model (`.pkl`) and necessary encoding assets.
* `assets/`: Directory for UI assets, map images, and agent icons used in the Streamlit app.

> **Note on Dataset:** The raw VCT CSV datasets used for training are excluded from this repository to comply with GitHub's 100MB file size limit. The complete data preprocessing and extraction logic is documented transparently in `data_prep.ipynb`.

## 💻 How to Run Locally
1. Clone this repository:
```bash
git clone https://github.com/FebrioDw/valorant-predictor.git
```
2. Navigate to the project directory:
```bash
cd valorant-predictor
```
3. Install the required dependencies:
```bash
pip install streamlit pandas numpy xgboost scikit-learn joblib
```
4. Run the Streamlit application:
```bash
streamlit run app.py
```

## ⚠️ Disclaimer
This analytical tool predicts statistical probabilities based purely on historical team compositions, map data, and agent synergy metrics. Actual FPS match outcomes are heavily influenced by unquantifiable factors such as individual mechanical skill, mid-game tactical adaptability, and player communication.
