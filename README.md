# Algerian Forest Fire Prediction

This project implements a complete end-to-end machine learning lifecycle to predict the Fire Weather Index (FWI) using meteorological data from two regions in Algeria. The project covers everything from data cleaning and exploratory data analysis (EDA) to model training, evaluation, and deployment as both a Flask and a Streamlit web application.

## 🚀 Project Overview

The primary goal is to build a regression model that accurately predicts the FWI, which is a crucial indicator of fire danger. The project demonstrates the practical steps involved in taking a raw dataset and turning it into a deployed, interactive ML application.

### Key Features:
- **Data Cleaning & Preprocessing:** Handles missing values, corrects data types, and cleans column names.
- **Exploratory Data Analysis (EDA):** Visualizes data distributions, correlations, and monthly fire trends.
- **Model Training:** Trains multiple regression models (Ridge, SVR, Decision Tree, Random Forest) to find the best performer.
- **Web Deployment:**
    - A premium, minimalist **Flask application** for predictions.
    - A modern, interactive **Streamlit dashboard** for real-time predictions and model selection.

---

## 📊 Dataset

The dataset contains 244 instances collected from two regions in Algeria: the Bejaia region and the Sidi Bel-abbes region, with 122 instances per region. The data spans from June 2012 to September 2012.

### Attribute Information:
1.  **Date**: (DD/MM/YYYY)
2.  **Temp**: Temperature in Celsius (22 to 42)
3.  **RH**: Relative Humidity in % (21 to 90)
4.  **Ws**: Wind speed in km/h (6 to 29)
5.  **Rain**: Total daily rain in mm (0 to 16.8)
6.  **FFMC**: Fine Fuel Moisture Code
7.  **DMC**: Duff Moisture Code
8.  **DC**: Drought Code
9.  **ISI**: Initial Spread Index
10. **BUI**: Buildup Index
11. **FWI**: Fire Weather Index (Target Variable)
12. **Classes**: Fire or Not Fire

---

## 📁 Project Structure

```
.
├── models/
│   ├── ridge.pkl
│   ├── scaler.pkl
│   ├── dt.pkl
│   ├── rf.pkl
│   └── svr.pkl
│
├── notebooks/
│   ├── 1.0-Data Ingestion.ipynb
│   └── 2.0-EDA And FE Algerian Forest Fires.ipynb
│
├── templates/
│   └── home.html
│
├── application.py      # Flask Application
├── app.py              # Streamlit Application
└── requirements.txt    # Project dependencies
```

---

## ⚙️ How to Run the Project Locally

### Prerequisites
- Python 3.8+
- `pip` package manager

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-folder-name>
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
Install all the required libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```
*(If you don't have a `requirements.txt` file, you can create one by running `pip freeze > requirements.txt` after installing the necessary libraries like Flask, Streamlit, scikit-learn, pandas, etc.)*

### 4. Run the Flask Web Application
This will start the premium black-and-white UI.
```bash
python application.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`**.

### 5. Run the Streamlit Dashboard
This will start the interactive dashboard with the fire animation.
```bash
streamlit run app.py
```
Open your browser and navigate to the local URL provided in the terminal (usually `http://localhost:8501`).

---

## 🛠️ Technologies Used

- **Data Manipulation & Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **ML Modeling:** Scikit-learn
- **Web Framework (Backend):** Flask
- **Web Framework (Dashboard):** Streamlit
- **Development Environment:** Jupyter Notebook,
