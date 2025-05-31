# 📡 Outage Prediction & Forecasting App

This is a user-friendly **Streamlit web app** that predicts potential telecom network outages using historical KPI data from Excel files. It provides **trend analysis**, **correlation heatmaps**, **future forecasting**, and **site-level filtering**—all without requiring coding knowledge.

---

## 🚀 Features

- ✅ **Outage Prediction** using a trained machine learning model  
- 📈 **KPI Trend Visualization** (Hourly, Daily, Weekly, Monthly)  
- 🔥 **Correlation Heatmap** to identify KPI relationships  
- 🔮 **Forecast Congestion (%)** using Prophet  
- 🏷️ **Filter by SITE or CELL_CODE** to focus on specific locations

---

## 📂 How to Use

1. Clone the repo and install requirements:
   ```bash
   git clone https://github.com/naeemakhtartheai/outage-prediction-app.git
   cd outage-prediction-app
   pip install -r requirements.txt
Add your trained model and imputer files to the models/ directory.

Run the app:

streamlit run app.py
Upload your Excel file with KPI data.

📊 Required Excel Columns
Ensure your file contains at least:

FRAGMENT_DATE (timestamp)

KPI columns like CONGESTION (%), ERAB_Drop rate, etc.

Optional: SITE, CELL_CODE for site-level filtering

📦 Dependencies
streamlit

pandas

numpy

matplotlib

seaborn

prophet

joblib

Install all with:

pip install -r requirements.txt


📄 License
MIT License

Author
Naeem Akhtar


