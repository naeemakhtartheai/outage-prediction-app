# # # # import streamlit as st
# # # # import pandas as pd
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # # # import seaborn as sns
# # # # from prophet import Prophet
# # # # import joblib
# # # # from datetime import datetime

# # # # # Load model and imputer
# # # # model = joblib.load('models/outage_predictor.pkl')
# # # # imputer = joblib.load('models/imputer.pkl')

# # # # st.title("📡 Outage Prediction & Forecasting App")
# # # # st.write("Upload your KPI Excel file to predict and forecast outages.")

# # # # uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

# # # # if uploaded_file:
# # # #     # Load and combine data
# # # #     df = pd.read_excel(uploaded_file)

# # # #     # Preprocess
# # # #     df['FRAGMENT_DATE'] = pd.to_datetime(df['FRAGMENT_DATE'], errors='coerce', format='%d-%b-%y %H')
# # # #     df['DATE'] = df['FRAGMENT_DATE'].dt.date
# # # #     df['HOUR'] = df['FRAGMENT_DATE'].dt.hour
# # # #     df['MONTH'] = df['FRAGMENT_DATE'].dt.month_name()

# # # #     # Fill missing values
# # # #     df.replace(0, np.nan, inplace=True)
# # # #     df['CONGESTION (%)'] = df['CONGESTION (%)'].fillna(0)
# # # #     df['ERAB_Drop rate'] = df['ERAB_Drop rate'].fillna(0)

# # # #     # Predict Potential Outage
# # # #     features = ['CONGESTION (%)', 'DL_AVG_THRPT_PER_USER-Mbps', 'ERAB_Drop rate',
# # # #                 'CELL_THRPT-Mbps', 'ERAB Success Rate', 'Average LTE Connected Users']
# # # #     X_imputed = imputer.transform(df[features])
# # # #     df['Potential_Outage'] = model.predict(X_imputed)

# # # #     st.subheader("📊 Sample Data with Predictions")
# # # #     st.dataframe(df[['FRAGMENT_DATE'] + features + ['Potential_Outage']].head(20))

# # # #     # KPIs Plots
# # # #     st.subheader("📈 KPI Trend")
# # # #     kpi = st.selectbox("Select KPI to Plot", features)
# # # #     fig, ax = plt.subplots()
# # # #     df.groupby('HOUR')[kpi].mean().plot(kind='line', marker='o', ax=ax)
# # # #     ax.set_title(f"Hourly Trend of {kpi}")
# # # #     st.pyplot(fig)

# # # #     # Correlation
# # # #     st.subheader("📉 Correlation Heatmap")
# # # #     fig, ax = plt.subplots()
# # # #     sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=ax)
# # # #     st.pyplot(fig)

# # # #     # Forecasting Congestion
# # # #     st.subheader("🔮 Forecasting Congestion (%)")
# # # #     df_prophet = df[['FRAGMENT_DATE', 'CONGESTION (%)']].dropna()
# # # #     df_prophet.columns = ['ds', 'y']

# # # #     m = Prophet()
# # # #     m.fit(df_prophet)

# # # #     future = m.make_future_dataframe(periods=90)
# # # #     forecast = m.predict(future)

# # # #     fig1 = m.plot(forecast)
# # # #     st.pyplot(fig1)

# # # #     fig2 = m.plot_components(forecast)
# # # #     st.pyplot(fig2)

# # # #     st.success("✅ Outage Prediction & Forecast Completed!")
# # # # else:
# # # #     st.info("📂 Please upload an Excel file with network KPI data.")





# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # from prophet import Prophet
# # # import joblib

# # # # Title and upload
# # # st.title("📡 Outage Prediction & Forecasting App")
# # # st.write("Upload your KPI Excel file to predict and forecast outages.")

# # # uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

# # # if uploaded_file:
# # #     # Load and preprocess
# # #     df = pd.read_excel(uploaded_file)
# # #     df['FRAGMENT_DATE'] = pd.to_datetime(df['FRAGMENT_DATE'], errors='coerce', format='%d-%b-%y %H')

# # #     df['DATE'] = df['FRAGMENT_DATE'].dt.date
# # #     df['HOUR'] = df['FRAGMENT_DATE'].dt.hour
# # #     df['DAY'] = df['FRAGMENT_DATE'].dt.day_name()
# # #     df['WEEK'] = df['FRAGMENT_DATE'].dt.to_period('W').astype(str)
# # #     df['MONTH'] = df['FRAGMENT_DATE'].dt.month_name()

# # #     df.replace(0, np.nan, inplace=True)
# # #     df['CONGESTION (%)'] = df['CONGESTION (%)'].fillna(0)
# # #     df['ERAB_Drop rate'] = df['ERAB_Drop rate'].fillna(0)

# # #     # Load model and imputer
# # #     model = joblib.load('models/outage_predictor.pkl')
# # #     imputer = joblib.load('models/imputer.pkl')

# # #     features = ['CONGESTION (%)', 'DL_AVG_THRPT_PER_USER-Mbps', 'ERAB_Drop rate',
# # #                 'CELL_THRPT-Mbps', 'ERAB Success Rate', 'Average LTE Connected Users']
# # #     X_imputed = imputer.transform(df[features])
# # #     df['Potential_Outage'] = model.predict(X_imputed)

# # #     st.subheader("📊 Sample Data with Predictions")
# # #     st.dataframe(df[['FRAGMENT_DATE'] + features + ['Potential_Outage']].head(20))

# # #     # KPI Trend - Select Granularity
# # #     st.subheader("📈 KPI Trend Over Time")
# # #     kpi = st.selectbox("Select KPI to Plot", features)
# # #     granularity = st.selectbox("View Trend by", ["Hourly", "Daily", "Weekly", "Monthly"])

# # #     if granularity == "Hourly":
# # #         group_col = "HOUR"
# # #     elif granularity == "Daily":
# # #         group_col = "DATE"
# # #     elif granularity == "Weekly":
# # #         group_col = "WEEK"
# # #     else:
# # #         group_col = "MONTH"

# # #     fig, ax = plt.subplots()
# # #     df.groupby(group_col)[kpi].mean().plot(kind='line', marker='o', ax=ax)
# # #     ax.set_title(f"{granularity} Trend of {kpi}")
# # #     ax.set_xlabel(group_col)
# # #     ax.set_ylabel(kpi)
# # #     plt.xticks(rotation=45)
# # #     st.pyplot(fig)

# # #     # Correlation Heatmap
# # #     st.subheader("📉 Correlation Heatmap")
# # #     fig, ax = plt.subplots()
# # #     sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=ax)
# # #     st.pyplot(fig)

# # #     # Forecasting Congestion
# # #     st.subheader("🔮 Forecasting Congestion (%)")
# # #     df_prophet = df[['FRAGMENT_DATE', 'CONGESTION (%)']].dropna().rename(columns={'FRAGMENT_DATE': 'ds', 'CONGESTION (%)': 'y'})

# # #     m = Prophet()
# # #     m.fit(df_prophet)

# # #     future = m.make_future_dataframe(periods=90)
# # #     forecast = m.predict(future)

# # #     st.write("### Forecast Plot")
# # #     st.pyplot(m.plot(forecast))

# # #     st.write("### Forecast Components")
# # #     st.pyplot(m.plot_components(forecast))

# # #     st.success("✅ Outage Prediction & Forecast Completed!")

# # # else:
# # #     st.info("📂 Please upload an Excel file with network KPI data.")





























# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from prophet import Prophet
# # import joblib

# # st.set_page_config(page_title="Outage Forecasting App", layout="wide")

# # # Title and instructions
# # st.title("📡 Outage Prediction & Forecasting App")
# # st.markdown("""
# # Welcome to the Outage Prediction app!  
# # This tool helps you visualize, predict, and forecast possible **network outages** using your KPI Excel data.

# # 👉 **Steps:**
# # 1. Upload your Excel file.
# # 2. View analyzed data with predictions.
# # 3. See KPI trends and relationships.
# # 4. Forecast **network congestion** for the future.
# # """)

# # uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"], help="Upload an Excel file with network KPI data (Date, Congestion %, Drop Rate, etc.)")

# # if uploaded_file:
# #     # Load and preprocess data
# #     df = pd.read_excel(uploaded_file)
# #     df['FRAGMENT_DATE'] = pd.to_datetime(df['FRAGMENT_DATE'], errors='coerce', format='%d-%b-%y %H')

# #     df['DATE'] = df['FRAGMENT_DATE'].dt.date
# #     df['HOUR'] = df['FRAGMENT_DATE'].dt.hour
# #     df['DAY'] = df['FRAGMENT_DATE'].dt.day_name()
# #     df['WEEK'] = df['FRAGMENT_DATE'].dt.to_period('W').astype(str)
# #     df['MONTH'] = df['FRAGMENT_DATE'].dt.month_name()

# #     df.replace(0, np.nan, inplace=True)
# #     df['CONGESTION (%)'] = df['CONGESTION (%)'].fillna(0)
# #     df['ERAB_Drop rate'] = df['ERAB_Drop rate'].fillna(0)

# #     # Load trained models
# #     model = joblib.load('models/outage_predictor.pkl')
# #     imputer = joblib.load('models/imputer.pkl')

# #     # Predict outages
# #     features = ['CONGESTION (%)', 'DL_AVG_THRPT_PER_USER-Mbps', 'ERAB_Drop rate',
# #                 'CELL_THRPT-Mbps', 'ERAB Success Rate', 'Average LTE Connected Users']
# #     X_imputed = imputer.transform(df[features])
# #     df['Potential_Outage'] = model.predict(X_imputed)

# #     # Show sample of predictions
# #     st.subheader("🧾 Sample Data with Outage Predictions")
# #     st.dataframe(df[['FRAGMENT_DATE'] + features + ['Potential_Outage']].head(20))

# #     # Add explanation
# #     st.info("""
# #     ✅ Rows labeled `1` in **Potential_Outage** indicate a likely service outage during that time period.
# #     Use this insight to proactively manage and troubleshoot your network.
# #     """)

# #     # KPI trend over time
# #     st.subheader("📈 KPI Trend Visualizer")
# #     kpi = st.selectbox("📊 Select a KPI to Visualize", features)
# #     granularity = st.radio("🔎 Choose Time Granularity", ["Hourly", "Daily", "Weekly", "Monthly"])

# #     if granularity == "Hourly":
# #         group_col = "HOUR"
# #     elif granularity == "Daily":
# #         group_col = "DATE"
# #     elif granularity == "Weekly":
# #         group_col = "WEEK"
# #     else:
# #         group_col = "MONTH"

# #     trend_df = df.groupby(group_col)[kpi].mean().reset_index()

# #     fig, ax = plt.subplots(figsize=(10, 5))
# #     sns.lineplot(x=group_col, y=kpi, data=trend_df, marker='o', ax=ax)
# #     ax.set_title(f"Average {kpi} by {granularity}", fontsize=14)
# #     ax.set_xlabel(granularity)
# #     ax.set_ylabel(f"{kpi}")
# #     ax.grid(True)
# #     plt.xticks(rotation=45)
# #     st.pyplot(fig)

# #     st.info(f"🔍 This chart shows how **{kpi}** changes over time, helping you spot trends or unusual patterns.")

# #     # Correlation Heatmap
# #     st.subheader("📉 KPI Relationship Heatmap")
# #     st.markdown("Explore how different KPIs are related. A strong correlation means two metrics move together.")
# #     fig, ax = plt.subplots(figsize=(8, 6))
# #     sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
# #     ax.set_title("Correlation Between KPIs", fontsize=14)
# #     st.pyplot(fig)

# #     # Forecasting with Prophet
# #     st.subheader("🔮 Forecast Future Congestion (%)")
# #     st.markdown("Using past data, the app predicts future **network congestion levels** to help in proactive planning.")

# #     df_prophet = df[['FRAGMENT_DATE', 'CONGESTION (%)']].dropna().rename(columns={'FRAGMENT_DATE': 'ds', 'CONGESTION (%)': 'y'})

# #     m = Prophet()
# #     m.fit(df_prophet)

# #     future = m.make_future_dataframe(periods=90)
# #     forecast = m.predict(future)

# #     st.markdown("### 📆 Forecast Chart")
# #     forecast_fig = m.plot(forecast)
# #     st.pyplot(forecast_fig)

# #     st.markdown("### 📊 Forecast Components (Trend, Weekly, Yearly Patterns)")
# #     component_fig = m.plot_components(forecast)
# #     st.pyplot(component_fig)

# #     st.success("✅ Forecasting and Outage Prediction Completed Successfully!")

# # else:
# #     st.warning("📂 Please upload an Excel file with network KPI data to begin.")






















# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from prophet import Prophet
# import plotly.express as px

# # App Title and Description
# st.set_page_config(page_title="Outage Prediction & Forecasting", layout="wide")
# st.title("📡 Outage Prediction & Forecasting App")
# st.markdown("Upload your **KPI Excel file** to **predict potential outages** and **forecast future congestion trends** using AI.")

# uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx"])

# if uploaded_file:
#     with st.spinner("Reading and analyzing the uploaded file..."):
#         try:
#             # Load and preprocess
#             df = pd.read_excel(uploaded_file)
#             df['FRAGMENT_DATE'] = pd.to_datetime(df['FRAGMENT_DATE'], errors='coerce', format='%d-%b-%y %H')

#             df['DATE'] = df['FRAGMENT_DATE'].dt.date
#             df['HOUR'] = df['FRAGMENT_DATE'].dt.hour
#             df['DAY'] = df['FRAGMENT_DATE'].dt.day_name()
#             df['WEEK'] = df['FRAGMENT_DATE'].dt.to_period('W').astype(str)
#             df['MONTH'] = df['FRAGMENT_DATE'].dt.month_name()

#             df.replace(0, np.nan, inplace=True)
#             df['CONGESTION (%)'] = df['CONGESTION (%)'].fillna(0)
#             df['ERAB_Drop rate'] = df['ERAB_Drop rate'].fillna(0)

#             # Load model and imputer
#             model = joblib.load('models/outage_predictor.pkl')
#             imputer = joblib.load('models/imputer.pkl')

#             features = ['CONGESTION (%)', 'DL_AVG_THRPT_PER_USER-Mbps', 'ERAB_Drop rate',
#                         'CELL_THRPT-Mbps', 'ERAB Success Rate', 'Average LTE Connected Users']
#             X_imputed = imputer.transform(df[features])
#             df['Potential_Outage'] = model.predict(X_imputed)

#             # Summary
#             outage_count = df['Potential_Outage'].sum()
#             st.success(f"✅ Outage Prediction Completed! {outage_count} potential outages detected in your dataset.")

#             # Display Sample Data
#             st.subheader("📊 Sample Data with Predictions")
#             st.dataframe(df[['FRAGMENT_DATE'] + features + ['Potential_Outage']].head(20), use_container_width=True)

#             # KPI Trend
#             st.subheader("📈 KPI Trend Over Time")
#             kpi = st.selectbox("Select KPI to Plot", features, help="Choose the Key Performance Indicator to visualize")
#             granularity = st.radio("View Trend by", ["Hourly", "Daily", "Weekly", "Monthly"], horizontal=True)

#             group_col = {"Hourly": "HOUR", "Daily": "DATE", "Weekly": "WEEK", "Monthly": "MONTH"}[granularity]
#             trend_data = df.groupby(group_col)[kpi].mean().reset_index()

#             fig_kpi = px.line(trend_data, x=group_col, y=kpi, markers=True,
#                               title=f"{granularity} Trend of {kpi}",
#                               labels={group_col: granularity, kpi: kpi})
#             st.plotly_chart(fig_kpi, use_container_width=True)

#             # Correlation Heatmap
#             st.subheader("📉 Correlation Between KPIs")
#             st.caption("This heatmap shows how different KPIs are related to each other. A value close to 1 means strong correlation.")
#             fig, ax = plt.subplots()
#             sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=ax)
#             st.pyplot(fig)

#             # Forecasting
#             st.subheader("🔮 Forecasting Congestion (%) Over Next 90 Days")
#             df_prophet = df[['FRAGMENT_DATE', 'CONGESTION (%)']].dropna().rename(columns={'FRAGMENT_DATE': 'ds', 'CONGESTION (%)': 'y'})

#             m = Prophet()
#             m.fit(df_prophet)

#             future = m.make_future_dataframe(periods=90)
#             forecast = m.predict(future)

#             st.markdown("### Forecast Plot")
#             fig_forecast = m.plot(forecast)
#             st.pyplot(fig_forecast)

#             st.markdown("### Forecast Components")
#             fig_components = m.plot_components(forecast)
#             st.pyplot(fig_components)

#             # Download Button
#             st.download_button("📥 Download Data with Predictions",
#                                data=df.to_csv(index=False).encode('utf-8'),
#                                file_name="outage_predictions.csv",
#                                mime='text/csv')

#         except Exception as e:
#             st.error(f"❌ Error processing file: {e}")

# else:
#     st.info("Please upload a valid KPI Excel file with a 'FRAGMENT_DATE' column and KPI fields.")



















import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import joblib

# --- Title & File Upload ---
st.title("📡 Outage Prediction & Forecasting App")
st.write("Upload your KPI Excel file to predict and forecast outages. Site-level filtering is now supported!")

uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Show columns for debug
    st.write("📋 Columns detected:", df.columns.tolist())

    # Parse date
    df['FRAGMENT_DATE'] = pd.to_datetime(df['FRAGMENT_DATE'], errors='coerce', format='%d-%b-%y %H')

    # Add time features
    df['DATE'] = df['FRAGMENT_DATE'].dt.date
    df['HOUR'] = df['FRAGMENT_DATE'].dt.hour
    df['DAY'] = df['FRAGMENT_DATE'].dt.day_name()
    df['WEEK'] = df['FRAGMENT_DATE'].dt.to_period('W').astype(str)
    df['MONTH'] = df['FRAGMENT_DATE'].dt.month_name()

    # Replace missing values
    df.replace(0, np.nan, inplace=True)
    df['CONGESTION (%)'] = df['CONGESTION (%)'].fillna(0)
    df['ERAB_Drop rate'] = df['ERAB_Drop rate'].fillna(0)

    # Detect site column
    site_col = None
    for col in ['SITE', 'SITE_ID', 'CELL_CODE', 'CELL_NAME']:
        if col in df.columns:
            site_col = col
            break

    # Site-specific filtering
    if site_col:
        site_selected = st.selectbox(f"🏷️ Select Site ({site_col}) to Analyze", df[site_col].unique())
        df = df[df[site_col] == site_selected]
        st.success(f"🔍 Analyzing data for site: `{site_selected}`")
    else:
        st.warning("⚠️ No site/cell column detected. Site-specific filtering will be disabled.")

    # Load model and imputer
    model = joblib.load('models/outage_predictor.pkl')
    imputer = joblib.load('models/imputer.pkl')

    # Feature columns for model
    features = ['CONGESTION (%)', 'DL_AVG_THRPT_PER_USER-Mbps', 'ERAB_Drop rate',
                'CELL_THRPT-Mbps', 'ERAB Success Rate', 'Average LTE Connected Users']

    # Prediction
    X_imputed = imputer.transform(df[features])
    df['Potential_Outage'] = model.predict(X_imputed)

    # Show sample predictions
    st.subheader("📊 Sample Data with Predictions")
    st.dataframe(df[['FRAGMENT_DATE'] + ([site_col] if site_col else []) + features + ['Potential_Outage']].head(20))

    # KPI Trend Plot
    st.subheader("📈 KPI Trend Over Time")
    kpi = st.selectbox("Select KPI to Plot", features)
    granularity = st.selectbox("View Trend by", ["Hourly", "Daily", "Weekly", "Monthly"])

    if granularity == "Hourly":
        group_col = "HOUR"
    elif granularity == "Daily":
        group_col = "DATE"
    elif granularity == "Weekly":
        group_col = "WEEK"
    else:
        group_col = "MONTH"

    fig, ax = plt.subplots()
    df.groupby(group_col)[kpi].mean().plot(kind='line', marker='o', ax=ax)
    ax.set_title(f"{granularity} Trend of {kpi}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(kpi)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("📉 Correlation Heatmap (Filtered Site Data)")
    fig, ax = plt.subplots()
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Forecasting Congestion
    st.subheader("🔮 Forecasting Congestion (%)")
    df_prophet = df[['FRAGMENT_DATE', 'CONGESTION (%)']].dropna().rename(columns={'FRAGMENT_DATE': 'ds', 'CONGESTION (%)': 'y'})

    if not df_prophet.empty:
        m = Prophet()
        m.fit(df_prophet)

        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)

        st.write("### Forecast Plot")
        st.pyplot(m.plot(forecast))

        st.write("### Forecast Components")
        st.pyplot(m.plot_components(forecast))
    else:
        st.error("Not enough data available for forecasting.")

    st.success("✅ Outage Prediction & Forecast Completed!")

else:
    st.info("📂 Please upload an Excel file with network KPI data.")
