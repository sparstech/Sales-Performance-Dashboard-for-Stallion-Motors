
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

st.set_page_config(layout="wide", page_title="Stallion Motors — Sales Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("stallion_sales_data.csv", parse_dates=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df

df = load_data()

st.title("Stallion Motors — Sales Performance Dashboard")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    date_range = st.date_input("Date range", [min_date, max_date])
    regions = ["All"] + sorted(df["Region"].unique().tolist())
    region = st.selectbox("Region", regions)
    salesperson_options = ["All"] + sorted(df["Salesperson"].unique().tolist())
    salesperson = st.selectbox("Salesperson", salesperson_options)
    model_options = ["All"] + sorted(df["Vehicle_Model"].unique().tolist())
    model = st.selectbox("Vehicle Model", model_options)
    include_forecast = st.checkbox("Show 6-month forecast (Prophet)", value=True)

# Apply filters
df_filtered = df.copy()
start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
df_filtered = df_filtered[(df_filtered["Date"] >= start) & (df_filtered["Date"] <= end)]
if region != "All":
    df_filtered = df_filtered[df_filtered["Region"] == region]
if salesperson != "All":
    df_filtered = df_filtered[df_filtered["Salesperson"] == salesperson]
if model != "All":
    df_filtered = df_filtered[df_filtered["Vehicle_Model"] == model]

# KPIs
col1, col2, col3, col4 = st.columns(4)
total_revenue = (df_filtered["Price"] * df_filtered["Quantity"]).sum()
total_units = df_filtered["Quantity"].sum()
avg_price = (df_filtered["Price"] * df_filtered["Quantity"]).sum() / max(1, total_units)
total_profit = df_filtered["Profit"].sum()

col1.metric("Total Revenue", f"${total_revenue:,.0f}")
col2.metric("Units Sold", f"{int(total_units):,}")
col3.metric("Avg Selling Price", f"${avg_price:,.0f}")
col4.metric("Total Profit", f"${total_profit:,.0f}")

# Time series
monthly = df_filtered.groupby(pd.Grouper(key="Date", freq="M")).agg(Revenue=("Price","sum"), Units=("Quantity","sum")).reset_index()
fig1 = px.line(monthly, x="Date", y="Revenue", title="Monthly Revenue")
st.plotly_chart(fig1, use_container_width=True)

# Top models
top_models = df_filtered.groupby("Vehicle_Model").agg(Revenue=("Price","sum"), Units=("Quantity","sum")).reset_index().sort_values("Revenue", ascending=False).head(10)
fig2 = px.bar(top_models, x="Vehicle_Model", y="Revenue", title="Top 10 Models by Revenue")
st.plotly_chart(fig2, use_container_width=True)

# Sales by region
sales_region = df_filtered.groupby("Region").agg(Revenue=("Price","sum")).reset_index()
fig3 = px.pie(sales_region, values="Revenue", names="Region", title="Revenue Share by Region")
st.plotly_chart(fig3, use_container_width=True)

# Show raw data
with st.expander("Show raw sales data (first 200 rows)"):
    st.dataframe(df_filtered.head(200))

# Forecast (if available)
if include_forecast:
    try:
        prophet_forecast = pd.read_csv("stallion_prophet_forecast.csv", parse_dates=["ds"])
        figf = px.line(prophet_forecast, x="ds", y="yhat", title="Prophet 6-Month Forecast (Revenue)")
        st.plotly_chart(figf, use_container_width=True)
    except Exception as e:
        st.warning("Prophet forecast not available. Run sales_forecast.ipynb to generate forecasts. Error: " + str(e))
