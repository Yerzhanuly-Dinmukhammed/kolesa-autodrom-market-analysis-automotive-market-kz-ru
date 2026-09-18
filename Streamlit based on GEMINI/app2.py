import os

import streamlit as st
import google.generativeai as genai
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data
def load_data():
    df_kolesa = pd.read_csv("kolesa.csv")
    df_kolesa["Currency"] = "KZT"

    df_drom = pd.read_csv("drom.csv")
    df_drom["Currency"] = "RUB"

    def convert_to_kzt(price, currency):
        if pd.isna(price):
            return None
        if currency == "KZT":
            return price
        elif currency == "RUB":
            return price * 5.5
        return price

    df_drom["Price"] = df_drom.apply(
        lambda row: convert_to_kzt(row["Price"], row["Currency"]),
        axis=1
    )

    df = pd.concat([df_kolesa, df_drom], ignore_index=True)

    df = df.dropna(subset=["Price", "Year", "Engine", "Mileage"])

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Engine"] = pd.to_numeric(df["Engine"], errors="coerce")
    df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")

    df = df.dropna(subset=["Price", "Year", "Engine", "Mileage"])

    return df


df = load_data()


# -------------------------
# TRAIN MODEL
# -------------------------

@st.cache_resource
def train_model(df):
    X = df[["Year", "Engine", "Mileage"]]
    y = df["Price"]

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    return model


model = train_model(df)


# -------------------------
# SIMILAR CARS
# -------------------------

def get_similar_cars(df, year, engine, mileage, n=5):
    df_temp = df.copy()

    df_temp["score"] = (
        abs(df_temp["Year"] - year) +
        abs(df_temp["Engine"] - engine) * 10 +
        abs(df_temp["Mileage"] - mileage) / 10000
    )

    return df_temp.sort_values("score").head(n)[
        ["Year", "Engine", "Mileage", "Price"]
    ]


# -------------------------
# GEMINI
# -------------------------

GEMINI_API_KEY = "AIzaSyDSdGAupZU_GNK_dIT20UoFffFz4y_QtbI"
genai.configure(api_key=GEMINI_API_KEY)


# -------------------------
# UI
# -------------------------

st.title("🚗 AI BMW Price Advisor")

st.write("BMW car price prediction using Drom + Kolesa datasets.")

brand = st.selectbox("Brand", ["BMW"])

year = st.number_input("Year", 2000, 2025, 2018)
engine = st.number_input("Engine (L)", 1.0, 6.0, 2.0)
mileage = st.number_input("Mileage", 0, 300000, 80000)

if st.button("Predict Price"):
    car_data = pd.DataFrame([{
        "Year": year,
        "Engine": engine,
        "Mileage": mileage
    }])

    predicted_price = model.predict(car_data)[0]

    similar_cars = get_similar_cars(df, year, engine, mileage)
    avg_price = similar_cars["Price"].mean()

    if predicted_price < avg_price * 0.9:
        verdict = "UNDERPRICED"
    elif predicted_price > avg_price * 1.1:
        verdict = "OVERPRICED"
    else:
        verdict = "FAIR PRICE"

    st.subheader(f"💰 Predicted Price: {predicted_price:,.0f} KZT")
    st.subheader(f"📊 Evaluation: {verdict}")

    st.write("### 🔍 Similar Cars")
    st.dataframe(similar_cars)

    prompt = f"""
You are a BMW car analyst.

Use ONLY this dataset information.

Target car:
Year: {year}
Engine: {engine}
Mileage: {mileage}

Predicted price: {predicted_price}
Evaluation: {verdict}

Similar cars from dataset:
{similar_cars.to_string(index=False)}

Explain briefly:
- why this price makes sense
- how it compares to similar cars
- whether it looks like a good deal
"""

    try:
        if not GEMINI_API_KEY:
            st.warning("Gemini API key is not set. Add GEMINI_API_KEY to your environment or Streamlit secrets.")
        else:
            gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            response = gemini_model.generate_content(prompt)

            st.write("### 🤖 AI Explanation")
            st.write(response.text)
    except Exception as e:
        st.error(f"Gemini API error: {e}")


# -------------------------
# DASHBOARD
# -------------------------

st.divider()
st.header("📊 Market Insights Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Average Price by Year")
    avg_by_year = df.groupby("Year")["Price"].mean().reset_index()
    st.line_chart(avg_by_year, x="Year", y="Price")

with col2:
    st.subheader("Price vs Mileage")
    st.scatter_chart(df, x="Mileage", y="Price")

st.subheader("Average Price by Engine Volume")
avg_by_engine = df.groupby("Engine")["Price"].mean().reset_index()
st.bar_chart(avg_by_engine, x="Engine", y="Price")


# -------------------------
# CAR COMPARISON
# -------------------------

st.divider()
st.header("🚘 Compare Two BMW Cars")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Car A")
    year_a = st.number_input("Year A", 2000, 2025, 2018)
    engine_a = st.number_input("Engine A", 1.0, 6.0, 2.0)
    mileage_a = st.number_input("Mileage A", 0, 300000, 80000)

with col2:
    st.subheader("Car B")
    year_b = st.number_input("Year B", 2000, 2025, 2016)
    engine_b = st.number_input("Engine B", 1.0, 6.0, 3.0)
    mileage_b = st.number_input("Mileage B", 0, 300000, 120000)

if st.button("Compare Cars"):
    car_a = pd.DataFrame([{
        "Year": year_a,
        "Engine": engine_a,
        "Mileage": mileage_a
    }])

    car_b = pd.DataFrame([{
        "Year": year_b,
        "Engine": engine_b,
        "Mileage": mileage_b
    }])

    price_a = model.predict(car_a)[0]
    price_b = model.predict(car_b)[0]

    st.subheader(f"Car A predicted price: {price_a:,.0f} KZT")
    st.subheader(f"Car B predicted price: {price_b:,.0f} KZT")

    if price_a < price_b:
        st.success("Car A looks cheaper based on prediction.")
    elif price_b < price_a:
        st.success("Car B looks cheaper based on prediction.")
    else:
        st.info("Both cars have almost the same predicted price.")