# Kazakhstan vs. Russia: BMW Market Analysis & AI Price Advisor

Comparative analysis of used BMW listings in Kazakhstan ([Kolesa.kz](https://kolesa.kz)) and Russia ([Drom.ru](https://auto.drom.ru)): web scraping, data cleaning, exploratory analysis, and a Streamlit app that predicts a fair price and explains it with an LLM.

## What's inside

| Stage | What was done | Where |
|---|---|---|
| 1. Scraping | `requests` + BeautifulSoup parsers for both sites, browser-like headers, random delays, retries with exponential backoff on HTTP 429 | `kolesa.kz/`, `drom.auto.ru/` |
| 2. Cleaning | Split listing descriptions into year, engine volume, fuel, gearbox, drive, mileage; type conversion; removal of missing values and duplicates | `Cleaning ...` notebooks |
| 3. EDA & visualization | Descriptive statistics, price and log-price distributions, outliers, correlations, group-by analysis, hypothesis testing; histograms, box/violin plots, scatter plots, heatmaps, bar and pie charts | `EDA ...`, `VIZUALIZATION ...` notebooks |
| 4. Price model | Random Forest regression on year, engine volume and mileage | `Regression Model Based on GEMINI/` |
| 5. App | Streamlit "AI BMW Price Advisor" | `Streamlit based on GEMINI/` |

## Data

- BMW listings only, so the two markets are compared on the same brand.
- About 2,750 cleaned listings from Kolesa.kz and about 3,400 from Drom.ru.
- Kolesa fields: price, year, engine volume, body type, fuel, gearbox, model, vehicle type.
- Drom fields: price, year, engine, fuel, transmission, drive type, mileage, location.
- Russian prices are converted to KZT at a fixed rate (1 RUB = 5.5 KZT) so both markets sit on one scale.

## Key findings

- Newer cars cost more; larger engines cost more.
- Higher mileage lowers the price.
- All-wheel-drive cars are generally more expensive.
- Price distributions are strongly right-skewed, so log price is used.

## AI BMW Price Advisor

Enter a year, engine volume and mileage. The app then:

1. predicts the price in KZT with a Random Forest trained on both markets;
2. finds the five most similar listings in the dataset;
3. labels the car **underpriced**, **fair** or **overpriced** against those listings;
4. asks an LLM (Gemini API) to explain the verdict using only the dataset.

## Run locally

```bash
pip install streamlit pandas scikit-learn google-generativeai
cd "Streamlit based on GEMINI"
streamlit run app2.py
```

The notebooks are written for Google Colab: open, upload the CSV, run all cells.

## Stack

Python · BeautifulSoup · Requests · Pandas · NumPy · Matplotlib / Seaborn · scikit-learn · Streamlit · Gemini API

## Limitations

- BMW only; the model uses three features (year, engine, mileage) and does not account for trim, condition or region.
- Fixed exchange rate; real prices move with the rate and import duties.
- Listing prices are asking prices, not final sale prices.
