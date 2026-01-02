import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@dataclass
class TrendSignal:
    direction: str
    slope: float
    r2: float
    window: int


def parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    date_col = None
    for col in df.columns:
        if str(col).strip().lower() in {"date", "datetime", "timestamp"}:
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]

    parsed = pd.to_datetime(df[date_col], errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")

    df = df.copy()
    df[date_col] = parsed
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.set_index(date_col)
    return df


def load_data(upload) -> pd.DataFrame:
    if upload is None:
        return sample_data()

    data = upload.read()
    if upload.name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data))
    else:
        df = pd.read_excel(io.BytesIO(data))
    return df


def sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=260, freq="B")
    spy = 350 + np.cumsum(rng.normal(0.2, 1.0, size=len(dates)))
    tsla = 200 + np.cumsum(rng.normal(0.4, 3.0, size=len(dates)))
    avgo = 45 + np.cumsum(rng.normal(0.1, 0.8, size=len(dates)))
    df = pd.DataFrame({"date": dates, "SPY": spy, "AVGO": avgo, "TSLA": tsla})
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
    numeric_df = numeric_df.dropna(how="all")
    return numeric_df


def resample_prices(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "Weekly":
        return df.resample("W-FRI").last().dropna(how="all")
    return df


def compute_zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def rolling_alpha(stock: pd.Series, benchmark: pd.Series, window: int) -> pd.Series:
    stock = stock.copy()
    benchmark = benchmark.copy()
    aligned = pd.concat([stock, benchmark], axis=1).dropna()
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    alphas = [np.nan] * (window - 1)

    for idx in range(window - 1, len(aligned)):
        xw = x[idx - window + 1 : idx + 1]
        yw = y[idx - window + 1 : idx + 1]
        x_mean = xw.mean()
        y_mean = yw.mean()
        denom = ((xw - x_mean) ** 2).sum()
        if denom == 0:
            alphas.append(np.nan)
            continue
        beta = ((xw - x_mean) * (yw - y_mean)).sum() / denom
        intercept = y_mean - beta * x_mean
        prediction = intercept + beta * xw[-1]
        residual = yw[-1] - prediction
        alphas.append(residual)

    return pd.Series(alphas, index=aligned.index)


def trend_signal(series: pd.Series, window: int = 60) -> TrendSignal:
    series = series.dropna()
    if len(series) < window:
        window = max(10, len(series))
    recent = series.tail(window)
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, np.log(recent.values), 1)
    pred = slope * x + intercept
    ss_res = ((np.log(recent.values) - pred) ** 2).sum()
    ss_tot = ((np.log(recent.values) - np.log(recent.values).mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    direction = "Uptrend" if slope > 0 else "Downtrend"
    return TrendSignal(direction=direction, slope=slope, r2=r2, window=window)


def chart_zscore(zscore: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, mode="lines", name="Z-Score"))
    for level, color in [(0, "#4c566a"), (1, "#88c0d0"), (2, "#a3be8c"), (-1, "#d08770"), (-2, "#bf616a")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, opacity=0.6)
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def chart_alpha(alpha: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=alpha.index, y=alpha, mode="lines", name="Rolling Alpha"))
    fig.add_hline(y=0, line_dash="dash", line_color="#4c566a", opacity=0.6)
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def chart_cumulative_alpha(cum_alpha: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_alpha.index, y=cum_alpha, mode="lines", name="Cumulative Alpha"))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
    return fig


st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("Professional Stock Analysis Dashboard")
st.caption("Upload a dataset to explore Z-Score, rolling alpha, and cumulative alpha insights.")

with st.sidebar:
    st.header("Data Controls")
    upload = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
    frequency = st.radio("Z-Score Frequency", ["Daily", "Weekly"], horizontal=True)
    alpha_window = st.slider("Regression Window (days)", min_value=20, max_value=200, value=60, step=5)

raw_df = load_data(upload)

if upload is None:
    st.info("Using sample data. Upload your dataset to analyze real prices.")

df = parse_date_column(raw_df)
df = clean_numeric(df)

if df.empty:
    st.error("No valid data found. Please upload a dataset with a date column and price columns.")
    st.stop()

price_columns = df.columns.tolist()
if not price_columns:
    st.error("No price columns detected.")
    st.stop()

benchmark_default = "SPY" if "SPY" in price_columns else price_columns[0]

with st.sidebar:
    benchmark = st.selectbox("Benchmark", price_columns, index=price_columns.index(benchmark_default))
    available_stocks = [col for col in price_columns if col != benchmark]
    if not available_stocks:
        available_stocks = price_columns
    stock = st.selectbox("Stock", available_stocks)

prices = df[[stock, benchmark]].dropna()

if prices.empty:
    st.error("Selected stock and benchmark do not have overlapping data.")
    st.stop()

resampled = resample_prices(prices, frequency)

zscore = compute_zscore(resampled[stock])
alpha = rolling_alpha(prices[stock], prices[benchmark], alpha_window)
cumulative_alpha = alpha.cumsum()

trend = trend_signal(prices[stock])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Z-Score", f"{zscore.dropna().iloc[-1]:.2f}")
col2.metric("Latest Rolling Alpha", f"{alpha.dropna().iloc[-1]:.2f}")
col3.metric("Cumulative Alpha", f"{cumulative_alpha.dropna().iloc[-1]:.2f}")
col4.metric("Trend Outlook", f"{trend.direction}")

st.divider()

left, right = st.columns([2, 1])
with left:
    st.subheader("Stock Z-Score")
    st.plotly_chart(chart_zscore(zscore), use_container_width=True)

with right:
    st.subheader("Trend Strength")
    st.write(
        f"Based on the last **{trend.window}** sessions, the price trend is **{trend.direction}** "
        f"with an R² of **{trend.r2:.2f}**."
    )
    st.caption("Higher R² indicates stronger trend consistency.")

st.subheader("Rolling Alpha vs Benchmark")
st.plotly_chart(chart_alpha(alpha), use_container_width=True)

st.subheader("Cumulative Alpha")
st.plotly_chart(chart_cumulative_alpha(cumulative_alpha), use_container_width=True)
