import io
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@dataclass
class TrendSignal:
    direction: str
    slope: float
    annualized_return: float
    r2: float
    t_stat: float
    p_value: float
    window: int


def normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def parse_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    parsed_ratio = parsed.notna().mean()

    if parsed_ratio < 0.5:
        formatted = pd.to_datetime(series.astype(str).str.strip(), format="%Y%m%d", errors="coerce")
        if formatted.notna().mean() > parsed_ratio:
            parsed = formatted

    return parsed


def detect_date_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).strip().lower() in {"date", "datetime", "timestamp"}:
            return col

    best_col = df.columns[0]
    best_ratio = 0.0
    for col in df.columns:
        parsed = parse_date_series(df[col])
        ratio = parsed.notna().mean()
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = col

    return best_col


def parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    date_col = detect_date_column(df)
    parsed = parse_date_series(df[date_col])

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
        return pd.read_csv(io.BytesIO(data))
    return pd.read_excel(io.BytesIO(data))


def sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=260, freq="B")
    spy = 350 + np.cumsum(rng.normal(0.2, 1.0, size=len(dates)))
    tsla = 200 + np.cumsum(rng.normal(0.4, 3.0, size=len(dates)))
    avgo = 45 + np.cumsum(rng.normal(0.1, 0.8, size=len(dates)))
    return pd.DataFrame({"date": dates, "SPY": spy, "AVGO": avgo, "TSLA": tsla})


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.copy()
    numeric_df = numeric_df.replace(
        {"#N/A": np.nan, "N/A": np.nan, "NA": np.nan, "null": np.nan, "-": np.nan, "": np.nan}
    )
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col].astype(str).str.replace(",", ""), errors="coerce")
    numeric_df = numeric_df.dropna(axis=1, how="all")
    numeric_df = numeric_df.dropna(how="all")
    return numeric_df


def resample_prices(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "Weekly":
        return df.resample("W-FRI").last().dropna(how="all")
    return df


def compute_zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def rolling_alpha(stock: pd.Series, benchmark: pd.Series, window: int) -> pd.Series:
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


def trend_signal(series: pd.Series, window: int) -> TrendSignal:
    series = series.dropna()
    if len(series) < window:
        window = max(10, len(series))
    recent = series.tail(window)
    y = np.log(recent.values)
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    residuals = y - pred
    ss_res = (residuals**2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    s_err = math.sqrt(ss_res / max(len(recent) - 2, 1))
    ssx = ((x - x.mean()) ** 2).sum()
    se_slope = s_err / math.sqrt(ssx) if ssx > 0 else float("nan")
    t_stat = slope / se_slope if se_slope and not math.isnan(se_slope) else 0
    p_value = 2 * (1 - normal_cdf(abs(t_stat)))

    annualized = math.exp(slope * 252) - 1
    direction = "Uptrend" if slope > 0 else "Downtrend"
    return TrendSignal(
        direction=direction,
        slope=slope,
        annualized_return=annualized,
        r2=r2,
        t_stat=t_stat,
        p_value=p_value,
        window=window,
    )


def future_index(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    freq = pd.infer_freq(index) or "B"
    start = index[-1]
    return pd.date_range(start=start, periods=horizon + 1, freq=freq)[1:]


def forecast_prices(series: pd.Series, window: int, horizon: int, conf_level: float) -> pd.DataFrame:
    series = series.dropna()
    window = min(len(series), window)
    recent = series.tail(window)
    y = np.log(recent.values)
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    residuals = y - pred
    s_err = math.sqrt((residuals**2).sum() / max(len(recent) - 2, 1))
    ssx = ((x - x.mean()) ** 2).sum()

    full_index = recent.index.append(future_index(recent.index, horizon))
    x_future = np.arange(len(full_index))
    pred_all = slope * x_future + intercept

    z_lookup = {0.9: 1.645, 0.95: 1.96, 0.99: 2.576}
    z_score = z_lookup.get(conf_level, 1.96)
    se_pred = s_err * np.sqrt(
        1 + 1 / len(recent) + ((x_future - x.mean()) ** 2) / ssx if ssx > 0 else 1
    )
    lower = np.exp(pred_all - z_score * se_pred)
    upper = np.exp(pred_all + z_score * se_pred)

    forecast_df = pd.DataFrame(
        {
            "forecast": np.exp(pred_all),
            "lower": lower,
            "upper": upper,
        },
        index=full_index,
    )
    return forecast_df


def styled_layout():
    st.markdown(
        """
        <style>
        .metric-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #e6e6e6;
            box-shadow: 0 6px 14px rgba(20, 20, 20, 0.05);
        }
        .metric-label {
            font-size: 0.85rem;
            color: #6b7280;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-size: 1.35rem;
            font-weight: 600;
            color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_zscore(zscore: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, mode="lines", name="Z-Score"))
    for level, color in [(0, "#4c566a"), (1, "#88c0d0"), (2, "#a3be8c"), (-1, "#d08770"), (-2, "#bf616a")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, opacity=0.6)
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def chart_alpha(alpha: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=alpha.index, y=alpha, mode="lines", name="Rolling Alpha"))
    fig.add_hline(y=0, line_dash="dash", line_color="#4c566a", opacity=0.6)
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def chart_cumulative_alpha(cum_alpha: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_alpha.index, y=cum_alpha, mode="lines", name="Cumulative Alpha"))
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def chart_forecast(price: pd.Series, forecast: pd.DataFrame, horizon: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price, mode="lines", name="Actual"))
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["forecast"],
            mode="lines",
            name="Trend Forecast",
            line=dict(color="#2563eb"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.15)",
            line=dict(width=0),
            name=f"{horizon}-step confidence",
        )
    )
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=30, b=20))
    return fig


st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
styled_layout()

st.title("Professional Stock Analysis Dashboard")
st.caption(
    "Upload an Excel/CSV dataset to explore Z-Score, rolling alpha, and trend-based forecasts "
    "grounded in historical significance."
)

with st.sidebar:
    st.header("Dataset")
    upload = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
    st.caption("Expected format: first column is date, remaining columns are prices.")

    st.header("Analysis Controls")
    frequency = st.radio("Z-Score Frequency", ["Daily", "Weekly"], horizontal=True)
    alpha_window = st.slider("Regression Window (days)", min_value=20, max_value=200, value=60, step=5)
    trend_window = st.slider("Trend Window (days)", min_value=30, max_value=250, value=120, step=10)
    forecast_horizon = st.slider("Forecast Horizon (days)", min_value=5, max_value=60, value=20, step=5)
    conf_level = st.select_slider("Confidence Level", options=[0.9, 0.95, 0.99], value=0.95)

raw_df = load_data(upload)

if upload is None:
    st.info("Using sample data. Upload your dataset to analyze real prices.")

with st.expander("Preview dataset"):
    st.dataframe(raw_df.head(12), use_container_width=True)

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

trend = trend_signal(prices[stock], trend_window)
forecast = forecast_prices(prices[stock], trend_window, forecast_horizon, conf_level)

metric_cols = st.columns(5)
metric_cols[0].markdown(
    f"<div class='metric-card'><div class='metric-label'>Latest Z-Score</div><div class='metric-value'>{zscore.dropna().iloc[-1]:.2f}</div></div>",
    unsafe_allow_html=True,
)
metric_cols[1].markdown(
    f"<div class='metric-card'><div class='metric-label'>Rolling Alpha</div><div class='metric-value'>{alpha.dropna().iloc[-1]:.2f}</div></div>",
    unsafe_allow_html=True,
)
metric_cols[2].markdown(
    f"<div class='metric-card'><div class='metric-label'>Cumulative Alpha</div><div class='metric-value'>{cumulative_alpha.dropna().iloc[-1]:.2f}</div></div>",
    unsafe_allow_html=True,
)
metric_cols[3].markdown(
    f"<div class='metric-card'><div class='metric-label'>Trend Outlook</div><div class='metric-value'>{trend.direction}</div></div>",
    unsafe_allow_html=True,
)
metric_cols[4].markdown(
    f"<div class='metric-card'><div class='metric-label'>Annualized Trend</div><div class='metric-value'>{trend.annualized_return * 100:.1f}%</div></div>",
    unsafe_allow_html=True,
)

st.divider()

summary_tab, alpha_tab, forecast_tab = st.tabs(["Z-Score", "Alpha", "Forecast"])

with summary_tab:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Stock Z-Score")
        st.plotly_chart(chart_zscore(zscore), use_container_width=True)

    with right:
        st.subheader("Trend Strength & Significance")
        st.write(
            f"Trend computed over the last **{trend.window}** sessions. "
            f"R²: **{trend.r2:.2f}**, t-stat: **{trend.t_stat:.2f}**, "
            f"p-value: **{trend.p_value:.3f}**."
        )
        st.caption("Lower p-value indicates higher statistical significance of the trend.")

with alpha_tab:
    st.subheader("Rolling Alpha vs Benchmark")
    st.plotly_chart(chart_alpha(alpha), use_container_width=True)

    st.subheader("Cumulative Alpha")
    st.plotly_chart(chart_cumulative_alpha(cumulative_alpha), use_container_width=True)

with forecast_tab:
    st.subheader("Trend Forecast with Confidence Band")
    st.plotly_chart(chart_forecast(prices[stock], forecast, forecast_horizon), use_container_width=True)
    st.caption(
        "Forecast is based on a log-linear trend in the selected window. It is not a guarantee of future performance."
    )

st.caption("All analytics are derived from historical prices and should be used for informational purposes only.")
