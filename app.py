import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from statsmodels.regression.rolling import RollingOLS


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


def best_sheet_name(excel_file: pd.ExcelFile) -> str:
    best_name = excel_file.sheet_names[0]
    best_score = -1

    for name in excel_file.sheet_names:
        sheet_df = excel_file.parse(name)
        if sheet_df.empty:
            continue
        date_col = detect_date_column(sheet_df)
        parsed_dates = parse_date_series(sheet_df[date_col])
        date_score = parsed_dates.notna().mean()
        numeric_score = sheet_df.drop(columns=[date_col], errors="ignore").apply(
            lambda col: pd.to_numeric(col, errors="coerce").notna().mean()
        )
        score = date_score + numeric_score.mean()
        if score > best_score:
            best_score = score
            best_name = name

    return best_name


@st.cache_data(show_spinner="Loading data...")
def load_excel(data: bytes, sheet_name: str | None) -> pd.DataFrame:
    excel_file = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    resolved_name = sheet_name or best_sheet_name(excel_file)
    return excel_file.parse(resolved_name)


def sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=260, freq="B")
    spy = 350 + np.cumsum(rng.normal(0.2, 1.0, size=len(dates)))
    tsla = 200 + np.cumsum(rng.normal(0.4, 3.0, size=len(dates)))
    avgo = 45 + np.cumsum(rng.normal(0.1, 0.8, size=len(dates)))
    return pd.DataFrame({"date": dates, "SPY": spy, "AVGO": avgo, "TSLA": tsla})


def load_data(upload, sheet_name: str | None, data: bytes | None) -> pd.DataFrame:
    if upload is None:
        return sample_data()

    if data is None:
        data = upload.read()
    if upload.name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))

    return load_excel(data, sheet_name)


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.copy()
    numeric_df = numeric_df.replace(
        {"#N/A": np.nan, "N/A": np.nan, "NA": np.nan, "null": np.nan, "-": np.nan, "": np.nan}
    )
    numeric_df = numeric_df.drop(columns=[col for col in numeric_df.columns if str(col).startswith("Unnamed:")])
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col].astype(str).str.replace(",", ""), errors="coerce")
    numeric_df = numeric_df.dropna(axis=1, how="all")
    numeric_df = numeric_df.dropna(how="all")
    return numeric_df


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in normalized.columns:
        first_valid = normalized[col].first_valid_index()
        if first_valid is not None and normalized[col].loc[first_valid] != 0:
            normalized[col] = normalized[col] / normalized[col].loc[first_valid]
    return normalized


@st.cache_data(show_spinner=True)
def calculate_moving_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / std


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    weekly_dates = (
        df.reset_index()
        .groupby(pd.Grouper(key=df.index.name, freq="W"))[df.index.name]
        .max()
        .tolist()
    )
    return df[df.index.isin(weekly_dates)]


def plot_zscore_chart(df_stock: pd.DataFrame, window: int, offset_years: int, frequency: str) -> go.Figure:
    df_local = df_stock.copy()

    if frequency == "Weekly":
        df_local = resample_weekly(df_local)

    df_local["z_score"] = calculate_moving_zscore(df_local["stock_price"], window)

    end_date = df_local.index.max()
    start_date = end_date - pd.DateOffset(months=offset_years * 12 - 1)
    df_local = df_local[(df_local.index >= start_date) & (df_local.index <= end_date)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_local.index, y=df_local["z_score"], mode="lines", name="Z-Score"))

    z_scores = [0, -2, 2, -4, 4]
    colors = ["black", "blue", "blue", "red", "red"]
    min_date, max_date = df_local.index.min(), df_local.index.max()

    for z_val, color in zip(z_scores, colors):
        fig.add_shape(
            type="line",
            line=dict(dash="dash", color=color),
            x0=min_date,
            x1=max_date,
            y0=z_val,
            y1=z_val,
        )

    fig.update_layout(
        template="plotly_white",
        title=f"Moving Z-Scores for Stock Price ({frequency})",
        xaxis_title="Date",
        yaxis_title="Z-Score",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


@st.cache_data(show_spinner=True)
def calculate_daily_returns(df_stock: pd.DataFrame, symbol1: str, symbol2: str) -> pd.DataFrame:
    df_local = df_stock.copy()
    df_local[f"return_{symbol1}"] = df_local[symbol1].pct_change()
    df_local[f"return_{symbol2}"] = df_local[symbol2].pct_change()
    df_local.dropna(inplace=True)
    return df_local


def calculate_alpha(df_in: pd.DataFrame, symbol1: str, symbol2: str, window: int) -> pd.DataFrame:
    df_sorted = df_in.sort_values("date", ascending=True)

    if f"return_{symbol1}" not in df_sorted.columns or f"return_{symbol2}" not in df_sorted.columns:
        raise ValueError("Return columns missing for alpha calculation.")

    y = df_sorted[f"return_{symbol1}"]
    x = sm.add_constant(df_sorted[f"return_{symbol2}"])
    model = RollingOLS(y, x, window=window)
    results = model.fit()

    df_result = pd.DataFrame(
        {
            "date": df_sorted["date"][window - 1 :],
            "alpha": results.params["const"],
        }
    )
    return df_result.dropna()


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


st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("Professional Stock Analysis Dashboard")
st.caption("Upload an Excel/CSV dataset to explore Z-Score and rolling alpha insights.")

with st.sidebar:
    st.header("Dataset")
    upload = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
    st.caption("Expected format: first column is date, remaining columns are prices.")

    st.header("Analysis Controls")
    zscore_window = st.number_input("Z-score window length", min_value=12, max_value=120, value=21)
    offset_years = st.slider("Years for Z-score window", min_value=1, max_value=7, value=3, step=1)
    frequency = st.selectbox("Z-score frequency", ["Weekly", "Daily"])
    alpha_window = st.number_input("Alpha window (days)", min_value=5, max_value=300, value=21)
    normalize_toggle = st.checkbox("Normalize prices from first observation", value=True)

sheet_options = None
selected_sheet = None
uploaded_bytes = None

if upload is not None:
    uploaded_bytes = upload.getvalue()
    if upload.name.endswith((".xlsx", ".xls")):
        try:
            sheet_options = pd.ExcelFile(io.BytesIO(uploaded_bytes), engine="openpyxl").sheet_names
        except Exception as exc:
            st.error(f"Unable to read Excel sheets: {exc}")
            sheet_options = []

with st.sidebar:
    if sheet_options:
        selected_sheet = st.selectbox("Excel Sheet", ["Auto (best match)"] + sheet_options)

raw_df = load_data(
    upload,
    None if selected_sheet in (None, "Auto (best match)") else selected_sheet,
    uploaded_bytes,
)

if upload is None:
    st.info("Using sample data. Upload your dataset to analyze real prices.")

with st.expander("Preview dataset"):
    st.dataframe(raw_df.head(12), use_container_width=True)


df = parse_date_column(raw_df)
df = clean_numeric(df)

if df.empty:
    st.error("No valid data found. Please upload a dataset with a date column and price columns.")
    st.stop()

if normalize_toggle:
    df = normalize_prices(df)

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
prices.index.name = "date"

if prices.empty:
    st.error("Selected stock and benchmark do not have overlapping data.")
    st.stop()

zscore_df = pd.DataFrame({"stock_price": prices[stock]}, index=prices.index)
zscore_source = resample_weekly(zscore_df) if frequency == "Weekly" else zscore_df
zscore_series = calculate_moving_zscore(zscore_source["stock_price"], zscore_window)
zscore_chart = plot_zscore_chart(zscore_df, zscore_window, offset_years, frequency)

returns_df = calculate_daily_returns(prices.reset_index(), stock, benchmark)
if len(returns_df) < alpha_window:
    st.warning("Not enough data for rolling alpha. Reduce the window or upload more history.")
    alpha_df = pd.DataFrame()
else:
    alpha_df = calculate_alpha(returns_df, stock, benchmark, alpha_window)

alpha_series = alpha_df.set_index("date")["alpha"] if not alpha_df.empty else pd.Series(dtype=float)
cumulative_alpha = alpha_series.cumsum()

metric_cols = st.columns(3)
latest_zscore = zscore_series.dropna()
latest_alpha = alpha_series.dropna()
latest_cum_alpha = cumulative_alpha.dropna()

metric_cols[0].metric("Latest Z-Score", f"{latest_zscore.iloc[-1]:.2f}" if not latest_zscore.empty else "N/A")
metric_cols[1].metric("Rolling Alpha", f"{latest_alpha.iloc[-1]:.5f}" if not latest_alpha.empty else "N/A")
metric_cols[2].metric(
    "Cumulative Alpha", f"{latest_cum_alpha.iloc[-1]:.5f}" if not latest_cum_alpha.empty else "N/A"
)

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Z-Score Trend")
    st.plotly_chart(zscore_chart, use_container_width=True)

with right:
    st.subheader("Rolling Alpha vs Benchmark")
    st.plotly_chart(chart_alpha(alpha_series), use_container_width=True)

st.subheader("Cumulative Alpha")
st.plotly_chart(chart_cumulative_alpha(cumulative_alpha), use_container_width=True)

st.caption("All analytics are derived from historical prices and should be used for informational purposes only.")
