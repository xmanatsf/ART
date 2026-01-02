const state = {
  rawRows: [],
  dates: [],
  columns: [],
  normalized: {},
};

const elements = {
  reloadButton: document.getElementById("reload-data"),
  status: document.getElementById("data-status"),
  stockSelect: document.getElementById("stock-select"),
  benchmarkSelect: document.getElementById("benchmark-select"),
  normalizeToggle: document.getElementById("normalize-toggle"),
  zscoreWindow: document.getElementById("zscore-window"),
  offsetYears: document.getElementById("offset-years"),
  offsetYearsValue: document.getElementById("offset-years-value"),
  frequencySelect: document.getElementById("frequency-select"),
  alphaWindow: document.getElementById("alpha-window"),
  latestDate: document.getElementById("latest-date"),
  metricZscore: document.getElementById("metric-zscore"),
  metricAlpha: document.getElementById("metric-alpha"),
  metricCumAlpha: document.getElementById("metric-cum-alpha"),
};

const formatNumber = (value, digits = 4) =>
  Number.isFinite(value) ? value.toFixed(digits) : "N/A";

const parseDateValue = (value) => {
  if (!value) return null;
  const trimmed = String(value).trim();
  if (/^\d{8}$/.test(trimmed)) {
    const year = Number(trimmed.slice(0, 4));
    const month = Number(trimmed.slice(4, 6)) - 1;
    const day = Number(trimmed.slice(6, 8));
    return new Date(Date.UTC(year, month, day));
  }
  const parsed = new Date(trimmed);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
};

const parseNumber = (value) => {
  if (value === null || value === undefined) return null;
  const cleaned = String(value).replace(/,/g, "").trim();
  if (cleaned === "" || cleaned === "#N/A" || cleaned === "NA" || cleaned === "N/A") {
    return null;
  }
  const num = Number(cleaned);
  return Number.isFinite(num) ? num : null;
};

const normalizeSeries = (series) => {
  const firstValid = series.find((val) => Number.isFinite(val));
  if (!Number.isFinite(firstValid) || firstValid === 0) return series.slice();
  return series.map((val) => (Number.isFinite(val) ? val / firstValid : null));
};

const movingZScore = (series, window) => {
  const result = [];
  for (let i = 0; i < series.length; i += 1) {
    if (i + 1 < window) {
      result.push(null);
      continue;
    }
    const slice = series.slice(i + 1 - window, i + 1).filter((v) => Number.isFinite(v));
    if (slice.length < window) {
      result.push(null);
      continue;
    }
    const mean = slice.reduce((acc, val) => acc + val, 0) / slice.length;
    const variance =
      slice.reduce((acc, val) => acc + (val - mean) ** 2, 0) / slice.length;
    const std = Math.sqrt(variance);
    result.push(std === 0 ? null : (series[i] - mean) / std);
  }
  return result;
};

const weeklyResample = (dates, series) => {
  const buckets = new Map();
  dates.forEach((date, idx) => {
    const weekKey = `${date.getUTCFullYear()}-${date.getUTCMonth()}-${getWeekNumber(date)}`;
    const existing = buckets.get(weekKey);
    if (!existing || date > existing.date) {
      buckets.set(weekKey, { date, value: series[idx] });
    }
  });
  const resampledDates = [];
  const resampledValues = [];
  [...buckets.values()]
    .sort((a, b) => a.date - b.date)
    .forEach((item) => {
      resampledDates.push(item.date);
      resampledValues.push(item.value);
    });
  return { dates: resampledDates, values: resampledValues };
};

const getWeekNumber = (date) => {
  const firstDay = new Date(Date.UTC(date.getUTCFullYear(), 0, 1));
  const dayDiff = Math.floor((date - firstDay) / (24 * 60 * 60 * 1000));
  return Math.ceil((dayDiff + firstDay.getUTCDay() + 1) / 7);
};

const calculateReturns = (series) =>
  series.map((val, idx) => {
    if (idx === 0 || !Number.isFinite(val) || !Number.isFinite(series[idx - 1])) {
      return null;
    }
    return val / series[idx - 1] - 1;
  });

const rollingAlpha = (stockReturns, benchmarkReturns, window) => {
  const alpha = [];
  for (let i = 0; i < stockReturns.length; i += 1) {
    if (i + 1 < window) {
      alpha.push(null);
      continue;
    }
    const sliceY = stockReturns.slice(i + 1 - window, i + 1);
    const sliceX = benchmarkReturns.slice(i + 1 - window, i + 1);
    if (sliceY.some((v) => v === null) || sliceX.some((v) => v === null)) {
      alpha.push(null);
      continue;
    }
    const meanX = sliceX.reduce((acc, val) => acc + val, 0) / window;
    const meanY = sliceY.reduce((acc, val) => acc + val, 0) / window;
    const covXY = sliceX.reduce((acc, val, idx) => acc + (val - meanX) * (sliceY[idx] - meanY), 0);
    const varX = sliceX.reduce((acc, val) => acc + (val - meanX) ** 2, 0);
    const beta = varX === 0 ? 0 : covXY / varX;
    alpha.push(meanY - beta * meanX);
  }
  return alpha;
};

const cumulativeSum = (series) => {
  let running = 0;
  return series.map((val) => {
    if (!Number.isFinite(val)) return null;
    running += val;
    return running;
  });
};

const updateSelectOptions = (select, options) => {
  select.innerHTML = "";
  options.forEach((option) => {
    const opt = document.createElement("option");
    opt.value = option;
    opt.textContent = option;
    select.appendChild(opt);
  });
};

const updateCharts = () => {
  if (!state.columns.length) return;

  const selectedStock = elements.stockSelect.value;
  const selectedBenchmark = elements.benchmarkSelect.value;
  const normalize = elements.normalizeToggle.checked;
  const zWindow = Number(elements.zscoreWindow.value);
  const offsetYears = Number(elements.offsetYears.value);
  const frequency = elements.frequencySelect.value;
  const alphaWindow = Number(elements.alphaWindow.value);

  const stockSeries = normalize
    ? state.normalized[selectedStock]
    : state.rawRows.map((row) => row[selectedStock]);
  const benchmarkSeries = normalize
    ? state.normalized[selectedBenchmark]
    : state.rawRows.map((row) => row[selectedBenchmark]);

  const { dates: zDates, values: zValues } =
    frequency === "Weekly"
      ? weeklyResample(state.dates, stockSeries)
      : { dates: state.dates, values: stockSeries };

  const zScores = movingZScore(zValues, zWindow);
  const endDate = zDates[zDates.length - 1];
  const startDate = new Date(endDate);
  startDate.setUTCMonth(startDate.getUTCMonth() - offsetYears * 12 + 1);

  const filtered = zDates.reduce(
    (acc, date, idx) => {
      if (date >= startDate && date <= endDate) {
        acc.dates.push(date);
        acc.values.push(zScores[idx]);
      }
      return acc;
    },
    { dates: [], values: [] }
  );

  if (filtered.dates.length === 0) {
    Plotly.newPlot(
      "zscore-chart",
      [],
      {
        margin: { t: 40, r: 20, b: 40, l: 50 },
        height: 360,
        title: "Moving Z-Scores",
        annotations: [
          {
            text: "Not enough data for Z-Score window.",
            showarrow: false,
            xref: "paper",
            yref: "paper",
            x: 0.5,
            y: 0.5,
          },
        ],
      },
      { displayModeBar: false }
    );
  } else {
    const zscoreTrace = {
      x: filtered.dates,
      y: filtered.values,
      type: "scatter",
      mode: "lines",
      name: "Z-Score",
      line: { color: "#2563eb" },
    };

    const zscoreLayout = {
      margin: { t: 40, r: 20, b: 40, l: 50 },
      height: 360,
      title: `Moving Z-Scores (${frequency})`,
      shapes: [0, -2, 2, -4, 4].map((level) => ({
        type: "line",
        x0: filtered.dates[0],
        x1: filtered.dates[filtered.dates.length - 1],
        y0: level,
        y1: level,
        line: {
          dash: "dash",
          color: level === 0 ? "#111827" : level < 0 ? "#ef4444" : "#3b82f6",
        },
      })),
    };

    Plotly.newPlot("zscore-chart", [zscoreTrace], zscoreLayout, { displayModeBar: false });
  }

  const stockReturns = calculateReturns(stockSeries);
  const benchmarkReturns = calculateReturns(benchmarkSeries);
  const alphaSeries = rollingAlpha(stockReturns, benchmarkReturns, alphaWindow);
  const alphaDates = state.dates;

  const alphaTrace = {
    x: alphaDates,
    y: alphaSeries,
    type: "scatter",
    mode: "lines",
    name: "Alpha",
    line: { color: "#0ea5e9" },
  };

  if (!alphaDates.length) {
    Plotly.newPlot(
      "alpha-chart",
      [],
      {
        margin: { t: 40, r: 20, b: 40, l: 50 },
        height: 360,
        title: "Rolling Alpha vs Benchmark",
        annotations: [
          {
            text: "Not enough data for rolling alpha.",
            showarrow: false,
            xref: "paper",
            yref: "paper",
            x: 0.5,
            y: 0.5,
          },
        ],
      },
      { displayModeBar: false }
    );
  } else {
    Plotly.newPlot(
      "alpha-chart",
      [alphaTrace],
      {
        margin: { t: 40, r: 20, b: 40, l: 50 },
        height: 360,
        title: "Rolling Alpha vs Benchmark",
        shapes: [
          {
            type: "line",
            x0: alphaDates[0],
            x1: alphaDates[alphaDates.length - 1],
            y0: 0,
            y1: 0,
            line: { dash: "dash", color: "#94a3b8" },
          },
        ],
      },
      { displayModeBar: false }
    );
  }

  const cumAlpha = cumulativeSum(alphaSeries);
  const cumTrace = {
    x: alphaDates,
    y: cumAlpha,
    type: "scatter",
    mode: "lines",
    name: "Cumulative Alpha",
    line: { color: "#7c3aed" },
  };

  if (!alphaDates.length) {
    Plotly.newPlot(
      "cum-alpha-chart",
      [],
      {
        margin: { t: 40, r: 20, b: 40, l: 50 },
        height: 360,
        title: "Cumulative Alpha",
        annotations: [
          {
            text: "Not enough data for cumulative alpha.",
            showarrow: false,
            xref: "paper",
            yref: "paper",
            x: 0.5,
            y: 0.5,
          },
        ],
      },
      { displayModeBar: false }
    );
  } else {
    Plotly.newPlot(
      "cum-alpha-chart",
      [cumTrace],
      {
        margin: { t: 40, r: 20, b: 40, l: 50 },
        height: 360,
        title: "Cumulative Alpha",
      },
      { displayModeBar: false }
    );
  }

  const latestZ = [...zScores].reverse().find((val) => Number.isFinite(val));
  const latestAlpha = [...alphaSeries].reverse().find((val) => Number.isFinite(val));
  const latestCum = [...cumAlpha].reverse().find((val) => Number.isFinite(val));

  elements.metricZscore.textContent = formatNumber(latestZ, 2);
  elements.metricAlpha.textContent = formatNumber(latestAlpha, 5);
  elements.metricCumAlpha.textContent = formatNumber(latestCum, 5);
};

const prepareData = (rows) => {
  const dateColumn = Object.keys(rows[0]).find((col) => col.toLowerCase() === "date") ?? "date";
  const columns = Object.keys(rows[0]).filter((col) => col !== dateColumn);

  const parsedRows = rows
    .map((row) => {
      const date = parseDateValue(row[dateColumn]);
      if (!date) return null;
      const record = { date };
      columns.forEach((col) => {
        record[col] = parseNumber(row[col]);
      });
      return record;
    })
    .filter(Boolean)
    .sort((a, b) => a.date - b.date);

  state.rawRows = parsedRows;
  state.dates = parsedRows.map((row) => row.date);
  state.columns = columns;
  state.normalized = columns.reduce((acc, col) => {
    acc[col] = normalizeSeries(parsedRows.map((row) => row[col]));
    return acc;
  }, {});

  updateSelectOptions(elements.stockSelect, columns);
  updateSelectOptions(elements.benchmarkSelect, columns);
  elements.benchmarkSelect.value = columns.includes("SPY") ? "SPY" : columns[0];

  const latest = state.dates[state.dates.length - 1];
  elements.latestDate.textContent = latest
    ? `Latest date: ${latest.toISOString().slice(0, 10)}`
    : "Latest date: —";
};

const loadCsv = () => {
  elements.status.textContent = "Loading stock_data.csv...";
  Papa.parse("stock_data.csv", {
    download: true,
    header: true,
    skipEmptyLines: true,
    complete: (results) => {
      if (!results.data.length) {
        elements.status.textContent = "No rows found in stock_data.csv.";
        return;
      }
      prepareData(results.data);
      elements.status.textContent = `Loaded ${results.data.length} rows.`;
      updateCharts();
    },
    error: (err) => {
      elements.status.textContent = `Error loading CSV: ${err.message}`;
    },
  });
};

elements.reloadButton.addEventListener("click", loadCsv);
elements.offsetYears.addEventListener("input", (event) => {
  elements.offsetYearsValue.textContent = event.target.value;
});

[
  elements.stockSelect,
  elements.benchmarkSelect,
  elements.normalizeToggle,
  elements.zscoreWindow,
  elements.offsetYears,
  elements.frequencySelect,
  elements.alphaWindow,
].forEach((el) => el.addEventListener("change", updateCharts));

loadCsv();
elements.offsetYearsValue.textContent = elements.offsetYears.value;
