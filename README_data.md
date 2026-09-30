# Data Sourcing Instructions

The reproduction script requires three CSV files.
**Data files are not included in this repository** due to licensing
restrictions. Download them yourself using the instructions below.

---

## Required Files

| Filename | Ticker | Source | Used for |
|----------|--------|--------|----------|
| `longterm_hyg.csv`   | HYG   | Yahoo Finance | Persistence diagrams, all classifiers, reference diagrams |
| `longterm_jnk.csv`   | JNK   | Yahoo Finance | JNK reference diagrams (Table 5) and the turbulence index (Table 19) |
| `longterm_sp500.csv` | ^GSPC | Yahoo Finance | 20-day forward return (outcome variable), drawdown strata (Table 17) |

All three files are required; the script loads each of them
unconditionally. No VIX file is needed — this differs from the companion
*Base Rates and Autocorrelation-Robust Diagnostics* reproduction bundle.

By default the files are read from `data/`. Other locations can be given
with `--hyg`, `--jnk` and `--sp500`.

---

## Download with yfinance (Python)

```python
import os
import yfinance as yf

os.makedirs('data', exist_ok=True)

tickers = {
    'longterm_hyg.csv':   ('HYG',   '2007-01-01'),
    'longterm_jnk.csv':   ('JNK',   '2007-01-01'),
    'longterm_sp500.csv': ('^GSPC', '1990-01-01'),
}

for filename, (ticker, start) in tickers.items():
    df = yf.download(ticker, start=start, auto_adjust=True)
    df = df[['Close']].dropna()
    df.index.name = 'trade_date'
    df.index = df.index.strftime('%Y-%m-%d')
    df.columns = ['close']
    df.to_csv(f'data/{filename}')
    print(f'Saved {filename}: {len(df)} rows')
```

---

## Expected CSV Format

```
trade_date,close
2007-04-11,31.8551
2007-04-12,31.8765
2007-04-13,31.9135
...
```

- Header row required: `trade_date,close`
- Date format: `YYYY-MM-DD`
- Rows in ascending date order
- No missing dates acceptable (weekends/holidays are naturally absent)

---

## Coverage Requirements

The script looks up three reference windows by date (the first trading
day on or up to 14 calendar days after each nadir) in both HYG and JNK:

| Reference | Nadir date |
|-----------|------------|
| GFC       | 2009-03-09 |
| European  | 2011-10-03 |
| COVID     | 2020-03-23 |

Each series must therefore begin at least 60 trading days (the Takens
window `W`) before 2009-03-09, and the S&P 500 series must cover the HYG
sample plus 20 trading days of forward horizon.

---

## Sample Frame and Data Vintage

All series are truncated at **2026-05-15** by default, which is the data
endpoint of Takagi (2026a). Pass `--end-date none` to use every
observation in the input files, or another `YYYY-MM-DD` date.

On each run the script prints a verification block against Takagi
(2026a, SSRN 7057698) Table 1 for the uniform-probability construction:

| Period | n (classified) | Mean | Pos. freq. |
|--------|---------------:|-----:|-----------:|
| Full | 84 | +2.98% | 77.4% |
| IS (≤2019) | 54 | +2.76% | 70.4% |
| OOS (2020–) | 30 | +3.38% | 90.0% |

Two things can break an exact match, and the script reports both when
the check fails:

1. **Sample frame.** The run ends on a date other than 2026-05-15.
2. **Data vintage.** Yahoo Finance periodically re-bases its dividend
   adjustment, which rescales the whole adjusted-price history by a
   constant. This is topologically inert — the Takens point cloud is
   scale-normalised — but it changes the fourth decimal in which prices
   are stored, and a small number of days sit close enough to the
   classifier threshold to be decided by that digit (Table 15).

A discrepancy of one or two classified days is expected under (2). To
confirm that the input series, rather than the code, is the source of a
difference, compare Table 5 (structure of the reference diagrams)
against the manuscript.

---

## Input Digest and Caches

The script prints an `input digest` — the first 12 hex characters of a
SHA-256 hash over the loaded price series and the geometry parameters.
Persistence diagrams and Wasserstein distances are cached under
`--cache` (default `.cache/`) with this digest in the file name, so a
cache built from one data vintage is never reused against another.
Delete `.cache/` to force a full recomputation.
