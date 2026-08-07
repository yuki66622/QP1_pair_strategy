# QP1 Pairs Trading Strategy

A pairs trading research system built from scratch: data pipeline, pair screening, statistical validation, and (eventually) backtesting and paper trading. Market-neutral, low-frequency, low-capital.

Repository: https://github.com/yuki66622/QP1_pair_strategy

## Why this project

Pairs trading is one of the oldest statistical arbitrage strategies, which makes it a poor place to look for easy alpha and a good place to learn the full quantitative research process: hypothesis, statistical testing, backtesting without self-deception, and execution. The goal of this project is a system where every statistical decision can be defended -- each test module is validated against synthetic data with known answers before it touches market data, and the calibration results (including the failures of textbook assumptions) are documented in the code.

Two findings from that calibration process, as examples of why it is done:

- The Johansen test's asymptotic critical values are systematically too small at this sample size (three years of daily data). The nominal 95% level empirically rejects about 10% of pure random walks. The module therefore defaults to the 99% column and documents the measured rates.
- A first implementation of the Lo-MacKinlay variance ratio divided by the period length twice. Every synthetic random walk then looked strongly mean-reverting -- exactly the kind of bug that produces a beautiful backtest on real data. The synthetic test caught it immediately.

## Current status

| Stage | Status |
|---|---|
| Data pipeline (Yahoo CSV to CRSP format) | Done |
| Single-stock and pair-level filters | Done |
| Statistical validation modules (EG, Johansen, ADF, VR, Hurst, half-life) | Done, tested |
| Large-scale testing over all candidate pairs | Next |
| Trading rules, backtesting, risk control | Planned |
| Paper trading via Alpaca | Planned |

## Tech stack

- Python 3.9+ with numpy, pandas, scipy, statsmodels (scikit-learn, matplotlib, seaborn, tqdm for analysis and plotting)
- Data: Yahoo Finance daily bars (manual CSV download for now), normalized into CRSP column format; CRSP via WRDS planned for backtesting
- Execution target: Alpaca paper trading API (not yet integrated)
- No build tools, no database; everything runs from plain CSV files

## Repository layout

```
data/raw/yahoo/manual/   input: one CSV per ticker from Yahoo Finance
data/processed/          pipeline output: CRSP-format prices, universe, candidate pairs
scripts/                 entry points: format conversion, pair filtering
src/data/                filters and return-matrix construction
src/pair/                statistical tests (one module per test)
tests/                   synthetic-data test suite, one file per module
```

## How to run

### First-time setup

macOS / Linux:

```bash
git clone https://github.com/yuki66622/QP1_pair_strategy.git
cd QP1_pair_strategy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
git clone https://github.com/yuki66622/QP1_pair_strategy.git
cd QP1_pair_strategy
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Everything below assumes the virtual environment is active. On Windows use `python` instead of `python3`.

### Pipeline

1. Download daily-bar CSV files from Yahoo Finance into `data/raw/yahoo/manual/`, one file per ticker (e.g. `AAPL.csv`).
2. Convert to CRSP format:

```bash
python3 scripts/process_yahoo_to_crsp_format.py
```

3. Apply single-stock and pair-level filters (liquidity, price, history length, same sector, correlation, volatility ratio):

```bash
python3 scripts/run_pair_filters.py
```

Output: `data/processed/candidate_pairs.csv`, the list of pairs that pass all filters and move on to statistical validation.

### Statistical tests

Each module in `src/pair/` takes pandas Series and returns a plain dict, so they can be used directly:

```python
from src.pair.johansen import johansen_test
from src.pair.adf_test import adf_test
from src.pair.half_life import half_life

res = johansen_test(price_a, price_b)     # cointegration + hedge ratio
spread = np.log(price_a) - res["hedge_ratio"] * np.log(price_b)
adf_test(spread)                          # spread stationarity
half_life(spread)                         # mean-reversion speed in days
```

Division of labor between the tests: Engle-Granger gives a continuous p-value (useful for ranking and multiple-testing control), Johansen is symmetric in the two assets and produces the hedge ratio, ADF tests spread stationarity for a fixed hedge ratio, the variance ratio tests the random-walk hypothesis at a chosen horizon with heteroskedasticity-robust inference, Hurst and half-life quantify the character and speed of mean reversion rather than test for it.

### Tests

```bash
python3 tests/test_adf.py
python3 tests/test_hurst.py
python3 tests/test_variance_ratio.py
python3 tests/test_half_life.py
python3 tests/test_johansen.py
```

Every suite generates data where the right answer is known (Ornstein-Uhlenbeck processes with preset half-life, pure random walks, persistent series) and asserts three things: the test points in the right direction, the guard branches handle short, constant, and dirty input, and the measured false-positive rate on hundreds of simulated random walks matches the nominal significance level. Where it does not match (Johansen), the measured rate is documented and asserted instead.

## Known limitations

- The current Yahoo-sourced universe contains only surviving stocks, so any backtest on it carries survivorship bias. This is the main reason CRSP data is planned for the backtesting stage.
- Candidate pair screening tests roughly 10^5 pairs; at any significance level this guarantees false positives. Multiple-testing control and out-of-sample confirmation are part of the next stage, not an afterthought.
- No trading results exist yet. Nothing here is investment advice.
