# HYG TDA Three-Axis Framework — Reproduction Code

This repository contains the reproduction script for the paper:

**"Topological Structure Changes in Credit Markets Lead Equity Returns:
Why the HYG TDA Three-Axis Framework Works"**

Keishi Takagi (2026) — Independent Researcher

Paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=XXXXXXX

---

## What This Paper Does

Using Topological Data Analysis (TDA) applied to the HYG high-yield bond ETF,
this paper constructs a 3×3×3 prediction framework that forecasts QQQ 20-day
forward returns under market stress conditions (VIX_SMA10 ≥ 20).

Key results:
- OOS predictive spread: **18.63pp** (all three IS/OOS splits: OOS > IS)
- Best cell (long signal): avg = **+8.15%**, win rate = **87%**, n = 15
- Outperforms all 25 alternative methods including fractal and entropy-based approaches

---

## Requirements

```
Python 3.8+
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

Three CSV files are required (daily closing prices):

| File | Description |
|---|---|
| `longterm_hyg.csv` | iShares iBoxx High Yield Corporate Bond ETF (HYG) |
| `longterm_qqq.csv` | Invesco QQQ Trust (NASDAQ-100 ETF) |
| `longterm_vix.csv` | CBOE Volatility Index (VIX) |

Each CSV must have two columns: `trade_date` (YYYY-MM-DD) and `close`.

Data source: Yahoo Finance or equivalent. Coverage: April 2007 – present.

---

## Usage

```bash
python reproduce_paper_v5_en.py \
    --hyg longterm_hyg.csv \
    --qqq longterm_qqq.csv \
    --vix longterm_vix.csv
```

The script reproduces all tables in the paper:

| Output | Paper Table |
|---|---|
| Table 1: Predictive spread by filter condition | Table 2 |
| Table 2: Cell scores under VIX_SMA10≥20 | Table 3 |
| Table 3: IS/OOS three-split validation | Table 4 |
| Table 4: Block bootstrap test | Table 5 |
| Table A: Alternative asset comparison | §2.1 |
| Table B: Alternative H1 indicators | §2.3 |
| Table C: Scalar TDA indicators | §2.4 |
| Table D: Window width sensitivity | §2.5 |
| Table 5: Non-TDA alternative indicators | Table 6 |

---

## Parameters

All parameters follow Gidea & Katz (2018):

| Parameter | Value | Description |
|---|---|---|
| W | 20 | Window width |
| d | 3 | Embedding dimension |
| τ | 3 | Time delay |
| ε | 0.5 | Euler characteristic threshold |
| Z-score window | 252 | Rolling normalization window (trading days) |

---

## License

MIT License — free to use and modify with attribution.

---

## Citation

```
Takagi, K. (2026). Topological Structure Changes in Credit Markets Lead
Equity Returns: Why the HYG TDA Three-Axis Framework Works.
SSRN Working Paper. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=XXXXXXX
```
