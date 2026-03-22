"""
reproduce_paper.py
==================
Topological Structure Changes in Credit Markets Lead Equity Returns
Keishi Takagi (2026) -- Reproduction script for all paper results

Usage:
    python reproduce_paper.py \
        --hyg longterm_hyg.csv \
        --qqq longterm_qqq.csv \
        --vix longterm_vix.csv

Requirements:
    pip install ripser numpy scipy
"""

import argparse
import csv
import math
import statistics
import numpy as np
from ripser import ripser
import warnings
warnings.filterwarnings('ignore')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Data Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_csv(path):
    """Load a two-column CSV with trade_date and close columns."""
    data = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data[row['trade_date']] = float(row['close'])
            except (KeyError, ValueError):
                continue
    return dict(sorted(data.items()))


def align_data(hyg, qqq, vix):
    """Extract common trading dates across all three series (same method as paper)."""
    common = sorted(set(hyg) & set(qqq) & set(vix))
    hyg_v = [hyg[d] for d in common]
    qqq_v = [qqq[d] for d in common]
    vix_v = [vix[d] for d in common]
    return common, hyg_v, qqq_v, vix_v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. VIX SMA10 Filter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_sma(dates, vals, n=10):
    """Compute n-day simple moving average."""
    result = {}
    for i in range(n - 1, len(vals)):
        result[dates[i]] = statistics.mean(vals[i - n + 1:i + 1])
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. TDA Feature Computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Paper parameters (following Gidea & Katz 2018)
W         = 20    # window width
D         = 3     # embedding dimension
DELAY     = 3     # time delay
EULER_THR = 0.5   # filtration threshold for Euler characteristic
ZSCORE_W  = 252   # rolling window for Z-score normalization
ZSCORE_N  = 30    # minimum samples for Z-score (warm-up; not specified in paper)
NEEDED    = W + (D - 1) * DELAY  # required observations per window = 26


def persistent_entropy(persistences):
    """Compute persistent entropy from a list of persistence values."""
    total = sum(persistences)
    if total == 0:
        return 0.0
    ps = [p / total for p in persistences if p > 0]
    return -sum(p * math.log(p) for p in ps)


def rolling_zscore(vals, window=ZSCORE_W, min_n=ZSCORE_N):
    """Normalize values using a 252-day rolling Z-score."""
    zscores = []
    for i in range(len(vals)):
        hist = [vals[j] for j in range(max(0, i - window), i)
                if vals[j] is not None]
        if len(hist) < min_n or vals[i] is None:
            zscores.append(None)
            continue
        m = statistics.mean(hist)
        s = statistics.stdev(hist)
        zscores.append((vals[i] - m) / s if s > 0 else 0.0)
    return zscores


def compute_tda_features(dates, vals, label=''):
    """
    Takens embedding -> Vietoris-Rips PH -> three features (cnt, ent, euler).
    Each feature is Z-score normalized with a 252-day rolling window.
    """
    h1cnt_raw = []
    ent_raw   = []
    euler_raw = []

    n = len(vals)
    if label:
        print(f"  {label} ({n} obs) ...", end='', flush=True)

    for i in range(n):
        seg = np.array(vals[max(0, i - NEEDED + 1):i + 1])

        if len(seg) < NEEDED:
            h1cnt_raw.append(None)
            ent_raw.append(None)
            euler_raw.append(None)
            continue

        # Takens point cloud: (W x D) matrix
        X = np.array(
            [[seg[j + k * DELAY] for k in range(D)] for j in range(W)],
            dtype=np.float64
        )

        # normalize to [0, 1]^3
        mn, mx = X.min(), X.max()
        if mx - mn < 1e-9:
            h1cnt_raw.append(0.0)
            ent_raw.append(0.0)
            euler_raw.append(1.0)
            continue
        X = (X - mn) / (mx - mn)

        # Vietoris-Rips persistent homology up to dimension 1
        dgms = ripser(X, maxdim=1)['dgms']
        h0 = dgms[0]
        h1 = dgms[1][dgms[1][:, 1] < np.inf]

        # Feature 1: H1 count
        h1cnt_raw.append(float(len(h1)))

        # Feature 2: persistent entropy
        persts = (h1[:, 1] - h1[:, 0]).tolist() if len(h1) > 0 else []
        ent_raw.append(persistent_entropy(persts) if persts else 0.0)

        # Feature 3: Euler characteristic (epsilon=0.5)
        b0 = int(np.sum((h0[:, 0] <= EULER_THR) & (h0[:, 1] > EULER_THR)))
        b1 = int(np.sum((h1[:, 0] <= EULER_THR) & (h1[:, 1] > EULER_THR))) \
             if len(h1) > 0 else 0
        euler_raw.append(float(b0 - b1))

        if label and (i + 1) % 1000 == 0:
            print(f" {(i+1)/n*100:.0f}%", end='', flush=True)

    if label:
        print(" done")

    # Z-score normalization
    cnt_z   = rolling_zscore(h1cnt_raw)
    ent_z   = rolling_zscore(ent_raw)
    euler_z = rolling_zscore(euler_raw)

    cnt_map   = {dates[i]: cnt_z[i]   for i in range(n) if cnt_z[i]   is not None}
    ent_map   = {dates[i]: ent_z[i]   for i in range(n) if ent_z[i]   is not None}
    euler_map = {dates[i]: euler_z[i] for i in range(n) if euler_z[i] is not None}

    return cnt_map, ent_map, euler_map


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 3x3x3 Prediction Framework
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def zone3(z):
    """Discretize a Z-score into Low / Mid / High."""
    if z is None:
        return None
    return 'Low' if z <= -1 else ('High' if z >= 1 else 'Mid')


def compute_ret20(dates, qqq_vals):
    """Compute 20-day forward QQQ return (%) for each date."""
    d_idx = {d: i for i, d in enumerate(dates)}
    ret = {}
    for d in dates:
        i = d_idx[d]
        if i + 20 < len(dates):
            ret[d] = (qqq_vals[i + 20] - qqq_vals[i]) / qqq_vals[i] * 100
    return ret


MIN_N = 10  # minimum observations per cell


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4b. Persistence Variants & Multi-scale TDA (v5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_persistence_features(dates, vals, label=''):
    """Compute H1 count, total/mean/max persistence in one pass."""
    cnt_raw = []; tp_raw = []; ap_raw = []; mp_raw = []
    needed = NEEDED
    if label:
        print(f"  {label}...", end='', flush=True)
    for i in range(len(vals)):
        seg = np.array(vals[max(0, i - needed + 1):i + 1])
        if len(seg) < needed:
            for lst in [cnt_raw, tp_raw, ap_raw, mp_raw]:
                lst.append(None)
            continue
        X = np.array([[seg[j + k * DELAY] for k in range(D)] for j in range(W)],
                     dtype=np.float64)
        x_min, x_max = X.min(), X.max()
        if x_max - x_min < 1e-9:
            cnt_raw.append(0.); tp_raw.append(0.)
            ap_raw.append(0.); mp_raw.append(0.)
            continue
        X = (X - x_min) / (x_max - x_min)
        dgms = ripser(X, maxdim=1)['dgms']
        h1f  = dgms[1][dgms[1][:, 1] < np.inf]
        h1p  = (h1f[:, 1] - h1f[:, 0]).tolist() if len(h1f) > 0 else []
        cnt_raw.append(float(len(h1f)))
        if h1p:
            tp_raw.append(float(sum(h1p)))
            ap_raw.append(float(sum(h1p) / len(h1p)))
            mp_raw.append(float(max(h1p)))
        else:
            tp_raw.append(0.); ap_raw.append(0.); mp_raw.append(0.)
        if label and (i + 1) % 1000 == 0:
            print(f"{(i+1)/len(vals)*100:.0f}%...", end='', flush=True)
    if label:
        print(" done")

    def to_map(raw):
        zs = rolling_zscore(raw)
        return {dates[i]: zs[i] for i in range(len(dates)) if zs[i] is not None}

    return to_map(cnt_raw), to_map(tp_raw), to_map(ap_raw), to_map(mp_raw)


def compute_tda_scale(dates, vals, w, label=''):
    """Compute three TDA features (H1 count, entropy, Euler) for a given window width W."""
    cnt_raw = []; ent_raw = []; eul_raw = []
    needed = w + (D - 1) * DELAY
    if label:
        print(f"  W={w} {label}...", end='', flush=True)
    for i in range(len(vals)):
        seg = np.array(vals[max(0, i - needed + 1):i + 1])
        if len(seg) < needed:
            cnt_raw.append(None); ent_raw.append(None); eul_raw.append(None)
            continue
        X = np.array([[seg[j + k * DELAY] for k in range(D)] for j in range(w)],
                     dtype=np.float64)
        x_min, x_max = X.min(), X.max()
        if x_max - x_min < 1e-9:
            cnt_raw.append(0.); ent_raw.append(0.); eul_raw.append(1.)
            continue
        X = (X - x_min) / (x_max - x_min)
        dgms = ripser(X, maxdim=1)['dgms']
        h0f  = dgms[0]
        h1f  = dgms[1][dgms[1][:, 1] < np.inf]
        h1p  = (h1f[:, 1] - h1f[:, 0]).tolist() if len(h1f) > 0 else []
        cnt_raw.append(float(len(h1f)))
        ent_raw.append(persistent_entropy(h1p) if h1p else 0.)
        b0 = int(np.sum((h0f[:, 0] <= EULER_THR) & (h0f[:, 1] > EULER_THR)))
        b1 = int(np.sum((h1f[:, 0] <= EULER_THR) & (h1f[:, 1] > EULER_THR))) if len(h1f) > 0 else 0
        eul_raw.append(float(b0 - b1))
        if label and (i + 1) % 1000 == 0:
            print(f"{(i+1)/len(vals)*100:.0f}%...", end='', flush=True)
    if label:
        print(" done")

    def to_map(raw):
        zs = rolling_zscore(raw)
        return {dates[i]: zs[i] for i in range(len(dates)) if zs[i] is not None}

    return to_map(cnt_raw), to_map(ent_raw), to_map(eul_raw)


def scalar_spread_is_oos(feat_map, filter_fn, ret_map):
    """Compute full-period, IS, and OOS spread for a single scalar feature."""
    def _sp(extra_fn):
        mat = {'Low': [], 'Mid': [], 'High': []}
        for d, z in feat_map.items():
            if not filter_fn(d) or not extra_fn(d):
                continue
            r = ret_map.get(d)
            if r is None:
                continue
            g = zone3(z)
            if g:
                mat[g].append(r)
        avgs = {k: statistics.mean(v) for k, v in mat.items() if len(v) >= MIN_N}
        sp = max(avgs.values()) - min(avgs.values()) if len(avgs) >= 2 else 0.0
        n  = sum(len(v) for v in mat.values())
        return sp, n
    sp_all, _  = _sp(lambda d: True)
    sp_is,  ni = _sp(lambda d: d <= '2019-12-31')
    sp_oos, no = _sp(lambda d: d >= '2020-01-01')
    return sp_all, sp_is, ni, sp_oos, no

def build_matrix(cnt_map, ent_map, euler_map, filter_fn, ret_map):
    """
    Assign returns to the 27 cells of the 3x3x3 framework for dates passing the filter.
    """
    labels = ['Low', 'Mid', 'High']
    mat = {(a, b, c): [] for a in labels for b in labels for c in labels}
    common = set(cnt_map) & set(ent_map) & set(euler_map)

    for d in common:
        if not filter_fn(d):
            continue
        r = ret_map.get(d)
        if r is None:
            continue
        cell = (zone3(cnt_map[d]), zone3(ent_map[d]), zone3(euler_map[d]))
        if None in cell:
            continue
        mat[cell].append(r)

    return mat


def spread(mat):
    """Compute the spread (best cell avg minus worst cell avg)."""
    avgs = {k: statistics.mean(v) for k, v in mat.items() if len(v) >= MIN_N}
    if len(avgs) < 3:
        return 0.0, None, None, avgs
    sp = max(avgs.values()) - min(avgs.values())
    best  = max(avgs, key=avgs.get)
    worst = min(avgs, key=avgs.get)
    return sp, best, worst, avgs


def split_spread(cnt_map, ent_map, euler_map, filter_fn, ret_map,
                 is_end='2019-12-31', oos_start='2020-01-01'):
    """Split into IS and OOS periods and return the spread for each."""
    results = {}
    for period, pfn in [
        ('IS',  lambda d: d <= is_end),
        ('OOS', lambda d: d >= oos_start),
    ]:
        mat = build_matrix(
            cnt_map, ent_map, euler_map,
            lambda d, f=filter_fn, p=pfn: f(d) and p(d),
            ret_map
        )
        sp, best, worst, avgs = spread(mat)
        n = sum(len(v) for v in mat.values())
        results[period] = (sp, n, best, worst, avgs)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Block Bootstrap Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BLOCK_SIZE = 20
N_BOOT     = 2000
SEED       = 42


def circular_block(arr, start, size):
    n = len(arr)
    idx = [(start + i) % n for i in range(size)]
    return arr[idx]


def block_bootstrap_pvalue(mat, n_boot=N_BOOT, block_size=BLOCK_SIZE, seed=SEED):
    """
    Block bootstrap test for the observed spread.
    Null hypothesis: the spread is indistinguishable from a random shuffle of returns.
    """
    labels = ['Low', 'Mid', 'High']
    cells  = []
    rets   = []
    for cell, vs in mat.items():
        for r in vs:
            cells.append(cell)
            rets.append(r)

    rets = np.array(rets)
    n    = len(rets)
    obs_sp, _, _, _ = spread(mat)

    np.random.seed(seed)
    n_blocks = math.ceil(n / block_size)
    boot_spreads = []

    for _ in range(n_boot):
        starts   = np.random.randint(0, n, size=n_blocks)
        shuffled = np.concatenate(
            [circular_block(rets, s, block_size) for s in starts]
        )[:n]

        shuf_mat = {(a, b, c): [] for a in labels for b in labels for c in labels}
        for i, cell in enumerate(cells):
            shuf_mat[cell].append(shuffled[i])

        sp, _, _, _ = spread(shuf_mat)
        boot_spreads.append(sp)

    boot_spreads = np.array(boot_spreads)
    p_val = float(np.mean(boot_spreads >= obs_sp))
    pct95 = float(np.percentile(boot_spreads, 95))
    pct99 = float(np.percentile(boot_spreads, 99))
    return p_val, pct95, pct99, boot_spreads.mean(), obs_sp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Output / Display
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def print_cell_table(mat, avgs, overall, top_n=6):
    labels = ['Low', 'Mid', 'High']
    rows = []
    for a in labels:
        for b in labels:
            for c in labels:
                v = mat[(a, b, c)]
                if len(v) < MIN_N:
                    continue
                avg = statistics.mean(v)
                win = sum(1 for r in v if r > 0) / len(v) * 100
                diff = avg - overall
                pt = ('+2' if diff > 3 else
                      '+1' if diff > 1.5 else
                      '-2' if diff < -3 else
                      '-1' if diff < -1.5 else '0')
                rows.append((abs(diff), a, b, c, avg, win, len(v), pt))

    rows.sort(reverse=True)
    print(f"\n  {'Cell (cnt x ent x Euler)':24s}  {'avg':7s}  {'Win%':5s}  {'n':4s}  {'PT'}")
    print("  " + "-"*52)
    for _, a, b, c, avg, win, n, pt in rows[:top_n]:
        arrow = ('▲▲' if pt == '+2' else '▲ ' if pt == '+1' else
                 '▼▼' if pt == '-2' else '▼ ' if pt == '-1' else '  ')
        print(f"  {arrow} {a}×{b}×{c}          {avg:+6.2f}%  {win:4.0f}%  {n:4d}  [{pt}]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main(hyg_path, qqq_path, vix_path):

    print("=" * 65)
    print("  Paper Results Reproduction Script")
    print("  Keishi Takagi (2026) -- HYG TDA 3x3x3 -> QQQ 20-day forward return prediction")
    print("=" * 65)

    # --- Load data ---
    print("\n[1] Loading data...")
    hyg_raw = load_csv(hyg_path)
    qqq_raw = load_csv(qqq_path)
    vix_raw = load_csv(vix_path)

    dates, hyg_v, qqq_v, vix_v = align_data(hyg_raw, qqq_raw, vix_raw)
    print(f"  Common dates: {dates[0]} to {dates[-1]}  ({len(dates)} trading days)")

    # --- VIX SMA10 filter ---
    vix_sma10 = calc_sma(dates, vix_v, n=10)
    n_vix20   = sum(1 for d in dates if vix_sma10.get(d, 0) >= 20)
    print(f"  Days with VIX_SMA10>=20: {n_vix20} ({n_vix20/len(dates)*100:.1f}%)")

    filter_vix20 = lambda d: vix_sma10.get(d, 0) >= 20
    filter_none  = lambda d: True

    # --- 20-day forward returns ---
    ret_map = compute_ret20(dates, qqq_v)

    # --- TDA feature computation ---
    print("\n[2] Computing TDA features (HYG)...")
    cnt_map, ent_map, euler_map = compute_tda_features(dates, hyg_v, 'HYG')

    # --- Predictive spread (overall) ---
    print_section("Table 1: Predictive Spread by Filter Condition (paper Table 2)")

    for label, filt in [
        ("No filter", filter_none),
        ("VIX_SMA10≥20",   filter_vix20),
    ]:
        mat  = build_matrix(cnt_map, ent_map, euler_map, filt, ret_map)
        sp, best, worst, avgs = spread(mat)
        n    = sum(len(v) for v in mat.values())
        ovr  = (statistics.mean([r for v in mat.values() for r in v])
                if any(mat.values()) else 0)

        splits = split_spread(cnt_map, ent_map, euler_map, filt, ret_map)
        sp_is, n_is = splits['IS'][0],  splits['IS'][1]
        sp_oos, n_oos = splits['OOS'][0], splits['OOS'][1]
        ratio = sp_oos / sp_is if sp_is > 0 else 0

        print(f"\n  Filter: {label}")
        print(f"  n={n}  full-period spread={sp:.2f}pp  overall avg={ovr:+.2f}%")
        print(f"  IS={sp_is:.2f}pp (n={n_is})  "
              f"OOS={sp_oos:.2f}pp (n={n_oos})  "
              f"OOS/IS={ratio:.2f}x  "
              f"{'* OOS>IS' if sp_oos > sp_is else '- IS>OOS'}")
        if best:
            print(f"  Best cell:  {best}  avg={avgs[best]:+.2f}%")
            print(f"  Worst cell: {worst}  avg={avgs.get(worst, 0):+.2f}%")

    # --- Cell score table (VIX>=20) ---
    print_section("Table 2: Cell Scores under VIX_SMA10>=20 (paper Table 3)")

    mat_v20 = build_matrix(cnt_map, ent_map, euler_map, filter_vix20, ret_map)
    sp_v20, best_v20, worst_v20, avgs_v20 = spread(mat_v20)
    ovr_v20 = (statistics.mean([r for v in mat_v20.values() for r in v])
               if any(mat_v20.values()) else 0)
    print(f"\n  overall avg={ovr_v20:+.2f}%  spread={sp_v20:.2f}pp")
    print_cell_table(mat_v20, avgs_v20, ovr_v20, top_n=8)

    # --- IS/OOS three-split validation ---
    print_section("Table 3: IS/OOS Validation -- Three Temporal Splits (paper Table 4)")

    splits_def = [
        ("thru-2015 / 2016+", "2015-12-31", "2016-01-01"),
        ("thru-2019 / 2020+", "2019-12-31", "2020-01-01"),
        ("thru-2021 / 2022+", "2021-12-31", "2022-01-01"),
    ]
    print(f"\n  {'Split':20s}  {'IS spread':10s}  {'OOS spread':12s}  OOS>IS")
    print("  " + "-"*55)
    for slabel, is_end, oos_start in splits_def:
        sp_r = split_spread(
            cnt_map, ent_map, euler_map, filter_vix20, ret_map,
            is_end=is_end, oos_start=oos_start
        )
        si, ni = sp_r['IS'][0],  sp_r['IS'][1]
        so, no = sp_r['OOS'][0], sp_r['OOS'][1]
        mark = '*' if so > si else '-'
        print(f"  {slabel:20s}  {si:.2f}pp (n={ni:4d})  "
              f"{so:.2f}pp (n={no:4d})  {mark}")

    # --- Block bootstrap test ---
    print_section("Table 4: Block Bootstrap Test (paper Table 5)")
    print(f"  Block length={BLOCK_SIZE} days  Replications={N_BOOT}  seed={SEED}")

    for period_label, pfn in [
        ("Full period",     filter_vix20),
        ("IS (thru 2019)", lambda d: filter_vix20(d) and d <= '2019-12-31'),
        ("OOS (2020+)",    lambda d: filter_vix20(d) and d >= '2020-01-01'),
    ]:
        mat_p = build_matrix(cnt_map, ent_map, euler_map, pfn, ret_map)
        n_p   = sum(len(v) for v in mat_p.values())
        p_val, pct95, pct99, boot_mean, obs_sp = block_bootstrap_pvalue(mat_p)
        sig = ('** p<0.01' if p_val < 0.01 else
               '*  p<0.05' if p_val < 0.05 else
               '   n.s.')
        print(f"\n  {period_label} (n={n_p})")
        print(f"    Observed spread={obs_sp:.2f}pp  "
              f"Boot mean={boot_mean:.2f}pp  99th pct={pct99:.2f}pp")
        print(f"    p-value={p_val:.4f}  {sig}")

    # --- Non-TDA alternative indicators ---
    # ----------------------------------------------------------------
    # Table A: Alternative Asset TDA Comparison (paper v5 §2.1)
    # ----------------------------------------------------------------
    print_section("Table A: Alternative Asset TDA Comparison (paper v5 §2.1)")
    print("  Note: Only VIX standalone computed here (GLD/VIX ratios require separate CSV files)")

    vix_cnt, vix_ent, vix_eul = compute_tda_scale(dates, vix_v, W, "VIX standalone")
    vix_mat  = build_matrix(vix_cnt, vix_ent, vix_eul, filter_vix20, ret_map)
    vix_sp   = spread(vix_mat)[0]
    vix_spl  = split_spread(vix_cnt, vix_ent, vix_eul, filter_vix20, ret_map)
    vix_si   = vix_spl['IS'][0]
    vix_so   = vix_spl['OOS'][0]

    print(f"\n  {'Asset':28s}  {'Full':6s}  {'IS':6s}  {'OOS':6s}  OOS>IS")
    print("  " + "-" * 56)
    hyg_mark = "★" if tda_splits['OOS'][0] > tda_splits['IS'][0] else "↓"
    vix_mark = "★" if vix_so > vix_si else "↓"
    print(f"  {'HYG (this paper)':28s}  {tda_sp:5.2f}   {tda_splits['IS'][0]:5.2f}   {tda_splits['OOS'][0]:5.2f}  {hyg_mark}")
    print(f"  {'VIX standalone':28s}  {vix_sp:5.2f}   {vix_si:5.2f}   {vix_so:5.2f}  {vix_mark}")
    print()
    print("  Paper reference values:")
    print("    HYG:              full=16.07  IS= 6.89  OOS=18.63  *")
    print("    VIX standalone:   full= 7.88  IS= 3.69  OOS=10.68  *")

    # ----------------------------------------------------------------
    # Table B: Alternative H1 Indicators (paper v5 §2.3)
    # ----------------------------------------------------------------
    print_section("Table B: Alternative H1 Indicators (paper v5 §2.3)")

    cnt_map2, tp_map, ap_map, mp_map = compute_persistence_features(
        dates, hyg_v, "four persistence indicators"
    )

    # pairwise correlations between axes
    shared_b = sorted(set(cnt_map) & set(tp_map) & set(ap_map) & set(mp_map))
    if len(shared_b) > 10:
        cv  = np.array([cnt_map[d] for d in shared_b])
        tv  = np.array([tp_map[d]  for d in shared_b])
        apv = np.array([ap_map[d]  for d in shared_b])
        mv  = np.array([mp_map[d]  for d in shared_b])
        r_tp = float(np.corrcoef(cv, tv)[0, 1])
        r_ap = float(np.corrcoef(cv, apv)[0, 1])
        r_mp = float(np.corrcoef(cv, mv)[0, 1])
        print(f"  Corr. with H1 count: total pers.={r_tp:+.3f}  mean pers.={r_ap:+.3f}  max pers.={r_mp:+.3f}")
        print("  Paper values: total pers.=+0.587      mean pers.=+0.117      max pers.=+0.329")

    print(f"\n  {'1st Axis':20s}  {'Full':6s}  {'IS':6s}  {'OOS':6s}  OOS>IS")
    print("  " + "-" * 52)
    for lbl, m1 in [
        ("H1 count (paper)",    cnt_map),
        ("Total persistence",    tp_map),
        ("Mean persistence",      ap_map),
        ("Max persistence",        mp_map),
    ]:
        mat_ = build_matrix(m1, ent_map, euler_map, filter_vix20, ret_map)
        sp_  = spread(mat_)[0]
        spl_ = split_spread(m1, ent_map, euler_map, filter_vix20, ret_map)
        si_  = spl_['IS'][0]
        so_  = spl_['OOS'][0]
        mk   = "★" if so_ > si_ else "↓"
        print(f"  {lbl:20s}  {sp_:5.2f}   {si_:5.2f}   {so_:5.2f}  {mk}")
    print()
    print("  Paper reference values:")
    print("    H1 count:       full=16.07  IS= 6.89  OOS=18.63  *")
    print("    Total pers.:    full= 9.18  IS= 6.77  OOS=16.68  *")
    print("    Mean pers.:     full=11.64  IS= 7.10  OOS= 8.61  *")
    print("    Max pers.:      full=11.37  IS=10.52  OOS=10.51  -")

    # ----------------------------------------------------------------
    # Table C: Scalar TDA Indicators vs 3x3x3 (paper v5 §2.4)
    # ----------------------------------------------------------------
    print_section("Table C: Scalar TDA Indicators vs 3x3x3 (paper v5 §2.4)")

    # Wasserstein distance (day-over-day)
    ws_raw   = [None]
    prev_h1c = None
    for i in range(1, len(hyg_v)):
        seg = np.array(hyg_v[max(0, i - NEEDED + 1):i + 1])
        if len(seg) < NEEDED:
            ws_raw.append(None)
            prev_h1c = None
            continue
        X = np.array([[seg[j + k * DELAY] for k in range(D)] for j in range(W)],
                     dtype=np.float64)
        x_min, x_max = X.min(), X.max()
        if x_max - x_min < 1e-9:
            ws_raw.append(0.)
            continue
        X = (X - x_min) / (x_max - x_min)
        dgms = ripser(X, maxdim=1)['dgms']
        h1f  = dgms[1][dgms[1][:, 1] < np.inf]
        if prev_h1c is None:
            ws_raw.append(None)
        else:
            def _pers(h):
                return np.sort(h[:, 1] - h[:, 0]) if len(h) > 0 else np.array([0.])
            pa = _pers(h1f)
            pb = _pers(prev_h1c)
            ml = max(len(pa), len(pb))
            a  = np.pad(pa, (0, ml - len(pa)))
            b  = np.pad(pb, (0, ml - len(pb)))
            ws_raw.append(float(np.mean(np.abs(np.sort(a) - np.sort(b)))))
        prev_h1c = h1f

    ws_z   = rolling_zscore(ws_raw)
    ws_map = {dates[i]: ws_z[i] for i in range(len(dates)) if ws_z[i] is not None}

    print(f"  {'Indicator':28s}  {'IS':6s}  {'OOS':6s}  OOS>IS")
    print("  " + "-" * 48)
    print(f"  {'3x3x3 (this paper)':28s}  {tda_splits['IS'][0]:5.2f}   {tda_splits['OOS'][0]:5.2f}  ★")
    for slbl, smap in [
        ("Total pers. (scalar)",     tp_map),
        ("Max pers. (scalar)",       mp_map),
        ("Wasserstein dist. (scalar)", ws_map),
    ]:
        _, sp_is, _, sp_oos, _ = scalar_spread_is_oos(smap, filter_vix20, ret_map)
        mk = "★" if sp_oos > sp_is else "↓"
        print(f"  {slbl:28s}  {sp_is:5.2f}   {sp_oos:5.2f}  {mk}")
    print()
    print("  Paper reference values:")
    print("    3x3x3:                IS= 6.89  OOS=18.63  *")
    print("    Landscape L1 norm:    IS= 1.68  OOS= 1.85  *")
    print("    Total pers. (scalar): IS= 0.72  OOS= 1.26  *")
    print("    Wasserstein (scalar): IS= 0.30  OOS= 1.26  *")
    print("    Max pers. (scalar):   IS= 2.69  OOS= 1.44  -")

    # ----------------------------------------------------------------
    # Table D: Window Width W Sensitivity Analysis (paper v5 §2.5)
    # ----------------------------------------------------------------
    print_section("Table D: Window Width W Sensitivity (paper v5 §2.5)")

    cnt5,  ent5,  eul5  = compute_tda_scale(dates, hyg_v,  5, "W=5")
    cnt60, ent60, eul60 = compute_tda_scale(dates, hyg_v, 60, "W=60")

    # short-long difference (W=5 Z-score minus W=60 Z-score, then re-normalized)
    diff_raw = []
    for d in dates:
        v5  = cnt5.get(d)
        v60 = cnt60.get(d)
        diff_raw.append(v5 - v60 if (v5 is not None and v60 is not None) else None)
    diff_z   = rolling_zscore(diff_raw)
    diff_map = {dates[i]: diff_z[i] for i in range(len(dates)) if diff_z[i] is not None}

    print(f"\n  {'Setting':28s}  {'Full':6s}  {'IS':6s}  {'OOS':6s}  OOS>IS")
    print("  " + "-" * 56)
    for wlbl, wc, we, wu in [
        ("W=5  (short)",   cnt5,    ent5,    eul5),
        ("W=20 (paper)",  cnt_map, ent_map, euler_map),
        ("W=60 (long)",   cnt60,   ent60,   eul60),
    ]:
        mat_ = build_matrix(wc, we, wu, filter_vix20, ret_map)
        sp_  = spread(mat_)[0]
        spl_ = split_spread(wc, we, wu, filter_vix20, ret_map)
        si_  = spl_['IS'][0]
        so_  = spl_['OOS'][0]
        mk   = "★" if so_ > si_ else "↓"
        print(f"  {wlbl:28s}  {sp_:5.2f}   {si_:5.2f}   {so_:5.2f}  {mk}")

    # short-long diff x Ent20 x Euler20
    sl_mat  = build_matrix(diff_map, ent_map, euler_map, filter_vix20, ret_map)
    sl_sp   = spread(sl_mat)[0]
    sl_spl  = split_spread(diff_map, ent_map, euler_map, filter_vix20, ret_map)
    sl_si   = sl_spl['IS'][0]
    sl_so   = sl_spl['OOS'][0]
    sl_mk   = "★" if sl_so > sl_si else "↓"
    print(f"  {'short-long diff x Ent x Euler':28s}  {sl_sp:5.2f}   {sl_si:5.2f}   {sl_so:5.2f}  {sl_mk}")
    print()
    print("  Paper reference values:")
    print("    W=5:                       full= 2.99  IS= 5.58  OOS= 1.79  -")
    print("    W=20 (this paper):         full=16.07  IS= 6.89  OOS=18.63  *")
    print("    W=60:                      full= 9.32  IS= 3.59  OOS= 6.84  *")
    print("    short-long x Ent x Euler:  full=14.85  IS= 9.85  OOS=15.46  *")

    print_section("Table 5: Non-TDA Alternative Indicators (paper Table 6)")

    # HYG 20-day momentum
    hyg_mom = {}
    for i, d in enumerate(dates):
        if i >= 20:
            hyg_mom[d] = (hyg_v[i] - hyg_v[i - 20]) / hyg_v[i - 20] * 100
    hyg_mom_z_raw = rolling_zscore([hyg_mom.get(d) for d in dates])
    hyg_mom_z = {dates[i]: hyg_mom_z_raw[i]
                 for i in range(len(dates)) if hyg_mom_z_raw[i] is not None}

    # HYG 20-day realized volatility
    hyg_rets_all = [(hyg_v[i] - hyg_v[i-1]) / hyg_v[i-1] for i in range(1, len(hyg_v))]
    hyg_vol = {}
    for i in range(20, len(hyg_rets_all) + 1):
        hyg_vol[dates[i]] = statistics.stdev(hyg_rets_all[i-20:i]) * math.sqrt(252) * 100
        # annualized with sqrt(252); not stated explicitly in paper but does not affect conclusions
    hyg_vol_z_raw = rolling_zscore([hyg_vol.get(d) for d in dates])
    hyg_vol_z = {dates[i]: hyg_vol_z_raw[i]
                 for i in range(len(dates)) if hyg_vol_z_raw[i] is not None}

    # HYG/QQQ relative strength
    hyg_rel = {d: hyg_v[i] / qqq_v[i] for i, d in enumerate(dates)}
    hyg_rel_z_raw = rolling_zscore([hyg_rel.get(d) for d in dates])
    hyg_rel_z = {dates[i]: hyg_rel_z_raw[i]
                 for i in range(len(dates)) if hyg_rel_z_raw[i] is not None}

    def single_feat_spread(feat_map, filt, ret_map):
        mat = {'Low': [], 'Mid': [], 'High': []}
        for d, z in feat_map.items():
            if not filt(d): continue
            r = ret_map.get(d)
            if r is None: continue
            g = zone3(z)
            if g: mat[g].append(r)
        avgs = {k: statistics.mean(v) for k, v in mat.items() if len(v) >= MIN_N}
        sp = max(avgs.values()) - min(avgs.values()) if len(avgs) >= 2 else 0
        # OOS
        mat_oos = {'Low': [], 'Mid': [], 'High': []}
        for d, z in feat_map.items():
            if not filt(d) or d < '2020-01-01': continue
            r = ret_map.get(d)
            if r is None: continue
            g = zone3(z)
            if g: mat_oos[g].append(r)
        avgs_oos = {k: statistics.mean(v) for k, v in mat_oos.items() if len(v) >= MIN_N}
        sp_oos = max(avgs_oos.values()) - min(avgs_oos.values()) if len(avgs_oos) >= 2 else 0
        return sp, sp_oos

    tda_mat  = build_matrix(cnt_map, ent_map, euler_map, filter_vix20, ret_map)
    tda_sp   = spread(tda_mat)[0]
    tda_splits = split_spread(cnt_map, ent_map, euler_map, filter_vix20, ret_map)
    tda_oos  = tda_splits['OOS'][0]

    print(f"\n  {'Indicator':30s}  {'Full spread':13s}  {'OOS spread':12s}  {'vs TDA'}")
    print("  " + "-"*65)
    alt_results = [
        ("HYG TDA 3x3x3 (paper)",  tda_sp,   tda_oos),
    ]
    for fname, fmap in [
        ("HYG 20-day momentum",    hyg_mom_z),
        ("HYG 20-day volatility",  hyg_vol_z),
        ("HYG/QQQ relative str.",  hyg_rel_z),
    ]:
        sp_a, sp_oos_a = single_feat_spread(fmap, filter_vix20, ret_map)
        alt_results.append((fname, sp_a, sp_oos_a))

    for name, sp_a, sp_oos_a in alt_results:
        ratio_str = "—" if name.startswith("HYG TDA") else f"{sp_oos_a/tda_oos*100:.0f}%"
        print(f"  {name:30s}  {sp_a:6.2f}pp          "
              f"{sp_oos_a:6.2f}pp       {ratio_str}")

    # --- Final summary ---
    print_section("Final Summary (key paper results)")
    print(f"""
  Framework: HYG TDA 3x3x3 (W=20, d=3, tau=3, epsilon=0.5)
  Filter:    VIX_SMA10 >= 20 (10-day simple moving average)

  Predictive spread (VIX_SMA10>=20): {tda_sp:.2f} pp
  IS spread (thru 2019):             {tda_splits['IS'][0]:.2f} pp  (n={tda_splits['IS'][1]})
  OOS spread (2020+):                {tda_splits['OOS'][0]:.2f} pp  (n={tda_splits['OOS'][1]})
  OOS / IS ratio:                    {tda_splits['OOS'][0]/tda_splits['IS'][0]:.2f} x

  Bootstrap OOS p-value: see Table 4 above
  Non-TDA alternatives max OOS: see Table 5 above (~13-18% of this paper)
    """)

    print("Reproduction complete.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Paper results reproduction script (Takagi 2026)'
    )
    parser.add_argument('--hyg', default='longterm_hyg.csv',
                        help='path to HYG CSV file')
    parser.add_argument('--qqq', default='longterm_qqq.csv',
                        help='path to QQQ CSV file')
    parser.add_argument('--vix', default='longterm_vix.csv',
                        help='path to VIX CSV file')
    args = parser.parse_args()
    main(args.hyg, args.qqq, args.vix)
