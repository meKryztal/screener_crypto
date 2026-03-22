"""
indicator.py — логика 1:1 из оригинального Pine скрипта
ОПТИМИЗАЦИЯ: векторизация через NumPy/Pandas, убраны iterrows()
- На 1m TF: проверяет текущую свечу (mid в тени тела)
- На HTF: из всех 1m баров внутри HTF свечи берёт один с макс объёмом
- Пузырь рисуется на координате HTF свечи (как в Pine plotshape)
"""

import pandas as pd
import numpy as np

LTF = "1m"

TF_MS = {
    "1m":  60_000,    "3m":  180_000,   "5m":  300_000,
    "15m": 900_000,   "30m": 1_800_000, "1h":  3_600_000,
    "2h":  7_200_000, "4h":  14_400_000,"1d":  86_400_000,
    "1w":  604_800_000,
}

TF_PANDAS = {
    "1m":  "1min",  "3m":  "3min",  "5m":  "5min",
    "15m": "15min", "30m": "30min", "1h":  "1h",
    "2h":  "2h",    "4h":  "4h",    "1d":  "1D",
    "1w":  "1W",
}


def _get_thresh(thresh_series: pd.Series, ts) -> float:
    try:
        val = thresh_series.loc[ts]
        return float(val) if not pd.isna(val) else np.nan
    except (KeyError, TypeError):
        idx = thresh_series.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return np.nan
        val = thresh_series.iloc[idx]
        return float(val) if not pd.isna(val) else np.nan


def _calc_absorption_vec(df: pd.DataFrame, thresh_series: pd.Series) -> pd.DataFrame:
    """
    Векторизованный расчёт absorption для same-TF (1m) случая.
    """
    thresh_aligned = thresh_series.reindex(df.index, method="ffill")

    h   = df["high"].values
    l   = df["low"].values
    o   = df["open"].values
    c   = df["close"].values
    vol = df["volume"].values
    thr = thresh_aligned.values

    mid = (h + l) / 2.0
    top = np.maximum(o, c)
    bot = np.minimum(o, c)

    v_up = (mid >= top) & (mid <= h)
    v_dn = (mid <= bot) & (mid >= l)
    in_shadow = v_up | v_dn

    valid = (
        ~np.isnan(thr) & (thr > 0) &
        ~np.isnan(vol) & (vol >= thr) &
        in_shadow
    )

    if not valid.any():
        return _empty()

    idx_valid = np.where(valid)[0]
    ratio     = vol[idx_valid] / thr[idx_valid]
    size_cat  = np.ones(len(idx_valid), dtype=int)
    size_cat[ratio >= 2] = 2
    size_cat[ratio >= 3] = 3
    size_cat[ratio >= 4] = 4

    results = []
    for i, orig_i in enumerate(idx_valid):
        ts = df.index[orig_i]
        results.append({
            "ts":       ts,
            "base_ts":  ts,
            "mid":      float(mid[orig_i]),
            "is_up":    bool(v_up[orig_i]),
            "is_dn":    bool(v_dn[orig_i]),
            "vol":      float(vol[orig_i]),
            "size_cat": int(size_cat[i]),
            "thresh":   float(thr[orig_i]),
        })

    df_out = pd.DataFrame(results).set_index("ts")
    df_out.sort_index(inplace=True)
    return df_out


def _calc_htf_absorption(base_df: pd.DataFrame, ltf_df: pd.DataFrame,
                          thresh_series: pd.Series, freq: str) -> pd.DataFrame:
    """
    Оптимизированный HTF расчёт absorption.
    Для каждой HTF свечи — лучший 1m бар (макс объём с mid в тени).
    Внутренний Python-цикл заменён на векторный argmax по маске.
    """
    h   = ltf_df["high"].values
    l   = ltf_df["low"].values
    o   = ltf_df["open"].values
    c   = ltf_df["close"].values
    vol = ltf_df["volume"].values

    mid = (h + l) / 2.0
    top = np.maximum(o, c)
    bot = np.minimum(o, c)
    v_up      = (mid >= top) & (mid <= h)
    v_dn      = (mid <= bot) & (mid >= l)
    in_shadow = v_up | v_dn

    base_index_set = set(base_df.index)

    grouped = ltf_df.groupby(
        pd.Grouper(freq=freq, label="left", closed="left")
    )

    results = []
    for base_ts, group in grouped:
        if group.empty:
            continue
        if base_ts not in base_index_set:
            continue

        thresh = _get_thresh(thresh_series, base_ts)
        if np.isnan(thresh) or thresh <= 0:
            continue

        # Индексы этой группы в ltf_df — векторно
        group_indices = ltf_df.index.searchsorted(group.index)
        group_indices = group_indices[group_indices < len(vol)]

        if len(group_indices) == 0:
            continue

        g_vol      = vol[group_indices]
        g_shadow   = in_shadow[group_indices]
        valid_mask = g_shadow & ~np.isnan(g_vol) & (g_vol >= thresh)

        if not valid_mask.any():
            continue

        # Находим argmax объёма среди валидных баров без Python-цикла
        masked_vol = np.where(valid_mask, g_vol, 0.0)
        best_local = int(np.argmax(masked_vol))
        best_gi    = group_indices[best_local]
        best_v     = float(vol[best_gi])

        ratio    = best_v / thresh
        size_cat = 4 if ratio >= 4 else 3 if ratio >= 3 else 2 if ratio >= 2 else 1

        results.append({
            "ts":       base_ts,
            "base_ts":  base_ts,
            "mid":      float(mid[best_gi]),
            "is_up":    bool(v_up[best_gi]),
            "is_dn":    bool(v_dn[best_gi]),
            "vol":      best_v,
            "size_cat": int(size_cat),
            "thresh":   float(thresh),
        })

    if not results:
        return _empty()

    df_out = pd.DataFrame(results).set_index("ts")
    df_out.sort_index(inplace=True)
    return df_out


def calc_absorption(base_df:    pd.DataFrame,
                    ltf_df:     pd.DataFrame,
                    base_tf:    str   = "5m",
                    mode:       str   = "Auto",
                    manual_vol: float = 100.0,
                    percentile: float = 99.0,
                    lookback:   int   = 4900) -> pd.DataFrame:
    """
    Логика 1:1 из Pine скрипта, оптимизирована через векторизацию.
    """

    if base_df is None or base_df.empty:
        return _empty()

    is_same_tf = (base_tf == LTF)

    # ── Порог объёма
    if mode == "Auto":
        if is_same_tf or ltf_df is None or ltf_df.empty:
            vol_for_thresh = base_df["volume"]
        else:
            vol_for_thresh = ltf_df["volume"]

        thresh_series = (
            vol_for_thresh
            .shift(1)
            .rolling(window=lookback, min_periods=10)
            .quantile(percentile / 100)
        )
    else:
        if is_same_tf or ltf_df is None or ltf_df.empty:
            thresh_series = pd.Series(manual_vol, index=base_df.index)
        else:
            thresh_series = pd.Series(manual_vol, index=ltf_df.index)

    # ── Same TF (1m) — векторизованный расчёт
    if is_same_tf:
        return _calc_absorption_vec(base_df, thresh_series)

    # ── HTF — оптимизированный расчёт
    if ltf_df is None or ltf_df.empty:
        return _empty()

    freq = TF_PANDAS.get(base_tf, "5min")
    return _calc_htf_absorption(base_df, ltf_df, thresh_series, freq)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["base_ts","mid","is_up","is_dn","vol","size_cat","thresh"]
    )


def absorption_to_json(df: pd.DataFrame) -> list:
    """
    Оптимизировано: убран iterrows(), заменён на векторные операции.
    """
    if df.empty:
        return []

    ts_ms   = (df.index.astype(np.int64) // 1_000_000).tolist()
    base_ms = (df["base_ts"].values.astype(np.int64) // 1_000_000).tolist()

    return [
        {
            "ts":       t,
            "base_ts":  b,
            "mid":      float(m),
            "is_up":    bool(u),
            "is_dn":    bool(d),
            "vol":      float(v),
            "size_cat": int(s),
            "thresh":   float(th),
        }
        for t, b, m, u, d, v, s, th in zip(
            ts_ms,
            base_ms,
            df["mid"].tolist(),
            df["is_up"].tolist(),
            df["is_dn"].tolist(),
            df["vol"].tolist(),
            df["size_cat"].tolist(),
            df["thresh"].tolist(),
        )
    ]


def ohlcv_to_json(df: pd.DataFrame) -> list:
    """
    Оптимизировано: убран iterrows(), заменён на векторные операции.
    """
    if df is None or df.empty:
        return []

    ts_sec = (df.index.astype(np.int64) // 1_000_000_000).tolist()

    return [
        {
            "time":   t,
            "open":   float(o),
            "high":   float(h),
            "low":    float(l),
            "close":  float(c),
            "volume": float(v),
        }
        for t, o, h, l, c, v in zip(
            ts_sec,
            df["open"].tolist(),
            df["high"].tolist(),
            df["low"].tolist(),
            df["close"].tolist(),
            df["volume"].tolist(),
        )
    ]