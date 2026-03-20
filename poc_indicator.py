"""
poc_indicator.py — расчёт POC (Point of Control) 1:1 из фапвапва.py
"""

import numpy as np
import pandas as pd

POC_TF_MIN = {
    "4H": 240,
    "1D": 1440,
    "1W": 10080,
    "1M": 43200,
    "1Y": 525600,
}

TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15,
    "30m": 30, "1h": 60, "2h": 120, "4h": 240,
    "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200,
}

POC_COLORS = {
    "4H": "#1f77b4",
    "1D": "#fc7601",
    "1W": "#298b29",
    "1M": "#d62728",
    "1Y": "#9467bd",
}

FREQ_MAP = {
    "4H": "4h",
    "1D": "D",
    "1W": "W",
    "1M": "ME",
    "1Y": "YE",
}


def auto_incr(df: pd.DataFrame) -> float:
    avg_range = (df["high"] - df["low"]).mean()
    last_close = df["close"].iloc[-1]
    mintick = max(last_close * 1e-5, 0.01)
    wn = avg_range / mintick
    if wn <= 0:
        return mintick
    pow_ = 10 ** int(np.log10(wn))
    step = 2 * np.ceil(wn / pow_) * mintick * pow_
    return round(step, 10)


def round_to_level(v, step):
    return round(round(v / step) * step, 10)


def calc_poc_for_period(period_df: pd.DataFrame, step: float) -> float:
    if period_df.empty or step <= 0:
        return np.nan

    lo_all = period_df["low"].min()
    hi_all = period_df["high"].max()
    lb = round_to_level(lo_all, step)
    ub = round_to_level(hi_all, step)

    if ub <= lb:
        return (lb + ub) / 2

    levels = np.arange(lb, ub + step, step)
    volumes = np.zeros(len(levels))

    for _, row in period_df.iterrows():
        lo, hi, vol = row["low"], row["high"], row["volume"]
        v_lo = max(0, int(np.floor((lo - lb) / step)))
        v_hi = min(len(levels) - 1, int(np.floor((hi - lb) / step)))
        tks = max(1, v_hi - v_lo)
        v_per_bin = vol / tks
        for i in range(v_lo, max(v_hi, v_lo + 1)):
            if i < len(volumes):
                volumes[i] += v_per_bin

    if volumes.max() == 0:
        return levels[len(levels) // 2]

    return float(levels[int(np.argmax(volumes))])


def calc_poc_series(base_df: pd.DataFrame,
                    poc_label: str,
                    step: float,
                    base_tf: str) -> pd.Series:
    """
    Возвращает Series с POC для каждого бара base_df.
    poc_label: "4H", "1D", "1W", "1M", "1Y"
    """
    tf_min  = POC_TF_MIN.get(poc_label, 999999)
    cur_min = TF_MINUTES.get(base_tf, 1)

    # Не показываем если текущий TF >= POC TF
    if cur_min >= tf_min:
        return pd.Series(np.nan, index=base_df.index)

    freq   = FREQ_MAP[poc_label]
    result = pd.Series(np.nan, index=base_df.index)

    grouped = base_df.resample(freq, label="left", closed="left")
    for period_start, group in grouped:
        if group.empty:
            continue
        poc_val = calc_poc_for_period(group, step)
        result.loc[group.index] = poc_val

    return result


def calc_all_pocs(base_df: pd.DataFrame,
                  base_tf: str,
                  show: dict) -> dict:
    """
    show = {"4H": True, "1D": True, "1W": True, "1M": False, "1Y": False}
    Возвращает dict label → Series
    """
    if base_df.empty:
        return {}

    step = auto_incr(base_df)
    result = {}

    for label, enabled in show.items():
        if not enabled:
            continue
        series = calc_poc_series(base_df, label, step, base_tf)
        if not series.dropna().empty:
            result[label] = series

    return result


def pocs_to_json(pocs: dict, max_display: int = 500) -> list:
    """
    Конвертирует POC серии в список сегментов для фронта.
    Каждый сегмент: {label, color, x0, x1, y}
    Ограничиваем до max_display последних баров.
    """
    segments = []

    for label, series in pocs.items():
        color = POC_COLORS.get(label, "#ffffff")
        # Берём только последние max_display баров
        s = series.iloc[-max_display:] if len(series) > max_display else series

        prev_val  = None
        seg_start = None

        for ts, val in s.items():
            if np.isnan(val):
                if seg_start is not None:
                    segments.append({
                        "label": label,
                        "color": color,
                        "x0":    int(seg_start.timestamp()),
                        "x1":    int(ts.timestamp()),
                        "y":     prev_val,
                    })
                    seg_start = None
                    prev_val  = None
                continue

            if val != prev_val:
                if seg_start is not None:
                    segments.append({
                        "label": label,
                        "color": color,
                        "x0":    int(seg_start.timestamp()),
                        "x1":    int(ts.timestamp()),
                        "y":     prev_val,
                    })
                seg_start = ts
                prev_val  = val

        if seg_start is not None and prev_val is not None:
            segments.append({
                "label": label,
                "color": color,
                "x0":    int(seg_start.timestamp()),
                "x1":    int(s.index[-1].timestamp()),
                "y":     prev_val,
            })

    return segments