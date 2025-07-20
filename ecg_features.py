# ai_ecg/features.py

import json
import numpy as np
import pandas as pd
import neurokit2 as nk


def extract_ecg_features(
    lead_I: np.ndarray,
    lead_II: np.ndarray,
    sampling_rate: float = 250.0
) -> dict:
    """
    From two 1-D NumPy arrays (Lead I & Lead II) of a single 10s ECG, compute:
      • median heart rate (bpm)
      • median QRS duration (ms)
      • median QT interval (ms)
      • frontal T‐wave axis (degrees)

    Parameters
    ----------
    lead_I : np.ndarray
        1-D array of samples from Lead I (length 2500).
    lead_II : np.ndarray
        1-D array of samples from Lead II (length 2500).
    sampling_rate : float
        Sampling frequency in Hz (default: 250).

    Returns
    -------
    feats : dict
        {
          "HeartRate_bpm": float,
          "QRS_ms":        float,
          "QT_ms":         float,
          "T_Axis_deg":    float
        }
    """
    # 1) Clean and detect R-peaks on Lead II
    clean = nk.ecg_clean(lead_II, sampling_rate=sampling_rate)
    peaks, _ = nk.ecg_peaks(clean, sampling_rate=sampling_rate)

    # 2) Delineate QRS and T waves
    delineate = nk.ecg_delineate(
        clean,
        peaks,
        sampling_rate=sampling_rate,
        method="dwt"
    )

    Q_on  = delineate["ECG_QRS_Onsets"]
    Q_off = delineate["ECG_QRS_Offsets"]
    T_off = delineate["ECG_T_Offsets"]
    T_pk  = delineate["ECG_T_Peaks"]

    # Helper: samples → milliseconds
    to_ms = lambda arr: (arr / sampling_rate) * 1000

    # 3) Heart‐rate series → median BPM
    hr_series = nk.ecg_rate(peaks, sampling_rate=sampling_rate)
    hr_bpm    = float(np.median(hr_series)) if len(hr_series) > 0 else np.nan

    # 4) Durations → median
    qrs_ms = float(np.median(to_ms(Q_off - Q_on))) if len(Q_on)>0 and len(Q_off)>0 else np.nan
    qt_ms  = float(np.median(to_ms(T_off - Q_on))) if len(Q_on)>0 and len(T_off)>0 else np.nan

    # 5) T-axis: from the first T-peak vector in (I,II) plane
    if len(T_pk) > 0:
        idx = T_pk[0]
        v1, v2 = lead_I[idx], lead_II[idx]
        theta = np.degrees(np.arctan2(v2, v1))
        if theta < 0:
            theta += 360
    else:
        theta = np.nan

    return {
        "HeartRate_bpm": hr_bpm,
        "QRS_ms":       qrs_ms,
        "QT_ms":        qt_ms,
        "T_Axis_deg":   theta,
    }


def append_ecg_features(
    df: pd.DataFrame,
    lead_order: list[str] = ["I","II","V1","V2","V3","V4","V5","V6"],
    sampling_rate: float = 250.0
) -> pd.DataFrame:
    """
    Given a DataFrame with JSON‐encoded Lead_<lead> columns,
    decode leads I & II and compute interval features for every row.
    Returns a new DataFrame with four extra columns:
      ['HeartRate_bpm','QRS_ms','QT_ms','T_Axis_deg']

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns Lead_I and Lead_II (JSON lists).
    lead_order : list of str
        Lead names in the DataFrame (default 8 leads).
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    df2 : pd.DataFrame
        A copy of `df` with new interval columns appended.
    """
    feats_list = []
    for _, row in df.iterrows():
        # parse Lead I & II from JSON
        try:
            lead_I  = np.array(json.loads(row[f"Lead_{lead_order[0]}"]), dtype=float)
            lead_II = np.array(json.loads(row[f"Lead_{lead_order[1]}"]), dtype=float)
        except Exception:
            # in case of parse error, fill with NaNs
            feats_list.append({
                "HeartRate_bpm": np.nan,
                "QRS_ms":        np.nan,
                "QT_ms":         np.nan,
                "T_Axis_deg":    np.nan,
            })
            continue

        feats = extract_ecg_features(lead_I, lead_II, sampling_rate=sampling_rate)
        feats_list.append(feats)

    feat_df = pd.DataFrame(feats_list, index=df.index)
    return pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
