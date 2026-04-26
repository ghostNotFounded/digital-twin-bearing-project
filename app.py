import os
import glob
import re
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import savgol_filter
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from flask import Flask, jsonify, render_template, request, abort

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "BearingData")
FS = 97656          # vibration sampling frequency (Hz)
PPR = 2             # tach pulses-per-revolution
SUBSAMPLE = 4096    # points to send to frontend (downsampled for speed)

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_mat_files():
    """Return sorted list of .mat file paths."""
    files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    def _ts(p):
        m = re.search(r"(\d{8}T\d{6}Z)", os.path.basename(p))
        return pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%SZ") if m else pd.Timestamp(0)
    return sorted(files, key=_ts)


def parse_timestamp(filepath):
    m = re.search(r"(\d{8}T\d{6}Z)", os.path.basename(filepath))
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%SZ").isoformat()
    return None

def estimate_rpm(tach, fs=FS, ppr=PPR):
    """Estimate mean RPM from tachometer signal."""
    threshold = np.max(np.abs(tach)) * 0.5
    crossings = np.where((tach[:-1] < threshold) & (tach[1:] >= threshold))[0]
    if len(crossings) < 2:
        return None
    rev_periods = np.diff(crossings) / fs          # seconds per pulse
    rev_times   = rev_periods * ppr                 # seconds per revolution
    rpm = 60.0 / np.mean(rev_times)
    return float(rpm)


def time_features(signal):
    v = signal
    rms      = float(np.sqrt(np.mean(v**2)))
    peak     = float(np.max(np.abs(v)))
    mean_abs = float(np.mean(np.abs(v)))
    kurt     = float(scipy_kurtosis(v, fisher=True))   # excess kurtosis
    crest    = peak / rms if rms else 0
    impulse  = peak / mean_abs if mean_abs else 0
    skew     = float(pd.Series(v).skew())
    std      = float(np.std(v))
    return dict(rms=rms, peak=peak, mean_abs=mean_abs,
                kurtosis=kurt, crest_factor=crest,
                impulse_factor=impulse, skewness=skew, std=std)


def freq_features(signal, fs, rpm=None):
    N   = len(signal)
    win = np.hanning(N)
    fft = np.fft.rfft(signal * win)
    mag = np.abs(fft) * 2 / N
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # Broadband energy bands
    bands = {
        "0-1kHz":   (0,    1000),
        "1-5kHz":   (1000, 5000),
        "5-10kHz":  (5000, 10000),
        "10-20kHz": (10000, 20000),
    }
    band_energy = {}
    for name, (lo, hi) in bands.items():
        idx = (freqs >= lo) & (freqs < hi)
        band_energy[name] = float(np.sum(mag[idx]**2))

    result = {"band_energy": band_energy}

    # Return downsampled spectrum for plotting (max 2048 points)
    step = max(1, len(freqs) // 2048)
    result["spectrum_freqs"] = freqs[::step].tolist()
    result["spectrum_mag"]   = mag[::step].tolist()
    return result


# ── Pre-compute global Health Index across all days ──────────────────────────

def _build_hi_series():
    files = get_mat_files()
    if not files:
        return {}, []

    rows = []
    for fp in files:
        try:
            mat  = loadmat(fp)
            vib  = mat["vibration"].flatten().astype(float)
            tach = mat["tach"].flatten().astype(float)
            ts   = parse_timestamp(fp)

            def signal_features(v, prefix):
                mean      = np.mean(v)
                std       = np.std(v)
                rms       = np.sqrt(np.mean(v**2))
                peak      = np.max(v)
                peak2peak = np.max(v) - np.min(v)
                energy    = np.sum(v**2)
                skewness  = np.mean((v - mean)**3) / (std**3 + 1e-9)
                kurtosis  = np.mean((v - mean)**4) / (std**4 + 1e-9)
                crest     = peak / (rms + 1e-9)
                return {
                    f"{prefix}_mean":         mean,
                    f"{prefix}_std":          std,
                    f"{prefix}_skewness":     skewness,
                    f"{prefix}_kurtosis":     kurtosis,
                    f"{prefix}_peak2peak":    peak2peak,
                    f"{prefix}_rms":          rms,
                    f"{prefix}_crest_factor": crest,
                    f"{prefix}_energy":       energy,
                }

            row = {}
            row.update(signal_features(vib,  "vib"))
            row.update(signal_features(tach, "tach"))
            row["filepath"]  = fp
            row["timestamp"] = ts
            rows.append(row)
        except Exception:
            continue

    if not rows:
        return {}, []

    df = pd.DataFrame(rows).set_index("timestamp")
    df.sort_index(inplace=True)

    all_feat_cols = [c for c in df.columns if c not in ("filepath",)]
    features = df[all_feat_cols].copy()

    # ── Step 1: Smooth with causal rolling mean (window=5) ──
    features_smooth = features.copy()
    for col in features.columns:
        features_smooth[col] = features[col].rolling(window=5, min_periods=1).mean()

    # ── Step 2: Monotonicity filter on vib_* features only ──
    def monotonicity(series):
        d = np.diff(series)
        return abs((d > 0).sum() - (d < 0).sum()) / len(d)

    vib_cols = [c for c in features_smooth.columns if c.startswith("vib")]
    mono_scores = {c: monotonicity(features_smooth[c].values) for c in vib_cols}

    threshold = 0.1
    feature_cols = [c for c, s in mono_scores.items() if s > threshold]
    if not feature_cols:
        feature_cols = vib_cols  # fallback: use all vib features

    # ── Step 3: Normalise using training window mean/std only ──
    N_TRAIN = max(2, int(len(features) * 0.4))
    X_all   = features_smooth[feature_cols].copy()
    X_train = X_all.iloc[:N_TRAIN]

    mean_train = X_train.mean()
    std_train  = X_train.std().replace(0, 1)  # guard zero-std columns

    X_all_norm   = (X_all   - mean_train) / std_train
    X_train_norm = (X_train - mean_train) / std_train

    # ── Step 4: Fit PCA on training data, project all data onto PC1 ──
    pca = PCA()
    pca.fit(X_train_norm)
    score = X_all_norm.values @ pca.components_[0]  # PC1 scores for all samples

    # ── Step 5: Causal rolling mean on PC1 score ──
    score_smooth = (
        pd.Series(score, index=features.index)
        .rolling(window=5, min_periods=1)
        .mean()
        .values
    )

    # ── Step 6: Flip if PC1 decreases over time (should increase toward failure) ──
    if np.corrcoef(np.arange(len(score_smooth)), score_smooth)[0, 1] < 0:
        score_smooth = -score_smooth

    # ── Step 7: Shift so day-1 = 0, then normalise against final score ──
    health_indicator = score_smooth - score_smooth[0]
    failure_score    = health_indicator[-1]

    if abs(failure_score) < 1e-9:
        hi_norm = np.ones(len(health_indicator))
    else:
        hi_norm = 1.0 - np.clip(health_indicator / failure_score, 0, 1)

    # ── Build outputs ──
    filepaths = df["filepath"].values
    hi_map = {fp: float(hi_norm[i]) for i, fp in enumerate(filepaths)}
    timeline = [
        {"timestamp": df.index[i], "hi": float(hi_norm[i])}
        for i in range(len(hi_norm))
    ]
    return hi_map, timeline


HI_MAP, HI_TIMELINE = _build_hi_series()

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data")
def data_page():
    files = get_mat_files()
    days  = []
    for i, fp in enumerate(files):
        ts = parse_timestamp(fp)
        days.append({
            "index": i,
            "label": f"Day {i+1}",
            "timestamp": ts,
            "filename": os.path.basename(fp),
        })
    return render_template("data.html", days=days, total_days=len(days))


@app.route("/api/days")
def api_days():
    files = get_mat_files()
    days  = []
    for i, fp in enumerate(files):
        ts = parse_timestamp(fp)
        days.append({
            "index": i,
            "label": f"Day {i+1}",
            "timestamp": ts,
            "filename": os.path.basename(fp),
            "hi": HI_MAP.get(fp),
        })
    return jsonify(days)


def compute_rul(hi_values, current_idx):
    actual_current = hi_values[current_idx]
    vals = hi_values[:current_idx + 1]
    n = len(vals)

    if n < 3:
        vals = hi_values
        n = len(vals)
    if n < 3:
        return None

    tail_n = max(3, int(n * 0.3))
    x = np.arange(tail_n)
    y = vals[-tail_n:]

    if np.std(y) < 1e-6:
        return None

    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]

    # HI now decreases toward 0, so slope must be negative
    if slope >= 0:
        return None

    # Extrapolate to HI = 0 (failure)
    steps_left = actual_current / abs(slope)
    return max(0, round(float(steps_left)))

@app.route("/api/data")
def api_data():
    day_idx = request.args.get("day", 0, type=int)
    files   = get_mat_files()
    if not files or day_idx >= len(files):
        abort(404)

    fp  = files[day_idx]
    mat = loadmat(fp)
    vib = mat["vibration"].flatten().astype(float)
    tac = mat["tach"].flatten().astype(float)

    rpm = estimate_rpm(tac, FS, PPR)
    tf  = time_features(vib)
    ff  = freq_features(vib, FS, rpm)

    # Downsample time-series for plot
    step = max(1, len(vib) // SUBSAMPLE)
    t    = (np.arange(len(vib)) / FS)[::step].tolist()
    v_ds = vib[::step].tolist()

    step_t = max(1, len(tac) // SUBSAMPLE)
    t_t    = (np.arange(len(tac)) / FS)[::step_t].tolist()
    tac_ds = tac[::step_t].tolist()

    hi_values = [p["hi"] for p in HI_TIMELINE]
    rul = compute_rul(hi_values, day_idx)

    return jsonify({
        "day_index": day_idx,
        "timestamp": parse_timestamp(fp),
        "filename":  os.path.basename(fp),
        "rpm":       rpm,
        "rul_days":  rul,
        "time_series": {"t": t, "vibration": v_ds},
        "tach_series": {"t": t_t, "tach": tac_ds},
        "time_features": tf,
        "freq_features": ff,
        "hi": HI_MAP.get(fp),
    })


@app.route("/api/health-index")
def api_health_index():
    day_idx = request.args.get("day", 0, type=int)
    print(day_idx)

    files = get_mat_files()
    if not files:
        return jsonify({"error": "No data files found"}), 404

    # Current (latest) day
    latest_fp = files[day_idx - 1]
    current_hi = HI_MAP.get(latest_fp, None)

    # Determine status label
    if current_hi is None:
        status = "unknown"
    elif current_hi > 0.7:
        status = "healthy"
    elif current_hi > 0.4:
        status = "degrading"
    elif current_hi > 0.15:
        status = "warning"
    else:
        status = "critical"

    # Simple RUL: fit linear trend on last 30% of HI, extrapolate to HI=1
    rul_days = None
    if HI_TIMELINE:
        vals = [p["hi"] for p in HI_TIMELINE]
        n    = len(vals)
        tail_n = max(3, int(n * 0.3))
        x    = np.arange(tail_n)
        y    = vals[-tail_n:]
        if np.std(y) > 1e-6:
            coeffs  = np.polyfit(x, y, 1)
            slope   = coeffs[0]
            current = vals[-1]
            if slope > 0:
                steps_left = (1.0 - current) / slope
                rul_days   = max(0, round(float(steps_left)))

    return jsonify({
        "current_hi":    current_hi,
        "status":        status,
        "rul_days":      rul_days,
        "total_days":    len(files),
        "timeline":      HI_TIMELINE,
        "latest_timestamp": parse_timestamp(latest_fp),
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)