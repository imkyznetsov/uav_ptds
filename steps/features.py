import numpy as np
import pandas as pd
import scipy.signal as sps

def band_energy(f, Pxx, fmin, fmax):
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        return 0.0
    return float(np.trapz(Pxx[m], f[m]))

def spec_entropy(Pxx):
    p = Pxx / (np.sum(Pxx)+1e-12)
    return float(-np.sum(p*np.log(p+1e-12)))

def window_feature_frame(win: pd.DataFrame, fs: int) -> dict:
    feats = {}
    axes = ["ax","ay","az","p","q","r","mag_x","mag_y","mag_z"]
    for a in axes:
        x = win[a].values
        feats[f"{a}_mean"] = float(np.mean(x))
        feats[f"{a}_std"] = float(np.std(x))
        feats[f"{a}_rms"] = float(np.sqrt(np.mean(x**2)))
        feats[f"{a}_min"] = float(np.min(x)); feats[f"{a}_max"] = float(np.max(x))
        x0 = x - np.mean(x)
        if len(x0) > 1:
            ac1 = np.correlate(x0[:-1], x0[1:])[0] / (np.sum(x0[:-1]**2)+1e-12)
        else:
            ac1 = 0.0
        feats[f"{a}_ac1"] = float(ac1)
        f, Pxx = sps.welch(x, fs=fs, nperseg=min(256, len(x)))
        feats[f"{a}_se"] = spec_entropy(Pxx)
        feats[f"{a}_b0_2"] = band_energy(f, Pxx, 0, 2)
        feats[f"{a}_b2_10"] = band_energy(f, Pxx, 2, 10)
        feats[f"{a}_b10_50"] = band_energy(f, Pxx, 10, 50)
        feats[f"{a}_fdom"] = float(f[np.argmax(Pxx)] if len(Pxx)>0 else 0.0)
        feats[f"{a}_spike_ratio"] = float(np.mean(np.abs(x - np.mean(x)) > 3*np.std(x)+1e-12))
    
    feats["alt_baro_gnss_diff_mean"] = float((win["baro_alt"] - win["gnss_alt"]).mean())
    feats["spd_gnss_vs_inertial_std"] = float((win["gnss_speed"] - win["speed"]).std())
    
    for c in ["altitude","speed","temperature","pitch","roll","heading","baro_alt","gnss_alt","gnss_speed"]:
        feats[f"{c}_mean"] = float(win[c].mean())
        feats[f"{c}_std"] = float(win[c].std())
    
    feats["y"] = int(win["label"].max() > 0)
    feats["failure_type"] = win["failure_type"].iloc[0]
    return feats

class FeatureBuilder:
    def __init__(self, fs=100, window_s=5, overlap=0.5, out_path="data/processed/features.parquet"):
        self.fs = fs; self.window_s = window_s; self.overlap = overlap; self.out_path = out_path

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        size = int(self.window_s*self.fs)
        step = max(1, int(size*(1-self.overlap)))
        rows = []
        for fid, dfg in df.groupby("flight_id"):
            for start in range(0, len(dfg)-size+1, step):
                win = dfg.iloc[start:start+size]
                feats = window_feature_frame(win, self.fs)
                feats["flight_id"] = int(fid)
                feats["t_start"] = float(win["t"].iloc[0])
                rows.append(feats)
        feats_df = pd.DataFrame(rows)
        feats_df.to_parquet(self.out_path, index=False)
        return feats_df