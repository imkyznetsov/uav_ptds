import pandas as pd
import numpy as np
import os

class Cleaner:
    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def concat_raw(self, raw_dir: str) -> pd.DataFrame:
        files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".parquet")]
        dfs = [pd.read_parquet(p) for p in files]
        return pd.concat(dfs, ignore_index=True)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Заполним редкие NaN GNSS методом ffill
        for c in ["gnss_alt", "gnss_speed"]:
            if c in df.columns:
                df[c] = df[c].interpolate().fillna(method="bfill").fillna(method="ffill")
        # Ограничим экстремумы
        for c in ["ax","ay","az","p","q","r","baro_alt","mag_x","mag_y","mag_z","gnss_alt","gnss_speed"]:
            if c in df.columns:
                df[c] = np.clip(df[c], np.nanpercentile(df[c], 0.01), np.nanpercentile(df[c], 99.99))
        return df