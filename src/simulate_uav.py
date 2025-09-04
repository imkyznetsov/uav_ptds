import os
from typing import Dict, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class SimConfig:
    accel_sigma: float = 0.02
    gyro_sigma: float = 0.01
    gnss_rate_hz: int = 10
    baro_sigma: float = 0.5
    mag_sigma: float = 0.2
    fs: int = 100
    horizon_s: int = 20
    
FAILURE_TYPES = [
    # аксы
    "accel_drift", "accel_high_noise", "accel_stuck", "accel_spikes", "accel_freq",
    # дусы
    "gyro_bias_jump", "gyro_high_noise", "gyro_saturation", "gyro_spikes",
    # GNSS
    "gnss_outage", "gnss_multipath_bias", "gnss_jitter",
    # baro
    "baro_drift", "baro_steps",
    # magnetometer
    "mag_hard_iron_bias", "mag_noise",
    # INS/GNSS
    "ins_filter_divergence", "sensor_desync"
]    

def base_motion(t: np.ndarray) -> Dict[str, np.ndarray]:
    # Траектория и истинные сигналы
    pitch = 5*np.sin(2*np.pi*0.02*t) * np.pi/180
    roll = 3*np.sin(2*np.pi*0.018*t + 0.4) * np.pi/180
    yaw_rate_true = 0.05*np.sin(2*np.pi*0.01*t)
    # Ускорения (в связанной СК, z - вертикальная)
    ax = 0.3*np.sin(2*np.pi*0.2*t) + 0.1*np.sin(2*np.pi*1.7*t)
    ay = 0.25*np.sin(2*np.pi*0.25*t + 0.7) + 0.08*np.sin(2*np.pi*2.0*t)
    az = 9.81 + 0.4*np.sin(2*np.pi*0.15*t)
    # Угловые скорости (упрощённо)
    p = np.gradient(roll, t)
    q = np.gradient(pitch, t)
    r = yaw_rate_true
    # Имитация полёта
    altitude = 100 + 0.8*t + 10*np.sin(2*np.pi*0.005*t)
    speed = 15 + 2*np.sin(2*np.pi*0.03*t) + np.random.normal(0, 0.5, len(t))
    temp = 15 - 0.0065*altitude + np.random.normal(0, 0.3, len(t))
    heading = np.cumsum(r)*(t[1]-t[0])
    return dict(ax=ax, ay=ay, az=az, p=p, q=q, r=r,
                altitude=altitude, speed=speed, temperature=temp,
                pitch=pitch, roll=roll, heading=heading)
    
    #подшумим сигналы - траектория + шум
def add_sensor_noise(signals: Dict[str, np.ndarray], cfg: SimConfig) -> Dict[str, np.ndarray]:
    s = signals.copy()
    # accels
    s["ax"] = s["ax"] + np.random.normal(0, cfg.accel_sigma, len(s["ax"]))
    s["ay"] = s["ay"] + np.random.normal(0, cfg.accel_sigma, len(s["ay"]))
    s["az"] = s["az"] + np.random.normal(0, cfg.accel_sigma, len(s["az"]))
    # gyros
    s["p"] = s["p"] + np.random.normal(0, cfg.gyro_sigma, len(s["p"]))
    s["q"] = s["q"] + np.random.normal(0, cfg.gyro_sigma, len(s["q"]))
    s["r"] = s["r"] + np.random.normal(0, cfg.gyro_sigma, len(s["r"]))
    # baro
    s["baro_alt"] = s["altitude"] + np.random.normal(0, cfg.baro_sigma, len(s["altitude"]))
    # magn
    s["mag_x"] = np.cos(s["heading"]) + np.random.normal(0, cfg.mag_sigma, len(s["heading"]))
    s["mag_y"] = np.sin(s["heading"]) + np.random.normal(0, cfg.mag_sigma, len(s["heading"]))
    s["mag_z"] = 0.1 + np.random.normal(0, cfg.mag_sigma, len(s["heading"]))
    # GNSS
    gnss_t = np.arange(0, len(s["ax"])) / cfg.fs
    gnss_step = int(cfg.fs / cfg.gnss_rate_hz)
    indices = np.arange(0, len(gnss_t), gnss_step)
    gnss_alt = s["altitude"][indices] + np.random.normal(0, 1.5, len(indices))
    gnss_spd = (s["speed"][indices] + np.random.normal(0, 0.3, len(indices)))
    gnss_ok = np.ones_like(indices, dtype=int)
    # синхронизируем
    s["gnss_alt"] = np.repeat(gnss_alt, gnss_step)[:len(gnss_t)]
    s["gnss_speed"] = np.repeat(gnss_spd, gnss_step)[:len(gnss_t)]
    s["gnss_ok"] = np.repeat(gnss_ok, gnss_step)[:len(gnss_t)]
    return s

    # и теперь еще сверху - сбои
def inject_failure(s: Dict[str, np.ndarray], t: np.ndarray, ftype: str, idx: int):
    s = s.copy()
    n = len(t)
    if ftype == "accel_drift":
        drift = 0.002*(t - t[idx]).clip(min=0)
        s["ax"][idx:] += drift[idx:]
    elif ftype == "accel_high_noise":
        s["ay"][idx:] += np.random.normal(0, 0.25, n-idx)
    elif ftype == "accel_stuck":
        s["az"][idx:] = s["az"][idx]
    elif ftype == "accel_spikes":
        k = int(0.05*n)
        pos = np.random.choice(n, k, replace=False)
        s["ax"][pos] += np.random.choice([-1,1], k)* (1.5*(0.5+np.random.rand(k)))
    elif ftype == "accel_freq":
        s["ay"][idx:] += 0.3*np.sin(2*np.pi*8.0*t[idx:])

    elif ftype == "gyro_bias_jump":
        s["p"][idx:] += 0.05
    elif ftype == "gyro_high_noise":
        s["q"][idx:] += np.random.normal(0, 0.1, n-idx)
    elif ftype == "gyro_saturation":
        s["r"][idx:] = np.clip(s["r"][idx:], -0.1, 0.1)
    elif ftype == "gyro_spikes":
        k = int(0.03*n)
        pos = np.random.choice(n, k, replace=False)
        s["r"][pos] += np.random.choice([-1,1], k)* (0.6*(0.5+np.random.rand(k)))

    elif ftype == "gnss_outage":
        s["gnss_ok"][idx:] = 0
        s["gnss_alt"][idx:] = np.nan
        s["gnss_speed"][idx:] = np.nan
    elif ftype == "gnss_multipath_bias":
        s["gnss_alt"][idx:] += 10.0
    elif ftype == "gnss_jitter":
        s["gnss_speed"][idx:] += np.random.normal(0, 3.0, n-idx)

    elif ftype == "baro_drift":
        s["baro_alt"][idx:] += (t[idx:] - t[idx]) * 0.2
    elif ftype == "baro_steps":
        steps = np.cumsum(np.random.choice([-2, 2], size=n-idx))
        s["baro_alt"][idx:] += steps

    elif ftype == "mag_hard_iron_bias":
        s["mag_x"][idx:] += 0.5; s["mag_y"][idx:] -= 0.3
    elif ftype == "mag_noise":
        s["mag_x"][idx:] += np.random.normal(0, 1.0, n-idx)
        s["mag_y"][idx:] += np.random.normal(0, 1.0, n-idx)

    elif ftype == "ins_filter_divergence":
        # проявляется как дивергенция baro/gnss/интеграл IMU
        s["baro_alt"][idx:] += (t[idx:] - t[idx]) * 0.5
        s["gnss_alt"][idx:] -= 5
    elif ftype == "sensor_desync":
        # рассинхронизация: сдвиг GNSS, например, на 1-2 сек
        shift = int(1.5 * (1/(t[1]-t[0])))
        s["gnss_alt"] = np.roll(s["gnss_alt"], shift)
        s["gnss_speed"] = np.roll(s["gnss_speed"], shift)
    return s

    # пронумеруем
def make_labels(ftype: str, idx: Optional[int], t: np.ndarray, horizon_s: int, fs: int):
    y = np.zeros(len(t), dtype=int)
    if ftype == "none" or idx is None:
        return y
    h = int(horizon_s * fs)
    start = max(0, idx - h)
    y[start:idx+1] = 1
    return y

    # поехали
def simulate_flight(cfg: SimConfig) -> pd.DataFrame:
    fs = cfg.fs
    T = cfg.flight_s
    t = np.arange(0, T, 1/fs)
    base = base_motion(t)
    sens = add_sensor_noise(base, cfg)

    failure_type = "none"
    failure_idx = None
    if np.random.rand() < cfg.failure_rate:
        failure_type = np.random.choice(FAILURE_TYPES)
        failure_idx = np.random.randint(int(0.4*len(t)), int(0.85*len(t)))
        sens = inject_failure(sens, t, failure_type, failure_idx)

    y = make_labels(failure_type, failure_idx, t, cfg.horizon_s, fs)

    df = pd.DataFrame({"t": t, **sens})
    df["label"] = y
    df["failure_type"] = failure_type
    return df

def generate_dataset(output_dir="data/raw", cfg=SimConfig()):
    np.random.seed(cfg.seed)
    os.makedirs(output_dir, exist_ok=True)
    meta = []
    for i in range(cfg.n_flights):
        df = simulate_flight(cfg)
        df["flight_id"] = i
        df.to_parquet(os.path.join(output_dir, f"flight_{i:05d}.parquet"), index=False)
        meta.append({"flight_id": i, "failure_type": df["failure_type"].iloc[0]})
    pd.DataFrame(meta).to_csv(os.path.join(output_dir, "meta.csv"), index=False)
    print(f"Generated {cfg.n_flights} flights in {output_dir}")


