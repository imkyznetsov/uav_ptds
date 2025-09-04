from typing import Dict
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
def add_sensors_noise(signals: Dict[str, np.ndarray], cfg: SimConfig) -> Dict[str, np.ndarray]:
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


