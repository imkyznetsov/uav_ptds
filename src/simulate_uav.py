import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class SimConfig:
    accel_sigma: float = 0.02
    gyro_sigma: float = 0.01
    