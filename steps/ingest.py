import os
from src.simulate_uav import generate_dataset, SimConfig

class Ingestion:
    def __init__(self, raw_dir="data/raw", seed=13, n_flights=300):
        self.raw_dir = raw_dir
        self.seed = seed
        self.n_flights = n_flights

    def load_data(self):
        if not os.path.exists(self.raw_dir) or len(os.listdir(self.raw_dir)) == 0:
            cfg = SimConfig(seed=self.seed, n_flights=self.n_flights)
            generate_dataset(self.raw_dir, cfg)
        return self.raw_dir  # возвращаем путь к сырым данным