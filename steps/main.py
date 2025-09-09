import logging, yaml, os
from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.features import FeatureBuilder
from steps.train import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main():
    with open("config.yml","r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["data"]["raw_dir"]
    processed_feats = cfg["data"]["features_path"]
    fs = cfg["params"]["fs"]; window_s = cfg["params"]["window_s"]; overlap = cfg["params"]["overlap"]

    ing = Ingestion(raw_dir=raw_dir, seed=cfg["params"]["seed"], n_flights=cfg["params"]["n_flights"])
    rd = ing.load_data()
    logging.info("Raw data ready")

    cl = Cleaner(processed_dir="data/processed")
    df_concat = cl.concat_raw(rd)
    df_clean = cl.clean(df_concat)
    logging.info("Clean completed")

    fb = FeatureBuilder(fs=fs, window_s=window_s, overlap=overlap, out_path=processed_feats)
    feats = fb.transform(df_clean)
    logging.info(f"Features saved: {feats.shape}")

    tr = Trainer(feats_path=processed_feats, experiment=cfg["mlflow"]["experiment"])
    df = tr.load()
    tr.train_cv(df)
    logging.info("Training done")

if __name__ == "__main__":
    main()