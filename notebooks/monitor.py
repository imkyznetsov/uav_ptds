import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently import ColumnMapping

def run_monitor(ref_path="data/processed/features.parquet", cur_path="data/production.csv", out_html="production_report.html"):
    ref = pd.read_parquet(ref_path)
    cur = pd.read_csv(cur_path)
    # Маппинг колонок (признаки числовые)
    cm = ColumnMapping()
    cm.target = "y" if "y" in ref.columns else None
    numeric = [c for c in ref.columns if ref[c].dtype != "object"]
    cm.numerical_features = [c for c in numeric if c not in ["y"]]
    rep = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    rep.run(reference_data=ref, current_data=cur, column_mapping=cm)
    rep.save_html(out_html)
    print(f"Saved report to {out_html}")

if __name__ == "__main__":
    run_monitor()