python=python

setup:
	$(python) -m pip install --upgrade pip
	$(python) -m pip install -r requirements.txt

run:
	MLFLOW_TRACKING_URI="file:./mlruns" $(python) main.py

mlflow-ui:
	mlflow ui --backend-store-uri file:./mlruns

clean:
	rm -rf data/raw/*.parquet data/processed/*.parquet

docker-build:
	docker build -t uav-health:latest -f dockerfile .

docker-run:
	docker run -p 8000:8000 uav-health:latest