.PHONY: bootstrap download-data train-wbc train-malaria train-yolo eval-all \
        federated-simulate edge-package docker-up test lint

bootstrap:
	pip install -e ".[dev,dp]"
	pre-commit install

download-data:
	python -m scripts.download_data

train-wbc:
	python -m ml.train_wbc --data data/raabin --epochs 30 --seed 1729

train-malaria:
	python -m ml.train_malaria --data data/malaria --epochs 20 --seed 1729

train-yolo:
	python -m ml.train_yolo --data data/bccd/bccd.yaml --model yolov8n.pt --seed 1729

eval-all:
	python -m ml.eval_all

federated-simulate:
	python -m federated.simulate --clients 5 --rounds 10 --alpha 0.3 --seed 1729

edge-package:
	python -m ml.onnx_export

docker-up:
	docker compose up --build

test:
	pytest --cov=ml --cov=api --cov-report=term-missing --cov-fail-under=80

lint:
	ruff check . && mypy ml api common
