VERSION = 2025-01-v01
export VERSION

MODEL_PATH = ./models/test_path/model_C_1_0.bin
export MODEL_PATH

.PHONY: run
run:
	pipenv run python predictions/serve.py

.PHONY: integration_test
integration_test:
	pipenv run python tests/predict_test.py

.PHONY:	docker_build
docker_build:
	docker build -t predictions .

.PHONY: docker_run
docker_run: docker_build
	docker run -it -p 127.0.0.1:9696:9696 predictions:latest