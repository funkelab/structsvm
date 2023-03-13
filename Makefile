default:
	pip install .

install-dev:
	pip install -e .[dev]

.PHONY: tests
tests:
	pytest -v --cov=structsvm -s --log-cli-level=INFO structsvm
	flake8 structsvm
