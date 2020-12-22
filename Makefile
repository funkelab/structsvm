default:
	pip install -r requirements.txt
	pip install .

install-dev:
	pip install -r requirements_dev.txt
	pip install -e .

.PHONY: tests
tests:
	pytest -v --cov=structsvm structsvm
	flake8 structsvm
