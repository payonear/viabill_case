tests:
	pytest tests/

code_quality_checks:
	isort .
	black .
	pylint --recursive=y .

setup:
	pip install -U pipenv
	pipenv install --dev
	pre-commit install
