tests:
	pytest tests/

code_quality_checks:
	isort .
	black .
	pylint --recursive=y .

setup:
	pipenv install --dev
	pre-commit install
