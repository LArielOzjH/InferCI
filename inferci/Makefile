.PHONY: install install-dev test lint format-check clean

install:
	pip install -e .

install-dev:
	pip install -e '.[dev]'

test:
	python -m unittest discover -s tests

lint:
	ruff check .

format-check:
	ruff format --check .

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f *.db
	rm -rf *.egg-info
