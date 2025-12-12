test:
	uv run pytest -v -W error

build-docs:
	uv run sphinx-build -M html docs/ docs/_build

view-docs:
	open docs/_build/html/index.html

lint:
	uv run ruff format tomsup/ --check
	uv run ruff check tomsup/ --fix

lint-check:
	uv run ruff format tomsup/ --check
	uv run ruff check tomsup/