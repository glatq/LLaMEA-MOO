# Package name, read from pyproject.toml so this Makefile is reusable across projects
PACKAGE := $(shell awk -F'"' '/^name = / {print $$2; exit}' pyproject.toml)

.PHONY: test

# Run the test suite with coverage (matches the current CI invocation)
test:
	poetry run pytest --cov=$(PACKAGE) tests/
