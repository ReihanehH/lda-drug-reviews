PYTHON_VERSION := 3.9
VENV_DIR := .venv

# Targets
.PHONY: all setup-virtualenv install-dependencies run-tests run-main

# Default target
all: setup-virtualenv install-dependencies run-tests run-main cleanup

setup-virtualenv:
	pip install virtualenv
	virtualenv $(VENV_DIR) -p $(PYTHON_VERSION)

# Install dependencies
install-dependencies: setup-virtualenv
	$(VENV_DIR)/bin/pip install -r requirements.txt
	$(VENV_DIR)/bin/python -m nltk.downloader -d ~/nltk_data stopwords

# Run tests
run-tests: install-dependencies
	$(VENV_DIR)/bin/python -m unittest tests/*

# Run main
run-main: run-tests
	$(VENV_DIR)/bin/python main.py

cleanup:
	rm -rf $(VENV_DIR)