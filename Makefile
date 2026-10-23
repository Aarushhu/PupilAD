PYTHON ?= .venv/bin/python
PUPILAD = PYTHONPATH=src $(PYTHON) -m pupilad

.PHONY: install download preprocess features validate train all test format

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

download:
	$(PUPILAD) download

preprocess:
	$(PUPILAD) preprocess

features:
	$(PUPILAD) features

validate:
	$(PUPILAD) validate

train:
	$(PUPILAD) train

all: download preprocess features validate

test:
	$(PYTHON) -m pytest -q

format:
	$(PYTHON) -m isort src tests scripts
	$(PYTHON) -m black src tests scripts
