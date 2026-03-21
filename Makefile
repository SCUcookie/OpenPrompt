PYTHON ?= python3

.PHONY: install smoke test

install:
	$(PYTHON) -m pip install -e .

smoke:
	$(PYTHON) scripts/smoke_test.py --config configs/experiments/geonexus_synthetic.yaml

test:
	$(PYTHON) -m pytest tests

