PYTHON ?= python3
.PHONY: check experiments submission
check:
	$(PYTHON) tests/check.py
experiments:
	$(PYTHON) scripts/experiments.py
submission:
	$(PYTHON) scripts/package_submission.py
