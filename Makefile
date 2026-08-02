UV ?= uv

.PHONY: setup lint fmt test coverage smoke check-no-pretrained reports reports-check check

setup:
	$(UV) sync

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

test:
	$(UV) run pytest

coverage:
	$(UV) run pytest --cov=glyphos --cov-report=term-missing

# End-to-end toy pipeline on CPU. Must stay under 5 minutes forever (§2.1).
smoke:
	$(UV) run python scripts/smoke.py --config configs/smoke.yaml

# Hard constraint gate (§1): fails if pretrained-weight loading appears outside contrib/openbook/.
check-no-pretrained:
	bash scripts/check_no_pretrained.sh

# Rebuild ALL report PDFs in one pass (never a single doc), then verify freshness.
reports:
	bash scripts/build_reports.sh
	$(UV) run python scripts/check_reports_fresh.py

reports-check:
	$(UV) run python scripts/check_reports_fresh.py

check: lint test check-no-pretrained smoke
	@echo "ALL CHECKS PASSED"
