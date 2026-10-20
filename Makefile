PY ?= python
TIER ?= smoke
WORKERS ?= 4

.PHONY: install test reproduce site report export lock-plan publish clean
install:
	$(PY) -m pip install -e ".[lab,dev]"

test:
	$(PY) -m pytest -m "not slow"

# smoke: minutes on a laptop (1 seed, a few scenarios), run in CI on every commit
# dev:   3 seeds, 20 scenarios per condition
# full:  10 seeds, 50 scenarios per condition; the published benchmark
reproduce:
	adaptarena reproduce --tier $(TIER) --workers $(WORKERS)

site:
	adaptarena site --tier $(TIER)

report:
	adaptarena report --tier $(TIER)

export:
	adaptarena export --tier $(TIER)

lock-plan:
	adaptarena lock-plan

# Rebuild the published static site from results/full and stage it for GitHub Pages (commit site/published).
publish:
	adaptarena site --tier full
	rm -rf site/published && cp -r site/dist site/published
