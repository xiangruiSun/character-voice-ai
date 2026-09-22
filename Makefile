.DEFAULT_GOAL := help
PY ?= python3

.PHONY: help install install-preprocess test test-fast demo talk audition validate lint clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the project and dev tools (Milestone 1 needs nothing else)
	$(PY) -m pip install -e ".[dev]"

install-preprocess: ## Add the Milestone 2 audio stack (ASR, separation, loudness)
	$(PY) -m pip install -e ".[preprocess]"

test: ## Run the full test suite
	$(PY) -m pytest -q

test-fast: ## Skip the slower end-to-end benchmark tests
	$(PY) -m pytest -q --deselect tests/test_benchmark.py

demo: ## Build the synthetic pack, run the benchmark, build a blind test, aggregate
	$(PY) -m cvai_evaluation.cli demo

talk: ## Talk to the synthetic demo character offline (no key, no GPU, no network)
	@test -d voicepacks/demo_zh || $(MAKE) demo
	CVAI_LLM=mock $(PY) -m cvai_api.talk_cli demo_zh

audition: ## Speak one line per style and write a page to listen through
	@test -d voicepacks/demo_zh || $(MAKE) demo
	CVAI_LLM=mock $(PY) -m cvai_api.talk_cli demo_zh --audition

validate: ## Validate every voice pack in the repository
	$(PY) -m cvai_core.cli.voicepack_cli list

lint: ## Byte-compile everything as a cheap syntax check
	$(PY) -m compileall -q packages services providers scripts tests

clean: ## Remove generated runs, caches and the synthetic demo pack
	rm -rf runs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf voicepacks/demo_zh
