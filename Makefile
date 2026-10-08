PY ?= python
BACKEND ?= reference
DEVICE ?= auto

.PHONY: help test test-npu bench bench-quick bench-skew lint fmt clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

test:  ## run the CPU-runnable suite (planning + reference backend)
	$(PY) -m pytest tests -q

test-npu:  ## run the whole suite including the NPU kernel tests
	$(PY) -m pytest tests -q -m ""

bench-quick:  ## two-shape smoke benchmark
	$(PY) benchmarks/bench_attention.py --quick --backend $(BACKEND) --device $(DEVICE)

bench:  ## full shape sweep
	$(PY) benchmarks/bench_attention.py --sweep --backend $(BACKEND) --device $(DEVICE)

bench-skew:  ## ragged-batch load-balancing comparison
	$(PY) benchmarks/bench_attention.py --skew --backend $(BACKEND) --device $(DEVICE)

lint:
	$(PY) -m ruff check tileinfer benchmarks tests || true

fmt:
	$(PY) -m ruff format tileinfer benchmarks tests || true

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
