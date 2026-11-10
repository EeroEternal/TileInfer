# Pick the project venv when it exists; override with `make PY=/path/to/python`.
PY ?= $(shell [ -x .venv_native213/bin/python ] && echo .venv_native213/bin/python || echo python)
BACKEND ?= reference
DEVICE ?= auto
CPU_MARKS := not npu and not tilelang

.PHONY: help test test-one test-npu test-device env bench bench-quick bench-skew lint fmt clean

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

env:  ## print the environment this repo expects (CANN, cache, python)
	@bash -c 'source scripts/env-ascend102.sh >/dev/null 2>&1; \
	  echo "python          = $(PY)"; \
	  echo "ASCEND_HOME     = $${ASCEND_HOME:-(unset)}"; \
	  echo "TILELANG_CACHE  = $${TILELANG_CACHE_DIR:-(unset)}"'

# The CPU suite is order-dependent today: the contract tests pin vLLM 0.23's API *shape* with
# tests/fake_vllm.py, and that fake only injects itself when no real `vllm` is importable. On a
# machine with vLLM installed the whole-suite result depends on collection order, so a red/green
# suite cannot be trusted. Until that is fixed (see docs/status.md §6), judge a change with
# `make test-one F=tests/test_x.py`, which runs one file in a fresh process.
test:  ## CPU suite (no NPU, no tilelang) — known red on machines with vLLM, see docs/status.md §6
	$(PY) -m pytest tests -q -m "$(CPU_MARKS)"

test-one:  ## one test file in a fresh process: make test-one F=tests/test_kv_cache_write.py
	@test -n "$(F)" || { echo "usage: make test-one F=tests/test_x.py"; exit 2; }
	$(PY) -m pytest "$(F)" -q -m "$(CPU_MARKS)"

test-npu:  ## everything, including the NPU kernel tests (needs the device)
	$(PY) -m pytest tests -q -m ""

test-device:  ## device CI: canary + pytest -m npu, with the toolchain exported
	bash scripts/run_device_ci.sh $(ARGS)

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

clean:  ## drop build artifacts and caches that live inside the repo
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .tilelang_cache bench_out
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
