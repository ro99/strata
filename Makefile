.PHONY: all configure build check check-layers test tui tui-check tui-clean check-all clean

CARGO ?= cargo
PYTHON ?= python3

all: build

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build build --parallel

check: configure
	cmake --build build --parallel
	ctest --test-dir build --output-on-failure

# Phase 1 layering enforcement (docs/experiments/0121). Static: reads the
# strata_* add_library source lists straight out of CMakeLists.txt, needs no
# build. Exits non-zero on any violation -- see the script's own docstring
# for exactly what the two checks are.
check-layers:
	$(PYTHON) scripts/check_layers.py

test: check

tui:
	$(CARGO) build --release --package strata-tui

tui-check:
	$(CARGO) fmt --all -- --check
	$(CARGO) clippy --workspace --all-targets -- -D warnings
	$(CARGO) test --workspace

tui-clean:
	$(CARGO) clean

check-all: check tui-check

clean:
	cmake -E remove_directory build
