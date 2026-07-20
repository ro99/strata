.PHONY: all configure build check test tui tui-check tui-clean check-all clean

CARGO ?= cargo

all: build

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build build --parallel

check: configure
	cmake --build build --parallel
	ctest --test-dir build --output-on-failure

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
