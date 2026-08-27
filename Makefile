.PHONY: all configure build check check-layers check-symbols check-equivalence \
        test check-all clean

PYTHON ?= python3

all: build

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build build --parallel

# Layering enforcement is part of `make check`, not opt-in beside it (brief
# 05, F3): the charter mandates `make check` before every result commit, and
# a check nothing runs by default is not enforcement. check-layers is static
# and runs first so a boundary violation fails fast, before paying for a
# build; check-symbols needs the archives check-layers doesn't, so it runs
# after the build; the equivalence oracle (F2) is a registered ctest entry,
# so `ctest` below already runs it -- it does not need its own line here.
check: configure
	$(PYTHON) scripts/check_layers.py
	cmake --build build --parallel
	$(PYTHON) scripts/check_symbols.py
	ctest --test-dir build --output-on-failure

# Phase 1 layering enforcement. Static: reads the
# strata_* add_library source lists straight out of CMakeLists.txt, needs no
# build. Exits non-zero on any violation -- see the script's own docstring
# for exactly what the two checks are.
check-layers:
	$(PYTHON) scripts/check_layers.py

# Phase 1 layering enforcement, link-symbol level (brief 05, F1). Needs a
# build: reads real nm output from build/lib*.a, which check-layers cannot
# see (an upward reference that exists at link time with no corresponding
# #include is invisible to a pure include-graph check).
check-symbols: build
	$(PYTHON) scripts/check_symbols.py

# Phase 0's equivalence oracle (brief 05, F2), runnable on its own without
# the rest of the suite. Needs a build and the pinned Gemma 4 checkpoint;
# skips loudly (see the script) if either is missing rather than silently
# reporting nothing.
check-equivalence: build
	$(PYTHON) scripts/check_equivalence.py

test: check

check-all: check

clean:
	cmake -E remove_directory build
