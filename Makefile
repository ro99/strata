.PHONY: all configure build check test clean

all: build

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build build --parallel

check: configure
	cmake --build build --parallel
	ctest --test-dir build --output-on-failure

test: check

clean:
	cmake -E remove_directory build
