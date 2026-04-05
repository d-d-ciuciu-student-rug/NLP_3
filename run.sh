#!/bin/bash

# We ran a profiler to see what exactly gives the overly-large CPU-side bottleneck.
#  It seems to be that the data-loading is too slow, but couldn't quite understand why.
#  Or maybe there are too many CPU-GPU synchronizations, and more processing should be moved GPU-side.

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"

   OMP_NUM_THREADS=8 \
&& MKL_NUM_THREADS=8 \
&& uv run python3 main.py
#&& uv run python3 -m cProfile -s tottime main.py  > "${timestamp}_trace.txt"
