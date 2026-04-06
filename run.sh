#!/bin/bash

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"

# Install `uv`
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv init assignment_NLP
uv add datasets matplotlib pandas scikit-learn torch tqdm "transformers>=5.5.0"

# Run within virtual environment
uv run python3 main.py

# Profile execution to identify bottlenecks
#uv run python3 -m cProfile -s tottime main.py  > "${timestamp}_trace.txt"
