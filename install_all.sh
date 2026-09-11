#!/bin/bash

# install foresight in editable mode
uv pip install -e .

# install foresight dependencies
bash install_foresight_deps.sh
