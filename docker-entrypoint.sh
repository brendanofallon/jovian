#!/usr/bin/env bash

set -ex

echo "App dir: $APPDIR"
echo "Python path: $PYTHONPATH"
python -c "import torch; print(torch.__file__)"
python $APPDIR/dnaseq2seq/main.py