#!/usr/bin/env bash
set -euo pipefail
EVALUATION_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CORE_ROOT=$(cd "$EVALUATION_DIR/../.." && pwd)
export PYTHONPATH="$CORE_ROOT/kvpress:$CORE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:-python}
exec "$PYTHON_BIN" "$EVALUATION_DIR/evaluate.py" \
  --config_file "$EVALUATION_DIR/evaluate_config.yaml" "$@"
