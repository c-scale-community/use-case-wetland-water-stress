#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
pushd "$SCRIPT_DIR/../" || exit 1
pytest --cov rattlinbog tests/
FAILED=$?
popd || exit $FAILED
exit $FAILED
