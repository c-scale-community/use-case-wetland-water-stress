#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
pushd "$SCRIPT_DIR" || exit
docker build -t swamydev/rattlinbog-base -f docker/base.Dockerfile .
popd
