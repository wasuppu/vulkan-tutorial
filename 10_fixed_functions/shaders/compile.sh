#!/bin/bash

if ! command -v glslc >/dev/null 2>&1; then
    echo "can't find glslc"
    exit 1
fi

SCRIPT_PATH=$0
SCRIPT_DIR=$(dirname $SCRIPT_PATH)

glslc "$SCRIPT_DIR/shader.vert" -o "$SCRIPT_DIR/vert.spv"
glslc "$SCRIPT_DIR/shader.frag" -o "$SCRIPT_DIR/frag.spv"