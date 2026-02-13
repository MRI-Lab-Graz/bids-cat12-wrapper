#!/bin/bash
# Fixed wrapper for CAT12 standalone execution
# Sets up MCR environment properly before calling cat_standalone.sh

# MCR paths
MCR_ROOT="/data/local/software/cat-12/external/MCR/v232/R2023b"
CAT12_ROOT="/data/local/software/cat-12/external/cat12"

# Set up library path for MCR
export LD_LIBRARY_PATH="${MCR_ROOT}/runtime/glnxa64:${MCR_ROOT}/bin/glnxa64:${MCR_ROOT}/sys/os/glnxa64:${MCR_ROOT}/sys/opengl/lib/glnxa64:${LD_LIBRARY_PATH}"

# Preload glibc_shim for RHEL7 if needed
if [ -e /usr/bin/ldd ] && ldd --version 2>&1 | grep -q "(GNU libc) 2\.17"; then
    export LD_PRELOAD="${MCR_ROOT}/bin/glnxa64/glibc-2.17_shim.so"
fi

# Call the original cat_standalone.sh with all arguments
exec "${CAT12_ROOT}/standalone/cat_standalone.sh" "$@"
