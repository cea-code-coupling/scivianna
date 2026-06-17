#!/usr/bin/bash
rm -rf build

# Copy updated tutorials to docs if needed
cp ../tutorials/*.ipynb source/tutorials 2>/dev/null || true

make html