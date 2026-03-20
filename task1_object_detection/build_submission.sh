#!/bin/bash
# Creates submission.zip with run.py at the root (required format)
# Usage: ./build_submission.sh
cd "$(dirname "$0")"
zip -r ../submission.zip . -x ".*" "__MACOSX/*" "build_submission.sh" "train.py"
echo "Created submission.zip — verify with: unzip -l ../submission.zip | head -10"
