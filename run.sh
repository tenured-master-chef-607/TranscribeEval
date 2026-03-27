#!/bin/bash
set -a
source "$(dirname "$0")/.env"
set +a

APP=$(find ~/Library/Developer/Xcode/DerivedData/TranscribeEval-*/Build/Products/Debug -name "TranscribeEval.app" -maxdepth 1 2>/dev/null | head -1)
if [ -z "$APP" ]; then
    echo "App not found. Build first:"
    echo "  xcodebuild -project TranscribeEval.xcodeproj -scheme TranscribeEval -configuration Debug build"
    exit 1
fi
exec "$APP/Contents/MacOS/TranscribeEval" "$@"
