#!/bin/bash
# PostToolUse hook: syntax-check only the file that was just edited.
# Deliberately does NOT run `make` -- several agents may edit this repo at once,
# and concurrent builds in the shared build/ directory would corrupt each other.
set -u
input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
[ -z "$file" ] || [ ! -f "$file" ] && exit 0
cd "$(dirname "$0")/../.." || exit 0

# Pull CXX/CXXFLAGS from the Makefile so the check never drifts from the real build.
# -MMD/-MP are dropped: with -fsyntax-only they would litter .d files in the cwd.
CXX=$(make -pn 2>/dev/null | sed -n 's/^CXX := //p' | head -1); CXX=${CXX:-clang++}
CXXFLAGS=$(make -pn 2>/dev/null | sed -n 's/^CXXFLAGS := //p' | head -1 | sed 's/-MMD//; s/-MP//')
CXXFLAGS=${CXXFLAGS:--std=c++20 -Wall -Wextra -Isrc}

out=""
case "$file" in
  *.cpp) out=$($CXX $CXXFLAGS -fsyntax-only "$file" 2>&1) ;;
  *.mm)  out=$($CXX $CXXFLAGS -fobjc-arc -fsyntax-only "$file" 2>&1) ;;
  *.h)   out=$($CXX $CXXFLAGS -fobjc-arc -x objective-c++-header -fsyntax-only "$file" 2>&1) ;;
  *.metal) xcrun -sdk macosx -f metal >/dev/null 2>&1 || exit 0; out=$(xcrun -sdk macosx metal -c "$file" -o /dev/null 2>&1) ;;
  *.py)  out=$(python3 -m py_compile "$file" 2>&1) ;;
  *) exit 0 ;;
esac
status=$?
if [ $status -ne 0 ] || printf '%s' "$out" | grep -q 'warning:'; then
  printf 'Syntax check for %s:\n%s\n' "$file" "$out" | head -60 >&2
  [ $status -ne 0 ] && exit 2
fi
exit 0
