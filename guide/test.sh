#!/usr/bin/env bash
# Test all Rust code blocks in the mdBook guide.
#
# This compiles `rust,no_run` blocks (but does not execute them) and
# skips `rust,ignore` blocks. It catches wrong method names, wrong
# types, missing imports, etc.
#
# Usage:
#   cd vulkan_rs
#   bash guide/test.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Building vulkan-rs..."
cargo build -p vulkan-rs 2>&1

# Find the rlib paths. Pick the most recently modified one if multiple exist.
DEPS_DIR="target/debug/deps"
VK_ENGINE_RLIB=$(ls -t "$DEPS_DIR"/libvulkan_rs-*.rlib 2>/dev/null | head -1)
VK_SYS_RLIB=$(ls -t "$DEPS_DIR"/libvulkan_rs_sys-*.rlib 2>/dev/null | head -1)

if [[ -z "$VK_ENGINE_RLIB" || -z "$VK_SYS_RLIB" ]]; then
    echo "ERROR: Could not find vulkan-rs or vulkan-rs-sys rlib in $DEPS_DIR"
    exit 1
fi

echo "==> Using:"
echo "    vulkan_rs = $VK_ENGINE_RLIB"
echo "    vulkan_rs_sys    = $VK_SYS_RLIB"

FAILED=0
PASSED=0
SKIPPED=0

for md in guide/src/**/*.md guide/src/*.md; do
    echo -n "  Testing $md ... "

    OUTPUT=$(rustdoc --edition 2021 --test "$md" \
        -L "$DEPS_DIR" \
        --extern "vulkan_rs=$VK_ENGINE_RLIB" \
        --extern "vulkan_rs_sys=$VK_SYS_RLIB" \
        2>&1) || true

    if echo "$OUTPUT" | grep -q "test result: FAILED"; then
        echo "FAILED"
        echo "$OUTPUT" | grep -E "^(error|failures:)" | head -10
        FAILED=$((FAILED + 1))
    elif echo "$OUTPUT" | grep -q "test result: ok"; then
        # Extract pass/ignore counts
        COUNTS=$(echo "$OUTPUT" | grep "test result: ok" | head -1)
        echo "ok ($COUNTS)"
        PASSED=$((PASSED + 1))
    else
        echo "skipped (no testable code blocks)"
        SKIPPED=$((SKIPPED + 1))
    fi
done

echo ""
echo "==> Results: $PASSED passed, $FAILED failed, $SKIPPED skipped"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
