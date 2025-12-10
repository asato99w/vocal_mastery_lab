#!/bin/bash
# Xcode test runner with automatic error analysis
# Usage: bash scripts/test_with_analysis.sh [test_name]

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

TEST_NAME="${1:-VocalisStudioTests}"
LOG_FILE="/tmp/vocalis_test_$(date +%Y%m%d_%H%M%S).log"

echo "🧪 Running tests: $TEST_NAME"
echo "📝 Log file: $LOG_FILE"
echo ""

# Run test and capture all output
xcodebuild test \
  -project VocalisStudio.xcodeproj \
  -scheme VocalisStudio \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  -only-testing:"$TEST_NAME" \
  -allowProvisioningUpdates \
  2>&1 | tee "$LOG_FILE"

TEST_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================="
echo "📊 AUTOMATIC ERROR ANALYSIS"
echo "========================================="
echo ""

# Analyze errors
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "❌ Tests failed (exit code: $TEST_EXIT_CODE)"
    echo ""

    # Show build errors
    echo "=== 🔴 BUILD ERRORS (First 10) ==="
    grep -i "error:" "$LOG_FILE" | head -10
    echo ""

    # Show syntax errors
    echo "=== ⚠️  SYNTAX ERRORS (First 10) ==="
    grep -E "Expected|Cannot find|Invalid|Extraneous|Undefined" "$LOG_FILE" | head -10
    echo ""

    # Show test failures
    echo "=== ❌ TEST FAILURES ==="
    grep -E "Test Case.*failed" "$LOG_FILE"
    echo ""

    echo "💡 Full log available at: $LOG_FILE"
else
    echo "✅ All tests passed!"

    # Show passed tests
    echo ""
    echo "=== ✅ TEST SUCCESSES ==="
    grep -E "Test Case.*passed" "$LOG_FILE" | tail -10
fi

echo ""
echo "========================================="

exit $TEST_EXIT_CODE
