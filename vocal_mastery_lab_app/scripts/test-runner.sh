#!/bin/bash

# VocalMasteryLab Test Runner Script
# Usage: ./scripts/test-runner.sh [all|ui|unit|critical|smoke|coverage] [test-name]
#
# Examples:
#   ./scripts/test-runner.sh ui                    # Run all UI tests
#   ./scripts/test-runner.sh unit                  # Run all Unit tests
#   ./scripts/test-runner.sh all                   # Run all tests
#   ./scripts/test-runner.sh critical              # Run critical UI tests only (~1 min)
#   ./scripts/test-runner.sh smoke                 # Run smoke UI tests (~3 min)
#   ./scripts/test-runner.sh ui PaywallUITests     # Run specific UI test class
#   ./scripts/test-runner.sh coverage              # Run unit tests with coverage report
#   ./scripts/test-runner.sh coverage --html       # Generate HTML coverage report

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT="VocalMasteryLab.xcodeproj"
DESTINATION="platform=iOS Simulator,name=iPhone 16 Clean"
COVERAGE_DIR="coverage_reports"
RESULT_BUNDLE_PATH="${COVERAGE_DIR}/TestResults.xcresult"

# Functions
print_usage() {
    echo -e "${BLUE}VocalMasteryLab Test Runner${NC}"
    echo ""
    echo "Usage: $0 [all|ui|unit|critical|smoke|coverage] [options]"
    echo ""
    echo "Test Types:"
    echo "  all      - Run all tests (Unit + UI)"
    echo "  ui       - Run all UI tests"
    echo "  unit     - Run Unit tests only"
    echo "  critical - Run critical UI tests only (~1 min)"
    echo "  smoke    - Run smoke UI tests (~3 min)"
    echo "  coverage - Run unit tests with coverage report"
    echo ""
    echo "Coverage Options:"
    echo "  coverage           - Run tests and show coverage summary"
    echo "  coverage --html    - Generate HTML coverage report"
    echo "  coverage --json    - Generate JSON coverage report"
    echo "  coverage --files   - Show per-file coverage breakdown"
    echo ""
    echo "Examples:"
    echo "  $0 ui                    # Run all UI tests"
    echo "  $0 unit                  # Run all Unit tests"
    echo "  $0 all                   # Run all tests"
    echo "  $0 critical              # Run critical tests (fastest)"
    echo "  $0 smoke                 # Run smoke tests (quick validation)"
    echo "  $0 ui PaywallUITests     # Run specific UI test class"
    echo "  $0 coverage              # Run tests with coverage summary"
    echo "  $0 coverage --html       # Generate HTML report in coverage_reports/"
    echo ""
}

# Critical tests - minimum viable tests for core functionality
# Expected: ~1 minute
run_critical_tests() {
    echo -e "${BLUE}Running CRITICAL tests (core functionality)${NC}"
    echo ""

    local cmd="xcodebuild test \
        -project ${PROJECT} \
        -scheme VocalMasteryLab-UIOnly \
        -destination '${DESTINATION}' \
        -parallel-testing-enabled NO \
        -allowProvisioningUpdates \
        -only-testing:VocalMasteryLabUITests/RecordingFlowUITests/testBasicRecordingFlow \
        -only-testing:VocalMasteryLabUITests/RecordingListUITests/testDeleteRecording \
        -only-testing:VocalMasteryLabUITests/RecordingLimitUITests/testRecordingLimitAlert_shouldAppear_whenAtLimit \
        -only-testing:VocalMasteryLabUITests/PaywallUITests/testPurchase_shouldUpdateToPremiumStatus"

    echo -e "${YELLOW}Tests: testBasicRecordingFlow, testDeleteRecording, testRecordingLimitAlert, testPurchase${NC}"
    echo ""

    if eval $cmd; then
        echo ""
        echo -e "${GREEN}✅ CRITICAL Tests PASSED${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}❌ CRITICAL Tests FAILED${NC}"
        return 1
    fi
}

# Smoke tests - quick validation of main features
# Expected: ~3 minutes
run_smoke_tests() {
    echo -e "${BLUE}Running SMOKE tests (main features validation)${NC}"
    echo ""

    local cmd="xcodebuild test \
        -project ${PROJECT} \
        -scheme VocalMasteryLab-UIOnly \
        -destination '${DESTINATION}' \
        -parallel-testing-enabled NO \
        -allowProvisioningUpdates \
        -only-testing:VocalMasteryLabUITests/RecordingFlowUITests/testBasicRecordingFlow \
        -only-testing:VocalMasteryLabUITests/RecordingListUITests/testDeleteRecording \
        -only-testing:VocalMasteryLabUITests/RecordingLimitUITests/testRecordingLimitAlert_shouldAppear_whenAtLimit \
        -only-testing:VocalMasteryLabUITests/PaywallUITests/testPurchase_shouldUpdateToPremiumStatus \
        -only-testing:VocalMasteryLabUITests/NavigationUITests/testMultipleRecordings \
        -only-testing:VocalMasteryLabUITests/PlaybackUITests/testPlaybackFullCompletion \
        -only-testing:VocalMasteryLabUITests/AnalysisUITests/testAnalysisViewDisplay"

    echo -e "${YELLOW}Tests: Critical + testMultipleRecordings, testPlaybackFullCompletion, testAnalysisViewDisplay${NC}"
    echo ""

    if eval $cmd; then
        echo ""
        echo -e "${GREEN}✅ SMOKE Tests PASSED${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}❌ SMOKE Tests FAILED${NC}"
        return 1
    fi
}

# Coverage tests - run unit tests with code coverage
run_coverage_tests() {
    local option=$1

    echo -e "${BLUE}Running tests with CODE COVERAGE enabled${NC}"
    echo ""

    # Create coverage directory
    mkdir -p "${COVERAGE_DIR}"

    # Remove old result bundle if exists
    if [ -d "${RESULT_BUNDLE_PATH}" ]; then
        rm -rf "${RESULT_BUNDLE_PATH}"
    fi

    local cmd="xcodebuild test \
        -project ${PROJECT} \
        -scheme VocalMasteryLab-UnitOnly \
        -destination '${DESTINATION}' \
        -parallel-testing-enabled NO \
        -allowProvisioningUpdates \
        -enableCodeCoverage YES \
        -resultBundlePath '${RESULT_BUNDLE_PATH}'"

    echo -e "${YELLOW}Executing tests with coverage...${NC}"
    echo ""

    # Run tests
    local test_result=0
    if ! eval $cmd; then
        echo ""
        echo -e "${RED}⚠️  Some tests failed, but generating coverage report anyway${NC}"
        test_result=1
    fi

    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}                    COVERAGE REPORT                         ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    # Check if result bundle exists
    if [ ! -d "${RESULT_BUNDLE_PATH}" ]; then
        echo -e "${RED}Error: Result bundle not found at ${RESULT_BUNDLE_PATH}${NC}"
        return 1
    fi

    case "$option" in
        --html)
            generate_html_report
            ;;
        --json)
            generate_json_report
            ;;
        --files)
            show_file_coverage
            ;;
        *)
            show_coverage_summary
            ;;
    esac

    return $test_result
}

# Show coverage summary
show_coverage_summary() {
    echo -e "${YELLOW}Coverage Summary:${NC}"
    echo ""
    xcrun xccov view --report "${RESULT_BUNDLE_PATH}" 2>/dev/null | head -50
    echo ""
    echo -e "${GREEN}Full report: xcrun xccov view --report '${RESULT_BUNDLE_PATH}'${NC}"
    echo -e "${GREEN}HTML report: $0 coverage --html${NC}"
}

# Show per-file coverage
show_file_coverage() {
    echo -e "${YELLOW}Per-File Coverage:${NC}"
    echo ""
    xcrun xccov view --report --files-for-target VocalMasteryLab "${RESULT_BUNDLE_PATH}" 2>/dev/null
    echo ""
}

# Generate HTML coverage report
generate_html_report() {
    local html_dir="${COVERAGE_DIR}/html"
    mkdir -p "${html_dir}"

    echo -e "${YELLOW}Generating HTML coverage report...${NC}"

    # Generate JSON first
    local json_file="${COVERAGE_DIR}/coverage.json"
    xcrun xccov view --report --json "${RESULT_BUNDLE_PATH}" > "${json_file}" 2>/dev/null

    # Create HTML report
    local html_file="${html_dir}/index.html"

    # Parse JSON and generate HTML
    cat > "${html_file}" << 'HTMLEOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VocalMasteryLab Coverage Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #007AFF; padding-bottom: 10px; }
        .summary { display: flex; gap: 20px; margin: 20px 0; }
        .stat { background: #f0f0f0; padding: 15px 25px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #007AFF; }
        .stat-label { color: #666; font-size: 0.9em; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f8f8; font-weight: 600; }
        .coverage-bar { width: 100px; height: 8px; background: #eee; border-radius: 4px; overflow: hidden; }
        .coverage-fill { height: 100%; border-radius: 4px; }
        .high { background: #34C759; }
        .medium { background: #FF9500; }
        .low { background: #FF3B30; }
        .file-name { font-family: monospace; font-size: 0.9em; }
        .timestamp { color: #999; font-size: 0.8em; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 VocalMasteryLab Coverage Report</h1>
        <p class="timestamp">Generated: TIMESTAMP_PLACEHOLDER</p>
        <div id="content">
            <p>Loading coverage data...</p>
        </div>
    </div>
    <script>
HTMLEOF

    # Add JavaScript to parse and display coverage
    cat >> "${html_file}" << JSEOF
        const coverageData = $(cat "${json_file}");

        function formatPercent(value) {
            return (value * 100).toFixed(1) + '%';
        }

        function getCoverageClass(value) {
            if (value >= 0.8) return 'high';
            if (value >= 0.5) return 'medium';
            return 'low';
        }

        function renderReport() {
            const targets = coverageData.targets || [];
            const mainTarget = targets.find(t => t.name === 'VocalMasteryLab.app') || targets[0];

            if (!mainTarget) {
                document.getElementById('content').innerHTML = '<p>No coverage data found</p>';
                return;
            }

            const coverage = mainTarget.lineCoverage || 0;
            const files = mainTarget.files || [];

            let html = \`
                <div class="summary">
                    <div class="stat">
                        <div class="stat-value">\${formatPercent(coverage)}</div>
                        <div class="stat-label">Line Coverage</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">\${files.length}</div>
                        <div class="stat-label">Files</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">\${mainTarget.coveredLines || 0}</div>
                        <div class="stat-label">Covered Lines</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">\${mainTarget.executableLines || 0}</div>
                        <div class="stat-label">Total Lines</div>
                    </div>
                </div>

                <h2>📁 File Coverage</h2>
                <table>
                    <thead>
                        <tr>
                            <th>File</th>
                            <th>Coverage</th>
                            <th style="width: 120px;">Progress</th>
                            <th>Lines</th>
                        </tr>
                    </thead>
                    <tbody>
            \`;

            files.sort((a, b) => (a.lineCoverage || 0) - (b.lineCoverage || 0));

            files.forEach(file => {
                const fileCoverage = file.lineCoverage || 0;
                const coverageClass = getCoverageClass(fileCoverage);
                const fileName = file.name || file.path?.split('/').pop() || 'Unknown';

                html += \`
                    <tr>
                        <td class="file-name">\${fileName}</td>
                        <td>\${formatPercent(fileCoverage)}</td>
                        <td>
                            <div class="coverage-bar">
                                <div class="coverage-fill \${coverageClass}" style="width: \${fileCoverage * 100}%"></div>
                            </div>
                        </td>
                        <td>\${file.coveredLines || 0} / \${file.executableLines || 0}</td>
                    </tr>
                \`;
            });

            html += '</tbody></table>';
            document.getElementById('content').innerHTML = html;
        }

        renderReport();
JSEOF

    cat >> "${html_file}" << 'HTMLEOF'
    </script>
</body>
</html>
HTMLEOF

    # Replace timestamp
    sed -i '' "s/TIMESTAMP_PLACEHOLDER/$(date '+%Y-%m-%d %H:%M:%S')/" "${html_file}"

    echo ""
    echo -e "${GREEN}✅ HTML coverage report generated!${NC}"
    echo -e "${GREEN}   Open: ${html_file}${NC}"
    echo ""

    # Try to open in browser
    if command -v open &> /dev/null; then
        open "${html_file}"
    fi
}

# Generate JSON coverage report
generate_json_report() {
    local json_file="${COVERAGE_DIR}/coverage.json"

    echo -e "${YELLOW}Generating JSON coverage report...${NC}"
    xcrun xccov view --report --json "${RESULT_BUNDLE_PATH}" > "${json_file}" 2>/dev/null

    echo ""
    echo -e "${GREEN}✅ JSON coverage report generated!${NC}"
    echo -e "${GREEN}   File: ${json_file}${NC}"
    echo ""

    # Show summary from JSON
    echo -e "${YELLOW}Coverage Summary:${NC}"
    cat "${json_file}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
targets = data.get('targets', [])
for t in targets:
    if 'VocalMasteryLab' in t.get('name', ''):
        coverage = t.get('lineCoverage', 0) * 100
        covered = t.get('coveredLines', 0)
        total = t.get('executableLines', 0)
        print(f'  Target: {t[\"name\"]}')
        print(f'  Coverage: {coverage:.1f}%')
        print(f'  Lines: {covered}/{total}')
        print()
" 2>/dev/null || echo "  (Install python3 for detailed summary)"
}

run_tests() {
    local scheme=$1
    local test_target=$2
    local test_filter=$3

    echo -e "${BLUE}Running tests with scheme: ${scheme}${NC}"

    local cmd="xcodebuild test \
        -project ${PROJECT} \
        -scheme ${scheme} \
        -destination '${DESTINATION}' \
        -parallel-testing-enabled NO \
        -allowProvisioningUpdates"

    # Add test filter if specified
    if [ -n "$test_filter" ]; then
        cmd="${cmd} -only-testing:${test_target}/${test_filter}"
        echo -e "${YELLOW}Filter: ${test_filter}${NC}"
    fi

    echo -e "${YELLOW}Executing: ${cmd}${NC}"
    echo ""

    # Run tests
    if eval $cmd; then
        echo ""
        echo -e "${GREEN}✅ Tests PASSED${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}❌ Tests FAILED${NC}"
        return 1
    fi
}

list_schemes() {
    echo -e "${BLUE}Available schemes:${NC}"
    ls -1 VocalMasteryLab.xcodeproj/xcshareddata/xcschemes/ | grep "\.xcscheme$" | sed 's/\.xcscheme$//' | while read scheme; do
        echo "  - $scheme"
    done
    echo ""
}

# Main script
main() {
    local test_type=$1
    local test_name=$2

    # Check if we're in the right directory
    if [ ! -d "$PROJECT" ]; then
        echo -e "${RED}Error: VocalMasteryLab.xcodeproj not found${NC}"
        echo "Please run this script from the VocalMasteryLab directory"
        exit 1
    fi

    # If no arguments, show usage
    if [ -z "$test_type" ]; then
        print_usage
        list_schemes
        exit 0
    fi

    # Select scheme and target based on test type
    case "$test_type" in
        all)
            run_tests "VocalMasteryLab-All" "" "$test_name"
            ;;
        ui)
            run_tests "VocalMasteryLab-UIOnly" "VocalMasteryLabUITests" "$test_name"
            ;;
        unit)
            run_tests "VocalMasteryLab-UnitOnly" "VocalMasteryLabTests" "$test_name"
            ;;
        critical)
            run_critical_tests
            ;;
        smoke)
            run_smoke_tests
            ;;
        coverage)
            run_coverage_tests "$test_name"
            ;;
        help|--help|-h)
            print_usage
            list_schemes
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown test type '${test_type}'${NC}"
            echo ""
            print_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
