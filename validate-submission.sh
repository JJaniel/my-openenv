#!/usr/bin/env bash
set -uo pipefail
DOCKER_BUILD_TIMEOUT=600
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'
log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() { printf "\n${RED}${BOLD}Validation stopped at %s.${NC}\n" "$1"; exit 1; }

PING_URL="${1:-http://localhost:8000}"
REPO_DIR="."

log "${BOLD}Step 1/3: Pinging local server${NC} ($PING_URL/reset) ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" || printf "000")
if [ "$HTTP_CODE" = "200" ]; then pass "Server is live"; else fail "Local server check failed: $HTTP_CODE"; stop_at "Step 1"; fi

log "${BOLD}Step 2/3: Running docker build${NC} ..."
if docker build -t email-triage-test .; then pass "Docker build succeeded"; else fail "Docker build failed"; stop_at "Step 2"; fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."
if openenv validate .; then pass "openenv validate passed"; else fail "openenv validate failed"; stop_at "Step 3"; fi

printf "\n${GREEN}${BOLD}  All checks passed locally! Ready for HF deployment.${NC}\n"
