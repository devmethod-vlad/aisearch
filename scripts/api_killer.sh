#!/bin/bash

LOG_PATH="${1}"

mkdir -p "$(dirname "$LOG_PATH")"

log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_entry="$timestamp [api_killer.sh] $message"

    echo "$log_entry"
    echo "$log_entry" >> "$LOG_PATH"
}

HEALTHCHECK_URL="${APP_HOST}:${APP_PORT}/healthcheck"

response=$(curl -s -w "\n%{http_code}" "$HEALTHCHECK_URL")
http_code=$(echo "$response" | tail -n1)
response_body=$(echo "$response" | head -n -1)

if [ "$http_code" != "200" ]; then
    log "💀 ERROR: API healthcheck FAILED - HTTP status: $http_code"
    log "🔍 Debug info:"
    log "  - Host: ${APP_HOST}"
    log "  - Port: ${APP_PORT}"

    if [ -n "$response_body" ]; then
        log "📋 Response body: $response_body"

        overall_status=$(echo "$response_body" | jq -r '.status' 2>/dev/null || echo "unknown")
        if [ "$overall_status" = "error" ]; then
            log "🔍 Failed services:"
            echo "$response_body" | jq -r '.services | to_entries[] | select(.value.status == "error") | "  - \(.key): \(.value.message)"' 2>/dev/null | while read line; do
                if [ -n "$line" ]; then
                    log "$line"
                fi
            done
        fi
    fi

    log "🚨 EMERGENCY: Terminating container"
    kill -TERM 1
    exit 1
fi

if [ -z "$response_body" ]; then
    log "💀 ERROR: API healthcheck returned empty response"
    log "🚨 EMERGENCY: Terminating container"
    kill -TERM 1
    exit 1
fi

overall_status=$(echo "$response_body" | jq -r '.status')

if [ "$overall_status" = "error" ]; then
    log "💀 ERROR: API healthcheck FAILED - Overall status: $overall_status"
    log "🔍 Failed services:"

    echo "$response_body" | jq -r '.services | to_entries[] | select(.value.status == "error") | "  - \(.key): \(.value.message)"' | while read line; do
        if [ -n "$line" ]; then
            log "$line"
        fi
    done

    log "🚨 EMERGENCY: Terminating container"
    kill -TERM 1
    exit 1
elif [ "$overall_status" != "ok" ]; then
    log "💀 ERROR: API healthcheck returned unknown status: '$overall_status'"
    log "📋 Full response: $response_body"
    log "🚨 EMERGENCY: Terminating container"
    kill -TERM 1
    exit 1
fi

if [ -z "${REQUIRED_CONTAINERS:-}" ]; then
    log "💀 ERROR: REQUIRED_CONTAINERS is not set"
    log "🚨 EMERGENCY: Terminating container"
    kill -TERM 1
    exit 1
fi

read -r -a REQUIRED_CONTAINERS_LIST <<< "$REQUIRED_CONTAINERS"

CONTAINER_FAILURES=0
MISSING_CONTAINERS=()

for container in "${REQUIRED_CONTAINERS_LIST[@]}"; do
    if ! getent hosts "$container" > /dev/null 2>&1; then
        log "❌ FAIL: $container is NOT reachable"
        ((CONTAINER_FAILURES++))
        MISSING_CONTAINERS+=("$container")
    fi
done

if [ $CONTAINER_FAILURES -gt 0 ]; then
    log "💀 ERROR: $CONTAINER_FAILURES container(s) are missing"
    log "📋 Missing containers: ${MISSING_CONTAINERS[*]}"
    log "🚨 EMERGENCY: Terminating container"
    kill -TERM 1
    exit 1
fi

exit 0
