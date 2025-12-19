#!/bin/bash

set -e

LOG_PATH="${1}"

log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_entry="$timestamp [wait_for_services.sh] $message"

    echo "$log_entry"
    echo "$log_entry" >> "$LOG_PATH"
}

if [ "$(cat /proc/1/comm 2>/dev/null)" = "systemd" ] || [ "$(cat /proc/1/comm 2>/dev/null)" = "init" ]; then
    log "❌ FATAL: запрещён запуск в host PID namespace (--pid=host)"
    log "   PID 1: $(cat /proc/1/comm 2>/dev/null)"
    log "   Уберите 'pid: host' из конфигурации контейнера"
    exit 1
fi

MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
WAIT_TIME=${WAIT_TIME:-3}

SERVICES=(
    "PostgreSQL:pg_isready -h ${POSTGRES_HOST} -U ${POSTGRES_USER} -p ${POSTGRES_PORT}"
    "Redis:redis-cli -h ${REDIS_HOSTNAME} -p ${REDIS_PORT} ping"
)

mkdir -p "$(dirname "$LOG_PATH")"

check_service() {
    local service_name=$1
    local check_command=$2
    local attempt=1

    while [ $attempt -le $MAX_ATTEMPTS ]; do
        if eval "$check_command" >/dev/null 2>&1; then
            log "✅ $service_name is ready"
            return 0
        else
            if [ $attempt -eq $MAX_ATTEMPTS ]; then
                local error_output=$(eval "$check_command" 2>&1 || true)
                log "💥 ERROR: $service_name failed all $MAX_ATTEMPTS attempts"
                log "🔎 Error details: $error_output"
                return 1
            fi
            log "⏳ $service_name not ready yet (attempt $attempt/$MAX_ATTEMPTS), waiting ${WAIT_TIME}s..."
            sleep $WAIT_TIME
        fi
        attempt=$((attempt + 1))
    done
}

log "🔍 Starting health checks for services..."

for service in "${SERVICES[@]}"; do
    IFS=':' read -r name command <<< "$service"

    if ! check_service "$name" "$command"; then
        log "💀 === Health Check FAILED ==="
        exit 1
    fi
done

log "🎉 All services are ready!"
exit 0
