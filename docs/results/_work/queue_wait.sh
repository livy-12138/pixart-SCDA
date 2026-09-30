#!/bin/bash
# Queue-chain waiting helpers.
#
# WHY THIS EXISTS
# ---------------
# The chain scripts coordinate by grepping a DONE marker out of a log that is
# APPENDED to across runs:
#
#     until grep -q "DITPAIR_DONE" "$L/ditpair_queue.log"; do sleep 120; done
#
# A marker is not one-shot, so that grep matches the PREVIOUS run's marker the
# instant the script starts.  On 2026-09-20 this bit twice: pairreplace_queue.sh
# began training on top of a still-running b2fg job (both need ~30 GB, so an OOM
# was likely), and a monitor reported a "DITPAIR_DONE" that was three hours old.
#
# THE RULE
# --------
# Only lines written AFTER this script started count as signals.  Record the
# log's line count at start and search the tail beyond it.  Waits also check
# that the producer is still alive, so a dead chain fails loudly instead of
# hanging forever.
#
# USAGE
# -----
#   source results/_work/queue_wait.sh
#   wait_for_new_marker "$L/b2fg_queue.log" "PROBE_DONE" "[b]2fg_queue.sh"

# wait_for_new_marker <log> <marker> [producer_pgrep_pattern]
# Blocks until <marker> appears in lines appended to <log> after this call.
# Returns 0 on the marker, 2 if the producer vanished first.
wait_for_new_marker() {
    local log="$1" marker="$2" producer="${3:-}"
    local lines0
    lines0=$(wc -l < "$log" 2>/dev/null || echo 0)

    while true; do
        if [ -f "$log" ] && tail -n +$((lines0 + 1)) "$log" 2>/dev/null | grep -q -- "$marker"; then
            return 0
        fi
        if [ -n "$producer" ] && ! pgrep -f "$producer" >/dev/null 2>&1; then
            echo "WARNING: producer '$producer' is gone without '$marker' $(date -Is)" >&2
            return 2
        fi
        sleep 120
    done
}
