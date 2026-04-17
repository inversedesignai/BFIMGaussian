#!/bin/bash
# notify.sh — send an autotune alert and append to the local alert log.
#
# Usage:
#   ./notify.sh "subject"  "body"
#
# Default channel: ntfy.sh (free, no setup). To receive alerts, either:
#   1. Install the ntfy app (iOS/Android) and subscribe to the topic below, OR
#   2. Open https://ntfy.sh/<TOPIC> in a browser and leave the tab open.
#
# The topic below was generated randomly at setup; keep it private.
# To change channels (email, Slack webhook, Telegram), edit the CHANNEL block.

NTFY_TOPIC="bfimg-autotune-19947d10c5a4"
NTFY_URL="https://ntfy.sh/$NTFY_TOPIC"

SUBJECT="${1:-autotune alert}"
BODY="${2:-(no body)}"

# ── Local log (always) ──────────────────────────────────────────────────────
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $SUBJECT" >> /home/zlin/BFIMGaussian/autotune_alerts.log
echo "    $BODY" >> /home/zlin/BFIMGaussian/autotune_alerts.log

# ── Channel: ntfy.sh ────────────────────────────────────────────────────────
curl -s --max-time 10 -X POST \
    -H "Title: $SUBJECT" \
    -H "Priority: default" \
    -H "Tags: warning" \
    -d "$BODY" \
    "$NTFY_URL" > /dev/null 2>&1 &

# ── Optional: local mail (uncomment to also email — requires working MTA) ──
# echo "$BODY" | mail -s "$SUBJECT" zlin@localhost

# ── Optional: Slack webhook (uncomment + fill SLACK_URL) ───────────────────
# SLACK_URL="https://hooks.slack.com/services/T00/B00/XXX"
# curl -s --max-time 10 -X POST -H 'Content-type: application/json' \
#     -d "{\"text\":\"*$SUBJECT*\n$BODY\"}" "$SLACK_URL" > /dev/null 2>&1 &

wait
exit 0
