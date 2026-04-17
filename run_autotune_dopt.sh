#!/bin/bash
# Wrapper to fire one autotune Claude session for the D-optimal run from cron.
#
# Cron environments are minimal: set PATH so both `claude` and `julia` are
# findable, cd into the project, and use flock to prevent overlapping fires.

export PATH=/home/zlin/julia-1.12.5/bin:/home/zlin/.local/bin:/usr/local/bin:/usr/bin:/bin
export HOME=/home/zlin
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

LOCK=/tmp/bfimgaussian_autotune_dopt.lock
LOG=/home/zlin/BFIMGaussian/autotune_dopt_cron.log

exec flock -n "$LOCK" bash -c '
cd /home/zlin/BFIMGaussian || exit 1
echo "==================== $(date -u +%Y-%m-%dT%H:%M:%SZ) dopt autotune fire ====================" >> "'"$LOG"'"
claude -p --permission-mode acceptEdits "$(cat autotune_dopt_prompt.md)" >> "'"$LOG"'" 2>&1
echo "" >> "'"$LOG"'"
'
