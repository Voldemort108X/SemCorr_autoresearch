#!/usr/bin/env bash
# discard the last experiment commit but keep results.tsv rows (program.md §3)
cd "$(dirname "$0")/.."
cp results.tsv /tmp/claude-0/results.tsv.bak 2>/dev/null || cp results.tsv results.tsv.bak
git reset --hard HEAD~1
cp /tmp/claude-0/results.tsv.bak results.tsv 2>/dev/null || mv results.tsv.bak results.tsv
