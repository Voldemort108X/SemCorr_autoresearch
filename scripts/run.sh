#!/usr/bin/env bash
# usage: bash scripts/run.sh <seq> [split=dev] [method=tlr]
# writes runs/<seq>_<commit>[_method][_split]/{run.log,err.npz,metrics.json}; prints only the summary lines.
cd "$(dirname "$0")/.."
seq=$1; split=${2:-dev}; method=${3:-tlr}
c=$(git rev-parse --short HEAD)
d=runs/${seq}_${c}
[ "$method" != tlr ] && d=${d}_${method}
[ "$split" != dev ] && d=${d}_${split}
mkdir -p $d
export PYTHONDONTWRITEBYTECODE=1
timeout ${TIMEOUT:-600} python scripts/run_exp.py $seq $split $method $d > $d/run.log 2>&1
rc=$?
echo "rc=$rc dir=$d"
grep "^${split}:\|^pck\|^sec:" $d/run.log
[ $rc -ne 0 ] && tail -n 40 $d/run.log
exit $rc
