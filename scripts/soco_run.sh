#!/bin/bash
# usage: [OVR=..] [SOCO_ARM=..] [SOCO_OUT=..] [TAG=..] [MAXP=20] [TIMEOUT=600] scripts/soco_run.sh <seq> <split> <backbone> <track> <status> "<description>"
seq=$1; split=$2; bb=$3; track=$4; status=$5; desc=$6
commit=$(git rev-parse --short HEAD)
d=runs/soco/${seq}_${commit}_${split}_${bb}${TAG:+_$TAG}
mkdir -p $d
export PYTHONDONTWRITEBYTECODE=1
timeout ${TIMEOUT:-600} python scripts/soco_eval.py $split $bb $d ${MAXP:-20} > $d/run.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "rc=$rc"; tail -3 $d/run.log; exit $rc; fi
python - "$d" "$seq" "$commit" "$track" "$status" "$desc" <<'PY'
import json, sys, os
d, seq, commit, track, status, desc = sys.argv[1:7]
m = json.load(open(f'{d}/soco_metrics.json'))
r, b = m['results'], m['boot']
a, o = r['argmax'], r.get('ours', r['argmax'])
dv = r.get('dv0.5', {'SOC': float('nan'), 'CC': float('nan')})
row = [seq, commit, m['split'], m['backbone'], str(m['n_pairs']), f"{o['SOC']:.2f}", f"{o['CC']:.2f}",
       f"{o['SOC']-a['SOC']:+.2f}", f"{b.get('ours', {}).get('dSOC_std', 0):.2f}", f"{o['CC']-a['CC']:+.2f}",
       f"{b.get('ours', {}).get('dCC_std', 0):.2f}", f"{dv['SOC']-a['SOC']:+.2f}", f"{dv['CC']-a['CC']:+.2f}",
       f"{a['SOC']:.2f}", f"{a['CC']:.2f}", f"{m['sec']:.0f}", track, status,
       desc + f" [ovr {m['ovr']}; arm {m['arm']}; out {m['out']}; dir {os.path.basename(d)}]"]
fn = 'soco_results.tsv'
if not os.path.exists(fn):
    open(fn, 'w').write('seq\tcommit\tsplit\tbackbone\tn_pairs\tSOC\tCC\tdSOC\tdSOC_std\tdCC\tdCC_std\tdv_dSOC\tdv_dCC\targmax_SOC\targmax_CC\tsec\ttrack\tstatus\tdescription\n')
open(fn, 'a').write('\t'.join(row) + '\n')
print('\t'.join(row[:16]))
PY
