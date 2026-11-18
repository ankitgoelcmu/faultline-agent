"""Post-hoc: on tau2-bench at 20% faults, count lost responses, identical writes repeated after a lost
response, and reads after a lost response, per model, domain and condition. Free; reads results only.

  .venv-tau2/bin/python scripts/tau2_repeats.py
"""
import json,glob,os,collections,sys
sys.path.insert(0,'.')
from lab.tau2_faults import LOST_MSG
key=lambda c: json.dumps({"t":c['name'],"a":c['arguments']},sort_keys=True,default=str)
READ=('get_','search_','list_','find_','calculate','think','transfer_to_human')
for m in ('sonnet','haiku','gemma'):
  for dom in ('airline','retail'):
    for cond in ('baseline','idempotent'):
        d=f'results/tau2/{m}-{dom}-{cond}-p0.2'
        sims=json.load(open(d+'/results.json'))['simulations']
        lost=dup=eps_dup=look=0
        for s in sims:
            calls={}; lostkeys=[]; had=False
            for msg in s['messages']:
                for c in (msg.get('tool_calls') or []):
                    k=key(c); calls[c['id']]=(k,c['name'])
                    if any(k2==k for k2,_ in lostkeys):
                        dup+=1; had=True
                    elif lostkeys and c['name'].startswith(READ): look+=1
                if msg.get('role')=='tool' and LOST_MSG[:30] in str(msg.get('content','')):
                    lost+=1; lostkeys.append(calls.get(msg.get('id'),('?','?')))
            eps_dup+=had
        ok=sum(1 for s in sims if (s.get('reward_info') or {}).get('reward',0)>=1)
        print(f'{m:7}{dom:8}{cond:11} n={len(sims)} solved={ok} ({100*ok/len(sims):.0f}%) lost={lost} repeats_after_lost={dup} episodes_with_repeat={eps_dup} reads_after_lost={look}')
