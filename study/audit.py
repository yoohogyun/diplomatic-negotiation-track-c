"""Offline checks and descriptive failure analysis of every saved evaluation path."""
import collections,gzip,json
from study import OUT
from pilot.live import write

def main():
    counts=collections.Counter(); weights=collections.defaultdict(float)
    stats=collections.defaultdict(lambda:collections.defaultdict(float))
    direct_example=None
    streams=[(p.name,gzip.open(p,'rt',encoding='utf-8')) for p in sorted((OUT/'raw').glob('*.gz'))]
    def inspect(r,key):
        nonlocal direct_example
        tr=r['trajectory']; counts['episodes']+=1;counts['turns']+=len(tr)
        assert len(tr)==r['length'] and 1<=len(tr)<=r['config']['horizon']
        assert abs(sum(t['common_reward'] for t in tr)-r['net_utility'])<1e-8
        assert abs(sum(sum(t['costs'].values()) for t in tr)-r['cost'])<1e-8
        assert abs(r['own_utility']-r['cost']-r['net_utility'])<1e-8
        assert r['acceptance_violations']==0
        for a,b in zip(tr,tr[1:]):assert a['after']==b['before']
        weights[(key,r['context'],r['condition'])]+=r['weight']
        repeated=sum(a['response']=='reject' and a['meaning']['kind']=='offer' and a['meaning']==b['meaning'] for a,b in zip(tr,tr[1:]))
        group=(r['policy'],r.get('seed'),r['event_group'],r['variant'])
        values={'net_utility':r['net_utility'],'agreement':r['agreement'],'queries':r['queries'],
                'repeated_rejected_offer':float(repeated>0),'repeated_rejected_offer_count':repeated,
                'exhaustion':float(r['outcome']=='exhausted'),'invalid_attempts':r['invalid_attempts'],
                'invalid_language':sum(t['meaning']['kind']=='invalid' for t in tr)}
        stats[group]['weight']+=r['weight']
        for k,v in values.items():stats[group][k]+=r['weight']*v
        if r['policy']=='Direct_Luna' and repeated and direct_example is None:direct_example=r
    for key,stream in streams:
        with stream:
            for line in stream:inspect(json.loads(line),key)
    for p in sorted((OUT/'luna_policy_cases').glob('*.json')):
        for r in json.loads(p.read_text(encoding='utf-8')):inspect(r,'Direct_Luna')
    assert all(abs(w-1)<1e-8 for w in weights.values())
    rows=[]
    for (policy,seed,group,variant),v in stats.items():
        rows.append(dict(policy=policy,seed=seed,event_group=group,variant=variant,
                         **{k:x/v['weight'] for k,x in v.items() if k!='weight'}))
    write(OUT/'diagnostics_by_group_style.json',rows)
    write(OUT/'direct_luna_failure_example.json',direct_example)
    result=dict(counts,weighted_context_settings_checked=len(weights),all_reward_and_weight_checks_passed=True,
                includes='Development, main test, rule sensitivities and direct Luna. Repeated synthetic paths, not historical negotiations.')
    write(OUT/'integrity_audit.json',result);print(json.dumps(result))

if __name__=='__main__':main()
