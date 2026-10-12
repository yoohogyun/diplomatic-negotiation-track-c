"""Five-opportunity diplomatic reciprocal-suspension simulator.

All numerical rules are researcher assumptions, NOT inferred historical facts.
The public-state transition function is shared by Gym and a belief-only planner.
Text is parsed before transition; an action label cannot override its meaning.
"""
from dataclasses import dataclass, replace, asdict
from functools import lru_cache
import re
import numpy as np
import gymnasium as gym
from gymnasium import spaces

NAMES = ['ambiguity', 'signal', 'query'] + [f'offer_{x}_{y}' for x in (1,2,3) for y in (1,2)] + ['ratify','withdraw']

@dataclass(frozen=True)
class Config:
    prior: float = .5
    reliability: float = .9
    query_cost: float = .02
    rejection_cost: float = .08
    signal_effect: float = .1
    signal_cost: float = .06
    hard_cost: float = .45
    outside: float = .12
    turn_cost: float = .01
    escalation_weight: float = .005
    reversal_cost: float = .08
    horizon: int = 5

@dataclass(frozen=True)
class Public:
    t: int = 0
    b: float = .5
    x: int = 0
    y: int = 0
    accepted: bool = False
    signaled: bool = False
    queried: bool = False
    provisional: bool = False
    commitment: int = 0
    escalation: int = 0

def render(a, cfg=Config()):
    if a == 0: return 'Our next proposal is provisional; we make no new public commitment.'
    if a == 1: return f'We commit resources at cost {cfg.signal_cost:g} to a monitored confidence-building measure.'
    if a == 2: return 'In a private channel, do you require a larger reciprocal suspension to accept our proposal?'
    if a == 9: return 'We ratify the currently accepted reciprocal suspension agreement.'
    if a == 10: return 'We withdraw from these negotiations.'
    x,y = map(int,NAMES[a].split('_')[1:])
    return f'We will suspend our large-scale military exercises for {x} weeks if you suspend your nuclear and missile tests for {y} weeks.'

def parse(text):
    s = text.lower().strip()
    if re.search(r'\b(not|never)\b',s): raise ValueError('Negation is outside the controlled language.')
    m = re.fullmatch(r'we will suspend our large-scale military exercises for ([123]) weeks if you suspend your nuclear and missile tests for ([12]) weeks\.',s)
    if m: return {'kind':'offer','x':int(m[1]),'y':int(m[2])}
    m = re.fullmatch(r'we commit resources at cost (\d+(?:\.\d+)?) to a monitored confidence-building measure\.',s)
    if m and 0 <= float(m[1]) <= 1: return {'kind':'signal','cost':float(m[1])}
    for a in (0,2,9,10):
        if s == render(a).lower(): return {'kind':NAMES[a]}
    raise ValueError('Uninterpretable controlled utterance.')

def own_utility(x,y): return .5*y - .18*x
def other_utility(cfg,x,y,hard,signaled):
    return .35*x - (cfg.hard_cost if hard else .2)*y + cfg.signal_effect*int(signaled)
def accepts(cfg,s,x,y,hard): return other_utility(cfg,x,y,hard,s.signaled) >= -1e-10

def posterior(b,lh,ls):
    p = b*lh+(1-b)*ls
    if p <= 0: raise ValueError('Impossible observation under the declared model.')
    return round(b*lh/p,10),p

def observations(cfg,s,meaning):
    """Response probability under each type; no actual private type input."""
    if meaning['kind']=='query' and not s.queried:
        q=cfg.reliability
        return [('hard',q,1-q),('soft',1-q,q)]
    if meaning['kind']=='offer':
        h=float(accepts(cfg,s,meaning['x'],meaning['y'],True))
        l=float(accepts(cfg,s,meaning['x'],meaning['y'],False))
        return [('accept',h,l),('reject',1-h,1-l)]
    return [('ack',1.,1.)]

def advance(cfg,s,meaning,response,b):
    """Transition determined by interpreted words/numbers and observed response."""
    k=meaning['kind']; n=replace(s,t=s.t+1,b=b)
    c={'turn':cfg.turn_cost,'query':0.,'signal':0.,'rejection':0.,'escalation':0.,'reversal':0.,'invalid':0.}
    done=False; outcome='ongoing'; terminal=0.; reversal=0
    if k=='query':
        c['query']=cfg.query_cost; n=replace(n,queried=True)
    elif k=='signal':
        c['signal']=meaning['cost']; n=replace(n,signaled=True)
    elif k=='ambiguity': n=replace(n,provisional=True)
    elif k=='offer':
        x,y=meaning['x'],meaning['y']
        reversal=int(x<s.commitment)
        c['reversal']=cfg.reversal_cost*reversal
        reject=response=='reject'
        c['rejection']=cfg.rejection_cost*reject
        n=replace(n,x=x,y=y,accepted=not reject,provisional=False,
                  commitment=s.commitment if s.provisional else x,
                  escalation=s.escalation+int(reject))
    elif k=='ratify':
        if s.accepted and own_utility(s.x,s.y)>=cfg.outside-1e-10:
            done=True; outcome='agreement'; terminal=own_utility(s.x,s.y)
        else: c['invalid']=.1
    elif k=='withdraw': done=True; outcome='withdrawal'; terminal=cfg.outside
    elif k=='invalid': c['invalid']=.1
    c['escalation']=cfg.escalation_weight*n.escalation**2
    if n.t>=cfg.horizon and not done:
        done=True; outcome='exhausted'; terminal=cfg.outside
    return n,terminal-sum(c.values()),done,dict(costs=c,terminal_utility=terminal,outcome=outcome,reversal=reversal)

def branches(cfg,s,a):
    m=parse(render(a,cfg)); out=[]
    for y,ph,ps in observations(cfg,s,m):
        p=s.b*ph+(1-s.b)*ps
        if p>1e-12:
            b,_=posterior(s.b,ph,ps)
            n,r,d,i=advance(cfg,s,m,y,b)
            out.append((p,n,r,d))
    return out

class Planner:
    """Exact public-model upper benchmark, never receives the true type."""
    def __init__(self,cfg,allow_query=True):
        self.cfg=cfg; self.actions=tuple(a for a in range(11) if allow_query or a!=2)
        self.solve=lru_cache(maxsize=None)(self._solve)
    def _solve(self,s):
        vals=[]
        for a in self.actions:
            v=sum(p*(r+(0 if d else self.solve(n)[0])) for p,n,r,d in branches(self.cfg,s,a))
            vals.append((v,a))
        return max(vals,key=lambda v:(round(v[0],12),-v[1]))
    def action(self,s): return self.solve(s)[1]

def features(cfg,s,mask=False):
    return np.array([s.t/5, cfg.prior if mask else s.b,s.x/3,s.y/2,float(s.accepted),float(s.signaled),float(s.queried),
                     float(s.provisional),s.commitment/3,s.escalation/5,cfg.prior,cfg.reliability,cfg.query_cost,
                     cfg.rejection_cost,cfg.signal_effect,cfg.hard_cost,cfg.outside],dtype=np.float32)

class NegotiationEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,configs=None,mask_belief=False):
        self.configs=configs or [Config()]; self.mask_belief=mask_belief
        self.action_space=spaces.Discrete(11)
        self.observation_space=spaces.Box(0,1,(17,),np.float32)
        self.cfg=self.configs[0]; self.s=Public(); self.done=False
    def reset(self,seed=None,options=None):
        super().reset(seed=seed); options=options or {}
        self.cfg=options.get('config',self.configs[int(self.np_random.integers(len(self.configs)))])
        self.s=Public(b=self.cfg.prior); self.done=False
        self.hard=bool(options.get('hard',self.np_random.random()<self.cfg.prior))
        self.query_uniform=float(options.get('query_uniform',self.np_random.random()))
        return features(self.cfg,self.s,self.mask_belief),{}
    def step(self,action): return self.step_text(render(int(action),self.cfg))
    def step_text(self,text):
        if self.done: raise RuntimeError('reset required after termination')
        before=self.s
        try: m=parse(text)
        except ValueError: m={'kind':'invalid'}
        responses=observations(self.cfg,self.s,m)
        if len(responses)==1: y,ph,ps=responses[0]
        elif m['kind']=='offer':
            yes=accepts(self.cfg,self.s,m['x'],m['y'],self.hard)
            y,ph,ps=responses[0 if yes else 1]
        else:
            p=self.cfg.reliability if self.hard else 1-self.cfg.reliability
            y,ph,ps=responses[0 if self.query_uniform<p else 1]
        b,_=posterior(self.s.b,ph,ps)
        self.s,r,self.done,info=advance(self.cfg,self.s,m,y,b)
        info.update(utterance=text,meaning=m,response=y,before=asdict(before),after=asdict(self.s))
        return features(self.cfg,self.s,self.mask_belief),r,self.done,False,info
