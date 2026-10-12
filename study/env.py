"""Shared Luna execution bank; interpreted utterance, never intended action, drives transitions."""
import json
from dataclasses import asdict
import numpy as np
from gymnasium import spaces
from study import OUT
from pilot.env import NegotiationEnv,Config,Public,features,observations,accepts,posterior,advance,own_utility

KINDS=['ambiguity','signal','query','offer','ratify','withdraw','invalid']
RESPONSE_TEXT={'hard':'Privately, we require a larger reciprocal concession before considering acceptance.',
               'soft':'Privately, a more limited reciprocal concession may meet our acceptance conditions.',
               'accept':'We accept the proposal subject to its stated reciprocal terms.',
               'reject':'We do not accept those reciprocal terms.',
               'ack':'We acknowledge your statement.'}

class LunaEnv(NegotiationEnv):
    def __init__(self,split='train',mask=None,remove_reward=None,disable_query=False,configs=None,bank=None):
        self.contexts=[x for x in json.loads((OUT/'contexts.json').read_text(encoding='utf-8')) if x['split']==split]
        numeric=json.loads((OUT/'numeric_conditions.json').read_text())[split]
        super().__init__(configs or [Config(**x) for x in numeric])
        self.bank=bank if bank is not None else json.loads((OUT/'language_bank.json').read_text(encoding='utf-8'))
        self.lookup={(z['context_id'],z['action'],z['variant']):z for z in self.bank}
        self.mask=mask; self.remove_reward=remove_reward; self.disable_query=disable_query
        self.observation_space=spaces.Box(0,1,(31,),np.float32)
        self.context=self.contexts[0]; self.variant=0; self.last_kind='ambiguity'
    def obs(self):
        a=features(self.cfg,self.s,self.mask=='belief')
        # Remaining declared costs and style are public; source identity isn't a shortcut.
        b=[self.cfg.signal_cost,self.cfg.turn_cost,self.cfg.escalation_weight,self.cfg.reversal_cost]
        return np.r_[a,b,np.eye(3)[self.variant],np.eye(7)[KINDS.index(self.last_kind)]].astype(np.float32)
    def reset(self,seed=None,options=None):
        options=options or {}; super().reset(seed=seed,options=options)
        self.context=options.get('context',self.contexts[int(self.np_random.integers(len(self.contexts)))])
        self.variant=int(options.get('variant',self.np_random.integers(3)))
        self.last_kind='ambiguity'
        return self.obs(),{}
    def step(self,action):
        if self.done: raise RuntimeError('reset required')
        z=self.lookup[(self.context['id'],int(action),self.variant)]
        m=z['meaning'] if not(self.disable_query and int(action)==2) else {'kind':'invalid'}
        return self.step_meaning(m,z['utterance'],int(action),z['id'])
    def step_meaning(self,m,text,action=-1,bank_id=None):
        before=self.s; rs=observations(self.cfg,before,m)
        if len(rs)==1: y,ph,ps=rs[0]
        elif m['kind']=='offer': y,ph,ps=rs[0 if accepts(self.cfg,before,m['x'],m['y'],self.hard) else 1]
        else:
            prob=self.cfg.reliability if self.hard else 1-self.cfg.reliability
            y,ph,ps=rs[0 if self.query_uniform<prob else 1]
        b,_=posterior(before.b,ph,ps); self.s,reward,self.done,info=advance(self.cfg,before,m,y,b)
        info.update(utterance=text,meaning=m,response=y,before=asdict(before),after=asdict(self.s),
                    action=action,bank_id=bank_id,common_reward=reward,context=self.context['id'],
                    opponent_utterance=RESPONSE_TEXT[y],opponent_text_origin='fixed synthetic template')
        self.last_kind=m['kind']
        train_reward=reward+info['costs'].get(self.remove_reward,0.)
        return self.obs(),train_reward,self.done,False,info
