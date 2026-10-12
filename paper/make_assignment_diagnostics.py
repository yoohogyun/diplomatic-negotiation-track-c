"""Reaggregate existing logs for the course appendix; no training or model calls."""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
R=P.parent
OUT=P/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
mat=json.loads((R/'study_results/transition_matrices.json').read_text())
groups=[0,1,2,3,3,3,3,3,3,4,5]
labels=['A','S','Q','O','R','W']
configs=[('Bandit, seed 17','test_Bandit_17'),('Heuristic','test_Heuristic_None'),('PPO, seed 17','test_PPO_base_17'),('Planner','test_Planner_None')]
fig,axes=plt.subplots(1,4,figsize=(6.3,1.85))
fig.subplots_adjust(left=.065,right=.925,bottom=.23,top=.82,wspace=.3)
count_rows=[]
for ax,(title,key) in zip(axes,configs):
    counts=np.zeros((6,6),dtype=int)
    for i,row in enumerate(mat[key]['counts']):
        for j,n in enumerate(row):counts[groups[i],groups[j]]+=n
    den=counts.sum(1)
    probs=np.divide(counts,den[:,None],out=np.full((6,6),np.nan),where=den[:,None]!=0)
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#E1E1E1')
    im=ax.imshow(probs,vmin=0,vmax=1,cmap=cmap)
    ax.set_title(title,fontsize=8)
    ax.set_xticks(range(6),labels);ax.set_yticks(range(6),labels)
    ax.tick_params(length=0,labelsize=7)
    ax.set_xlabel('Next action',fontsize=7)
    for i in range(6):
        for j in range(6):
            count_rows.append({'policy':title,'from':labels[i],'to':labels[j],'count':int(counts[i,j]),'probability':None if np.isnan(probs[i,j]) else float(probs[i,j])})
axes[0].set_ylabel('Current action',fontsize=7)
cax=fig.add_axes([.945,.31,.012,.44]);fig.colorbar(im,cax=cax,ticks=[0,.5,1]);cax.tick_params(labelsize=7)
fig.savefig(OUT/'assignment_transitions.pdf');fig.savefig(OUT/'assignment_transitions.png',dpi=220);plt.close(fig)
with (P/'assignment_transition_counts.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=count_rows[0].keys());w.writeheader();w.writerows(count_rows)

fig,axes=plt.subplots(1,3,figsize=(6.3,1.8))
fig.subplots_adjust(left=.07,right=.985,bottom=.3,top=.8,wspace=.42)
for seed,color in [(17,'#087F83'),(29,'#CF8A44'),(43,'#456CA0')]:
    logs=list(csv.DictReader((R/f'study_results/training_logs/base_{seed}/progress.csv').open()))
    for ax,field,sign in zip(axes,['rollout/ep_rew_mean','train/entropy_loss','train/value_loss'],[1,-1,1]):
        rows=[r for r in logs if r[field]]
        ax.plot([float(r['time/total_timesteps'])/1000 for r in rows],[sign*float(r[field]) for r in rows],linewidth=.7,color=color,label=str(seed))
for ax,title in zip(axes,['Training return','Policy entropy','Value loss']):
    ax.set_title(title,fontsize=8);ax.set_xlabel('Training steps (thousands)',fontsize=7)
    ax.tick_params(labelsize=7);ax.spines[['top','right']].set_visible(False)
axes[0].legend(fontsize=6,frameon=False,loc='lower right')
fig.savefig(OUT/'assignment_learning.pdf');fig.savefig(OUT/'assignment_learning.png',dpi=220);plt.close(fig)

summ={r['policy']:r for r in csv.DictReader((R/'study_results/summary.csv').open())}
rows=[]
for k,label in [('Random','Random'),('Bandit','Bandit'),('Heuristic','Heuristic'),('PPO_base','PPO'),('Planner','Planner')]:
    r=summ[k]
    vals=[float(r[m]) for m in ['own_utility','opponent_utility','cost','breakdown','length','max_escalation','mean_escalation','reversals','acceptance_violations']]
    rows.append(label+' & '+' & '.join(f'{v:.3f}' for v in vals)+r' \\')
(P/'assignment_metrics_table.tex').write_text('\\begin{tabular}{lrrrrrrrrr}\n\\toprule\nPolicy & Own & Other & Cost & Break & Turns & Max $e$ & Mean $e$ & Reverse & Viol. \\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')
sens=json.loads((R/'study_results/sensitivity_metrics.json').read_text())
rows=[]
for field,values,symbol in [('reliability',[.5,.7,.9,1.0],'q'),('signal_effect',[0,.1,.25],r'\delta')]:
    for value in values:
        subset=[r for r in sens if r['field']==field and r['value']==value]
        planner=next(r for r in subset if r['policy']=='Planner')
        ppo=np.mean([r['net_utility'] for r in subset if r['policy']=='PPO'])
        rows.append(f'${symbol}={value:.2f}$ & {planner["net_utility"]:.3f} & {planner["queries"]:.2f} & {ppo:.3f}'+r' \\')
(P/'assignment_sensitivity_table.tex').write_text('\\begin{tabular}{lrrr}\n\\toprule\nChanged rule & Planner & Questions & PPO \\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')
print('Generated two figures, grouped counts/probabilities, and two table bodies from saved results.')
