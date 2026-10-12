"""Draw selected saved paths; no model calls or new experiments.

Source: small_model_results/outline_matched_discourse.json,
the Iran ir1/condition 0/style 0/soft counterpart trace;
learned policy seed 17. Full speech-act discussion in appendix_experiments.tex.
"""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pilot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig, ax = plt.subplots(figsize=(3.15, 1.48))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0,1), ylim=(0,1)); ax.axis('off')
rows = [
 ('Luna none', '1:2 rejected ×4 → Withdraw', '−0.80', '#884F40'),
 ('Planner / Sol medium', 'Query → Signal → 1:2 → Ratify', '0.70', '#087F83'),
 ('Distilled PPO', 'Signal → 2:2 → Ratify', '0.55', '#37627D'),
]
traces = {r['label']: r for r in json.loads((ROOT/'small_model_results/outline_matched_discourse.json').read_text(encoding='utf-8'))}
assert abs(traces['Planner']['net_utility'] - traces['Sol medium']['net_utility']) < 1e-10
for index, key in enumerate(['Luna none', 'Planner', 'Learned17']):
    name, route, _, color = rows[index]
    value = f"{traces[key]['net_utility']:.2f}".replace('-', '−')
    rows[index] = (name, route, value, color)
for y, (name, route, value, color) in zip([.88,.56,.24],rows):
    ax.text(.01,y,name,fontsize=8,fontweight='bold',color=color,va='center')
    ax.text(.01,y-.13,route,fontsize=7.9,color='#193849',va='center')
    ax.text(.99,y-.06,value,fontsize=9,fontweight='bold',color=color,va='center',ha='right')
for y in [.68,.36]: ax.plot([.01,.99],[y,y],color='#CCD8DF',linewidth=.5)
for ext in ['pdf','svg','png']:
    fig.savefig(ROOT/'paper/figures'/f'discourse_paths.{ext}',dpi=300)
plt.close(fig)
print('Saved discourse_paths figure from documented trace values.')
