from pathlib import Path
import sys
import csv
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pilot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=ROOT/'paper/figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
navy='#193849'; teal='#187f80'; gray='#56646e'
# The full-width teaser is authored in make_teaser.mjs and exported from PowerPoint.

with (ROOT/'improvement_results/summary.csv').open() as f:
    rows={r['policy']:r for r in csv.DictReader(f)}
base=float(rows['distilled_ppo']['net_utility'])
labels=['Without signals','Belief masked','Without questions']
keys=['distilled_ppo_no_signal','distilled_ppo_masked_belief','distilled_ppo_no_query']
loss=[base-float(rows[k]['net_utility']) for k in keys]
fig,ax=plt.subplots(figsize=(3.15,1.48))
fig.subplots_adjust(left=.38,right=.95,bottom=.31,top=.91)
ax.barh([2,1,0],loss,color=[teal,'#49768f','#a2b5c0'],height=.55)
ax.set_yticks([2,1,0],labels,fontsize=7.8)
ax.set_xlim(0,.115); ax.set_xticks([0,.05,.10],['0','.05','.10'])
ax.tick_params(axis='both',length=0,labelsize=7.5)
ax.set_xlabel('Decrease in mean net utility',fontsize=8,labelpad=5)
ax.spines['left'].set_visible(False); ax.spines['bottom'].set_color('#acb6bc')
for y,v in zip([2,1,0],loss):
    ax.text(v+.003,y,f'{v:.3f}',va='center',fontsize=8,color=navy)
ax.set_axisbelow(True); ax.grid(axis='x',color='#dfe5e8',linewidth=.6)
for ext in ('pdf','svg','png'):
    fig.savefig(OUT/f'intervention_losses.{ext}',dpi=250)
plt.close(fig)
print('Created vector PDF/SVG figures and PNG previews from saved data.')
