import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const ROOT=process.cwd();
process.env.RUNTIME_NODE_MODULES='C:/Users/user/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const OUT=path.join(ROOT,'release/diplomatic-negotiation-track-c/paper/figures');
const BUILD=path.join(ROOT,'tmp/teaser_pptx');
const SKILL='C:/Users/user/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations';
const PYTHON='C:/Users/user/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const {applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const p=Presentation.create({slideSize:{width:1680,height:560}});
const s=p.slides.add(); s.background.fill='#FFFFFF';
const C={ink:'#15374F', teal:'#087F83', pale:'#DDF1F2', blue:'#DCEBF6', amber:'#F8E3AE', grey:'#597080'};
function shape(name,x,y,w,h,fill,geometry='rect',stroke=C.ink){
 return s.shapes.add({name,geometry,position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,style:'solid',width:stroke==='none'?0:2}});
}
function text(name,t,x,y,w,h,size=28,bold=false,align='left',color=C.ink){
 const a=shape(name,x,y,w,h,'none','textbox','none'); a.text=t;
 a.text.style={fontFamily:'Arial',typeface:'Arial',fontSize:size,bold,color,alignment:align,verticalAlignment:'middle',insets:0,autoFit:'none'};
 return a;
}
function box(name,t,x,y,w,h,fill=C.blue,size=28){
 const a=shape(name,x,y,w,h,fill,'roundRect'); a.borderRadius=10;
 a.text=t;a.text.style={fontFamily:'Arial',typeface:'Arial',fontSize:size,bold:true,color:C.ink,alignment:'center',verticalAlignment:'middle',insets:8,autoFit:'none'};return a;
}
function arrow(name,x,y,w,h,dir='right',color=C.ink){return shape(name,x,y,w,h,color,dir+'Arrow','none');}
function line(name,x,y,w,h,color='#B8C9D4'){return shape(name,x,y,w,h,'none','line',color);}
text('heading-a','A  Real diplomatic context',20,12,410,44,31,true);
text('heading-b','B  Explicit strategic control',464,12,635,44,31,true);
text('heading-c','C  Matched utility',1150,12,510,44,31,true);
line('divider-a',443,10,0,524);line('divider-b',1127,10,0,524);

// Panel A: editable document icon and fixed language preparation.
shape('document-back',36,96,66,83,'#FFFFFF','roundRect');
shape('document-middle',47,86,66,83,'#FFFFFF','roundRect');
shape('document-front',58,76,66,83,'#FFFFFF','roundRect');
for(let i=0;i<4;i++)line('document-line-'+i,71,94+i*14,39,0,C.ink);
text('source-count','12 documents\n6 event groups',145,79,270,91,30,true);
text('source-examples','Ceasefires and\nreciprocal measures',26,185,385,71,28);
arrow('offline-arrow',199,260,25,39,'down');
box('language-bank','GPT-5.6 Luna\n396 fixed utterances\n+ interpreted meanings',27,310,386,126,C.pale,29);
text('offline-label','Prepared once, offline',26,447,385,38,27,false,'center');
text('data-scope','Sources supply context;\nnumerical rules are synthetic.',26,489,395, sixty(),25);
function sixty(){return 60;}

// Panel B: two alternative controllers; only the teacher-to-student arrow is distillation.
box('public-state','Public state: belief, commitments, costs',472,84,606,72,C.blue,28);
arrow('state-to-planner',576,166,24,36,'down');
arrow('state-to-ppo',968,166,24,36,'down');
box('planner','Belief-state\nplanner',472,216,252,84,C.pale,29);
box('student','Distilled PPO',868,216,210,84,C.pale,29);
arrow('distillation',745,245,101,19);
text('teacher-label','distill',744,207,105,33,25,false,'center');
text('alternative-label','Choose one controller',580,307,423,34,26,false,'center',C.grey);
arrow('planner-execution',587,307,20,68,'down');
// Student route reaches the same execution box below the controller row.
line('student-route-1',972,300,0,54,C.ink);
line('student-route-2',659,354,313,0,C.ink);
arrow('student-route-3',649,354,20,22,'down');
box('execute-fixed','Execute fixed\nLuna utterance',472,385,252,91,C.blue,28);
arrow('utterance-to-counterpart',748,416,90,23);
box('counterpart','Synthetic\ncounterpart',868,385,210,91,C.amber,28);
// Feedback runs around the outside of the controller graph.
line('feedback-up',1102,119,0,311,C.ink);
line('feedback-start',1078,430,24,0,C.ink);
arrow('feedback-end',1078,108,24,22,'left');
text('feedback-label','Observed reply → Bayesian update',473,490,605,38,27,false,'center');
text('hidden-type-note','Hidden type is never visible to the controller.',473,530,610,25,23,false,'center',C.grey);

text('matched-subtitle','64 matched conditions',1150,71,505,42,28);
const chart=s.charts.add('bar',{
 position:{left:1140,top:128,width:523,height:300},
 categories:['Distilled PPO','Sol medium','Planner'],
 series:[{name:'Net utility',values:[0.475317708,0.514011719,0.533428125],fill:C.teal,points:[{idx:2,fill:C.teal},{idx:1,fill:'#8EBAD7'},{idx:0,fill:'#9CD0D1'}]}],
 barOptions:{direction:'bar',grouping:'clustered',gapWidth:72},hasLegend:false,
 xAxis:{visible:true,min:0,max:0.6,majorUnit:0.2,numberFormatCode:'0.0',textStyle:{fontSize:24,fill:C.ink},line:{fill:C.ink,width:1,style:'solid'},majorGridlines:null},
 yAxis:{visible:true,min:0,max:0.6,majorUnit:0.2,numberFormatCode:'0.0',textStyle:{fontSize:25,fill:C.ink},line:{fill:C.ink,width:1,style:'solid'},majorGridlines:null},
 dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{fontSize:26,bold:true,fill:C.ink}},
 chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'
});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('utility-note','Net utility (higher is better)',1150,429,505,38,27,false,'center');
text('calls','Local controllers: 0 further\nmodel calls to choose actions',1150,481,505,72,28,true,'center',C.teal);
s.speakerNotes.text='Editable redraw of the generated concept teaser_concept.png. All diagram elements, text and the chart are native editable objects. The diagram shows alternative local controllers, a planner-to-PPO distillation arrow, and fixed offline language execution. No online language generation is implied. Values: planner 0.533428125, Sol medium 0.514011719, distilled PPO mean 0.475317708. Sources: sol_results/matched_policy_controls.csv, sol_results/direct_comparison.csv, improvement_results/summary.csv, paper/methods_results.tex. Costs shown refer only to online action selection, excluding offline preparation and training. 64 document–configuration conditions, not 64 historical events.';
await fs.mkdir(OUT,{recursive:true});
async function save(file,blob){await fs.writeFile(file,new Uint8Array(await blob.arrayBuffer()));}
await save(path.join(BUILD,'teaser_preview.png'),await p.export({slide:s,format:'png',scale:1.5}));
// PDF is exported from the finalized PPTX using installed PowerPoint.
await fs.writeFile(path.join(BUILD,'teaser.layout.json'),await(await s.export({format:'layout'})).text());
const candidatePath=path.join(BUILD,'teaser_candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath,finalPath:path.join(OUT,'teaser_editable_v2.pptx'),pythonExecutable:PYTHON,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','16002000,5334000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[1],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'teaser.v2.validation.json')});
console.log(JSON.stringify(result));



