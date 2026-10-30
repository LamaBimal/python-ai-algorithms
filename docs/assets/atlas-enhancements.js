// Contrasting data and accessible explanations for the graph laboratory.
const graphExamples={
 default:{graph:{S:{A:1,B:4},A:{C:1},B:{G:1},C:{G:1},G:{}},h:{S:3,A:2,B:0,C:1,G:0},note:'Admissible heuristic. BFS minimizes edge count; UCS and A* minimize weighted cost.'},
 unreachable:{graph:{S:{A:1,B:4},A:{C:1},B:{},C:{},G:{}},h:{S:0,A:0,B:0,C:0,G:0},note:'No route reaches G. Exhausting the frontier is a valid failure result.'},
 cycle:{graph:{S:{A:1,B:4},A:{C:1},B:{G:1},C:{S:1,G:1},G:{}},h:{S:0,A:0,B:0,C:0,G:0},note:'C links back to S. Discovery tracking and best-cost checks prevent endless cycling.'},
 equal:{graph:{S:{A:1,B:1},A:{C:1},B:{G:2},C:{G:1},G:{}},h:{S:0,A:0,B:0,C:0,G:0},note:'Both routes cost 3. Equal priorities use insertion order; optimal paths need not be unique.'},
 inconsistent:{graph:{S:{A:3,B:1},A:{C:1},B:{A:1,G:10},C:{G:1},G:{}},h:{S:0,A:0,B:3,C:0,G:0},note:'h(B)=3 exceeds cost(B,A)+h(A)=1, so the heuristic is inconsistent, yet admissible. A* must accept improved routes and reopen A after B improves its cost.'}
};
const note=document.createElement('p');note.id='graph-note';note.setAttribute('role','status');$('search-svg').after(note);
function describeGraph(){const edges=Object.entries(graph).flatMap(([a,ns])=>Object.entries(ns).map(([b,w])=>`${a} to ${b}, cost ${w}`)).join('; ');$('search-desc').textContent='Directed graph: '+edges+'. Heuristics: '+JSON.stringify(heuristic)+'. Use Next expansion to inspect queued candidates.';const hint=$('search-svg').closest('.panel').querySelector('.hint');if(hint)hint.textContent='Edge labels are costs. Current heuristic: '+Object.entries(heuristic).map(([k,v])=>`${k}=${v}`).join(', ')+'. Frontier entries can include stale candidates.';$('graph-note').textContent=graphExamples[$('graph-preset').value].note;}
$('graph-preset').addEventListener('change',()=>{const example=graphExamples[$('graph-preset').value];graph=example.graph;heuristic=example.h;searchTraces=Object.fromEntries(Object.keys(names).map(k=>[k,trace(k)]));searchIndex=-1;drawSearch();drawComparison();describeGraph();});
function drawFrontier(step,key){const root=$('frontier-detail');if(!root)return;root.replaceChildren();const p=document.createElement('p');p.textContent='Selection rule: '+priorities[key]+'. Ties use insertion order. Stale best-cost candidates are discarded without expansion.';root.append(p);const table=document.createElement('table'),head=document.createElement('tr');for(const label of ['Candidate','g: cost paid','h: estimate','Priority','Status']){const th=document.createElement('th');th.scope='col';th.textContent=label;head.append(th);}table.append(head);const candidates=step.candidates||[{node:'S',cost:0,heuristic:heuristic.S,priority:key==='astar'||key==='greedy'?heuristic.S:0}];for(const c of candidates){const row=document.createElement('tr');for(const value of [c.node,c.cost,c.heuristic,c.priority,c.stale?'stale · ignored if popped':'queued']){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}table.append(row);}root.append(table);}
// Text mirrors SVG changes and is announced without relying on node color.
$('search-state').setAttribute('aria-live','polite');$('search-state').setAttribute('aria-atomic','true');
describeGraph();drawSearch();
function confusionAt(threshold){const scores=[.95,.8,.65,.55,.45,.35,.2,.05],truth=[1,0,1,0,1,0,1,0];let tp=0,fp=0,tn=0,fn=0;scores.forEach((s,i)=>{if(s>=threshold){if(truth[i])tp++;else fp++;}else{if(truth[i])fn++;else tn++;}});return {tp,fp,tn,fn,precision:tp+fp?tp/(tp+fp):null,recall:tp+fn?tp/(tp+fn):null};}
function drawConfusion(){const t=Number($('threshold').value),m=confusionAt(t);$('threshold-value').textContent=t.toFixed(2);$('confusion').textContent=`TP ${m.tp} · FP ${m.fp} · TN ${m.tn} · FN ${m.fn}. Precision ${m.precision===null?'undefined':m.precision.toFixed(2)}; recall ${m.recall===null?'undefined':m.recall.toFixed(2)}.`;}
$('threshold').addEventListener('input',drawConfusion);drawConfusion();

for(const card of cards){
 const heading=card.querySelector('h3');const link=document.createElement('a');link.href='#'+card.id;link.textContent='Link to this method';link.setAttribute('aria-label','Link to '+heading.textContent);card.lastElementChild.append(link);
 const label=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.dataset.complete=card.id;label.append(box,document.createTextNode(' Mark '+heading.textContent+' complete on this device'));card.lastElementChild.append(label);
}
function revealMethod(){const id=location.hash.slice(1),card=cards.find(c=>c.id===id);if(!card)return;$('query').value='';document.querySelector('[data-filter="All"]').click();card.scrollIntoView({block:'start'});}
window.addEventListener('hashchange',revealMethod);revealMethod();
