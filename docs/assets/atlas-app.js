'use strict';

// Keep source and Markdown links useful when this file is served by GitHub Pages.
if (location.hostname.toLowerCase() === 'lamabimal.github.io') {
 for (const link of document.querySelectorAll('a[href]')) {
  const target=link.getAttribute('href');
  if (target.startsWith('#') || /^(https?:|mailto:)/i.test(target)) continue;
  const repositoryPath=target.startsWith('../') ? target.slice(3) : 'docs/'+target;
  link.href='https://github.com/LamaBimal/python-ai-algorithms/blob/main/'+repositoryPath.split('/').map(encodeURIComponent).join('/');
 }
}

const $ = id => document.getElementById(id);
let graph={S:{A:1,B:4},A:{C:1},B:{G:1},C:{G:1},G:{}};
let heuristic={S:3,A:2,B:0,C:1,G:0};
const positions={S:[65,150],A:[225,60],B:[225,245],C:[390,60],G:[505,150]};
const names={bfs:'BFS',dfs:'DFS',ucs:'UCS',greedy:'Greedy',astar:'A*'};
function trace(strategy){return searchTrace(graph,heuristic,strategy);}
let searchTraces=Object.fromEntries(Object.keys(names).map(k=>[k,trace(k)]));
const ns='http://www.w3.org/2000/svg';
function svgElement(tag,attrs={},text){const el=document.createElementNS(ns,tag);for(const[key,value]of Object.entries(attrs))el.setAttribute(key,String(value));if(text!==undefined)el.textContent=text;return el;}
function addText(svg,x,y,text,attrs={}){svg.append(svgElement('text',{x,y,...attrs},text));}
let searchIndex=-1;
function drawSearch(){
 const key=$('strategy').value,result=searchTraces[key];
 const step=searchIndex<0?{current:null,path:['S'],cost:0,expanded:[],frontier:['S']}:result.snapshots[searchIndex];
 const svg=$('search-svg');svg.querySelectorAll(':scope > g,:scope > defs').forEach(el=>el.remove());
 const defs=svgElement('defs'),marker=svgElement('marker',{id:'edge-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto'});
 marker.append(svgElement('path',{d:'M0 0L10 5L0 10Z',fill:'#7d9385'}));defs.append(marker);svg.append(defs);
 const edges=svgElement('g');
 for(const[a,neighbors]of Object.entries(graph))for(const[b,cost]of Object.entries(neighbors)){
  const[x1,y1]=positions[a],[x2,y2]=positions[b],d=Math.hypot(x2-x1,y2-y1),ux=(x2-x1)/d,uy=(y2-y1)/d;
  const highlighted=step.path.some((n,i)=>n===a&&step.path[i+1]===b);
  edges.append(svgElement('line',{x1:x1+ux*24,y1:y1+uy*24,x2:x2-ux*27,y2:y2-uy*27,stroke:highlighted?'#17654e':'#b6c1b5','stroke-width':highlighted?3:1.5,'marker-end':'url(#edge-arrow)'}));
  edges.append(svgElement('text',{x:(x1+x2)/2+uy*13,y:(y1+y2)/2-ux*13,'font-size':14,'text-anchor':'middle'},cost));
 }
 svg.append(edges);const nodes=svgElement('g');
 for(const[node,[x,y]]of Object.entries(positions)){
  const color=node===step.current?'#efb949':step.expanded.includes(node)?'#9dc7ad':step.frontier.includes(node)?'#9bcbd8':'#dfe4dd';
  nodes.append(svgElement('circle',{cx:x,cy:y,r:23,fill:color,stroke:'#698673','stroke-width':1}));
  nodes.append(svgElement('text',{x,y:y+5,'text-anchor':'middle','font-size':17,'font-weight':600},node));
  nodes.append(svgElement('text',{x,y:y+43,'text-anchor':'middle','font-size':11},'h='+heuristic[node]));
 }
 svg.append(nodes);$('search-state').replaceChildren();const dl=document.createElement('dl');
 for(const[term,value]of [['Current',step.current||'Ready'],['Frontier',step.frontier.join(' → ')||'Empty'],['Expanded',step.expanded.join(' → ')||'None'],['Path',step.path.join(' → ')],['Cost',String(step.cost)],['Status',step.current==='G'?'Goal reached':searchIndex===result.snapshots.length-1?'Goal unreachable':searchIndex<0?'Start is queued':'Continue expanding']]){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=term;dd.textContent=value;dl.append(dt,dd);}
 $('search-state').append(dl);if(typeof drawFrontier==='function')drawFrontier(step,key);$('step-count').textContent=`${searchIndex+1} / ${result.snapshots.length} expansions`;
 $('back').disabled=searchIndex<0;$('next').disabled=searchIndex>=result.snapshots.length-1;
}
$('strategy').addEventListener('change',()=>{searchIndex=-1;drawSearch();});
$('next').addEventListener('click',()=>{searchIndex++;drawSearch();});
$('back').addEventListener('click',()=>{searchIndex--;drawSearch();});
$('reset').addEventListener('click',()=>{searchIndex=-1;drawSearch();});
const priorities={bfs:'Depth',dfs:'Stack order',ucs:'g(n)',greedy:'h(n)',astar:'g(n) + h(n)'};
function drawComparison(){ $('search-comparison').replaceChildren();for(const[key,result]of Object.entries(searchTraces)){
 const tr=document.createElement('tr');
 for(const text of [names[key],priorities[key],result.path?result.path.join(' → '):'Unreachable',result.cost===null?'—':String(result.cost),['ucs','astar'].includes(key)?'Minimum cost':key==='bfs'?'Fewest edges':'No optimality guarantee']){const td=document.createElement('td');td.textContent=text;tr.append(td);}
 $('search-comparison').append(tr);
}}
drawComparison();
drawSearch();
function descent(rate,steps=30){
 const x=[-2,-1,0,1,2],z=x.map(v=>v/Math.sqrt(2)),target=x.map(v=>1+2*v),history=[];
 let b=0,w=0;
 for(let i=0;i<=steps;i++){
  const residual=z.map((value,j)=>b+w*value-target[j]);
  history.push({b,w,loss:residual.reduce((sum,r)=>sum+r*r,0)/x.length});
  b-=rate*2*residual.reduce((a,v)=>a+v,0)/x.length;
  w-=rate*2*residual.reduce((sum,r,j)=>sum+r*z[j],0)/x.length;
 }
 return history;
}
function drawDescent(){
 const rate=Number($('rate').value),iteration=Number($('iteration').value),history=descent(rate),current=history[iteration];
 $('rate-value').value=rate.toFixed(2);$('iteration-value').value=String(iteration);
 const svg=$('fit-svg');svg.replaceChildren();
 const sx=x=>48+(x+2.5)/5*420,sy=y=>180-(y+5)/12*155;
 const defs=svgElement('defs'),clip=svgElement('clipPath',{id:'fit-clip'});clip.append(svgElement('rect',{x:48,y:25,width:420,height:155}));defs.append(clip);svg.append(defs);
 svg.append(svgElement('path',{d:'M48 25V180H468',stroke:'#a6b6a8',fill:'none'}));
 const lines=svgElement('g',{'clip-path':'url(#fit-clip)'});
 const start=current.b+current.w*(-2.5/Math.sqrt(2)),end=current.b+current.w*(2.5/Math.sqrt(2));
 lines.append(svgElement('line',{x1:sx(-2.5),y1:sy(start),x2:sx(2.5),y2:sy(end),stroke:'#17654e','stroke-width':2.5}));
 for(const x of [-2,-1,0,1,2])lines.append(svgElement('circle',{cx:sx(x),cy:sy(1+2*x),r:5,fill:'#efb949',stroke:'#89661b'}));
 svg.append(lines);addText(svg,48,15,'Target points and current prediction',{'font-size':12});addText(svg,390,208,'Feature x',{'font-size':12});
 for(const x of [-2,0,2])addText(svg,sx(x),198,String(x),{'font-size':10,'text-anchor':'middle'});
 for(const y of [-5,1,7])addText(svg,38,sy(y)+4,String(y),{'font-size':10,'text-anchor':'end'});
 const loss=$('loss-svg');loss.replaceChildren();
 const logs=history.map(p=>Math.log10(Math.max(p.loss,1e-12))),lo=Math.min(...logs)-.2,hi=Math.max(...logs)+.2;
 const ly=v=>130-(v-lo)/(hi-lo)*95;
 loss.append(svgElement('path',{d:'M48 28V130H468',stroke:'#a6b6a8',fill:'none'}));
 loss.append(svgElement('polyline',{points:logs.map((v,i)=>`${48+i/30*420},${ly(v)}`).join(' '),fill:'none',stroke:'#17654e','stroke-width':2}));
 loss.append(svgElement('circle',{cx:48+iteration/30*420,cy:ly(logs[iteration]),r:5,fill:'#efb949',stroke:'#89661b'}));
 addText(loss,48,16,'Training MSE (log₁₀ scale; floor 10⁻¹²)',{'font-size':12});
 addText(loss,48,151,'0',{'font-size':10});addText(loss,452,151,'30 steps',{'font-size':10});
 addText(loss,38,ly(hi)+4,hi.toFixed(1),{'font-size':10,'text-anchor':'end'});addText(loss,38,ly(lo)+4,lo.toFixed(1),{'font-size':10,'text-anchor':'end'});
 $('gd-state').textContent=`Step ${iteration}: MSE ${current.loss.toExponential(3)}; intercept ${current.b.toFixed(3)}; original-unit slope ${(current.w/Math.sqrt(2)).toFixed(3)}. `+(rate>1?'This step size diverges on this quadratic; fitted lines may be clipped.':rate===1?'This rate oscillates instead of converging.':'Target: intercept 1, slope 2. Falling training loss is not test performance.');
}
$('rate').addEventListener('input',drawDescent);$('iteration').addEventListener('input',drawDescent);drawDescent();
let activeFilter='All';const cards=[...document.querySelectorAll('.algo')];
function filterAlgorithms(){
 const query=$('query').value.trim().toLocaleLowerCase();let visible=0;
 for(const card of cards){const match=(activeFilter==='All'||card.dataset.category===activeFilter)&&(card.textContent+" "+card.dataset.keywords).toLocaleLowerCase().includes(query);card.hidden=!match;if(match)visible++;}
 $('result-count').textContent=`${visible} of ${cards.length} methods`;$('no-results').hidden=visible!==0;
}
$('query').addEventListener('input',filterAlgorithms);
for(const button of document.querySelectorAll('[data-filter]'))button.addEventListener('click',()=>{activeFilter=button.dataset.filter;for(const b of document.querySelectorAll('[data-filter]')){b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));}filterAlgorithms();});
$('clear-filters').addEventListener('click',()=>{$('query').value='';document.querySelector('[data-filter="All"]').click();$('query').focus();});
$('print').addEventListener('click',()=>window.print());
let printDetails=[];
window.addEventListener('beforeprint',()=>{printDetails=[...document.querySelectorAll('.algo details')].map(d=>d.open);cards.forEach(c=>c.hidden=false);$('no-results').hidden=true;document.querySelectorAll('.algo details').forEach(d=>d.open=true);});
window.addEventListener('afterprint',()=>{document.querySelectorAll('.algo details').forEach((d,i)=>d.open=printDetails[i]);filterAlgorithms();});
