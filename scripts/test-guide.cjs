const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict'),cp=require('node:child_process');
const root=path.resolve(__dirname,'..'),context=vm.createContext({});
const models=fs.readFileSync(path.join(root,'docs/assets/atlas-models.js'),'utf8');
const examples=fs.readFileSync(path.join(root,'docs/assets/atlas-enhancements.js'),'utf8').split('const note=')[0];
vm.runInContext(models+examples+';globalThis.guide={searchTrace,graphExamples};',context);
const fixtures=[];
for(const [name,example]of Object.entries(context.guide.graphExamples))for(const strategy of ['bfs','dfs','ucs','greedy','astar']){
 const result=context.guide.searchTrace(example.graph,example.h,strategy);
 fixtures.push({name,strategy,graph:example.graph,heuristic:example.h,path:result.path,cost:result.cost});
 if(['ucs','astar'].includes(strategy))assert.equal(result.cost,name==='unreachable'?null:name==='inconsistent'?4:3);
 if(name==='unreachable')assert.equal(result.path,null);
}
const reopen=context.guide.searchTrace(context.guide.graphExamples.inconsistent.graph,context.guide.graphExamples.inconsistent.h,'astar');
assert(reopen.snapshots.filter(s=>s.current==='A').length===2,'Inconsistent heuristic must reopen A');
const complexity=fs.readFileSync(path.join(root,'docs/assets/complexity.js'),'utf8');vm.runInContext(complexity+';globalThis.count=complexityCounts;',context);
assert.equal(context.count(8).linear,8);assert.equal(context.count(16).pairs,256);assert.equal(context.count(8).halving,4);assert.equal(context.count(20).subsets,1048576);assert.throws(()=>context.count(0));
for(const name of ['atlas-models.js','atlas-app.js','atlas-enhancements.js','complexity.js'])new vm.Script(fs.readFileSync(path.join(root,'docs/assets',name),'utf8'));
const out=path.join(root,'.cache');fs.mkdirSync(out,{recursive:true});const file=path.join(out,'search-fixtures.json');fs.writeFileSync(file,JSON.stringify(fixtures));
const result=cp.spawnSync(process.env.PYTHON||'python',[path.join(__dirname,'verify-guide-parity.py'),file],{encoding:'utf8'});if(result.status!==0)throw Error(result.stdout+result.stderr);console.log(result.stdout.trim());
console.log('Graph reopening, no-route results, five presets, and Big O counted examples passed.');
