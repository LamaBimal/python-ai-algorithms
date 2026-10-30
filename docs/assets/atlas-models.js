function searchTrace(graph, heuristic, strategy, start='S', goal='G'){
 let sequence=0;
 function candidate(node,path,cost){const i=sequence++;const priority={bfs:path.length-1,dfs:-i,ucs:cost,greedy:heuristic[node],astar:cost+heuristic[node]}[strategy];return {node,path,cost,index:i,priority};}
 let queue=[candidate(start,[start],0)],best={[start]:0},discovered=new Set([start]),visited=new Set(),expanded=[],snapshots=[];
 while(queue.length){
  queue.sort((a,b)=>a.priority-b.priority||a.index-b.index);
  const {node,path,cost}=queue.shift();
  if(['ucs','astar'].includes(strategy)&&cost!==best[node])continue;
  if(strategy==='dfs'){if(visited.has(node))continue;visited.add(node);}
  expanded.push(node);
  if(node!==goal){
   let neighbors=Object.entries(graph[node]||{});if(strategy==='dfs')neighbors.reverse();
   for(const[neighbor,weight]of neighbors){
    const nextCost=cost+weight;
    if(['ucs','astar'].includes(strategy)){if(nextCost>=(best[neighbor]??Infinity))continue;best[neighbor]=nextCost;}
    else if(strategy==='dfs'){if(visited.has(neighbor))continue;}
    else{if(discovered.has(neighbor))continue;discovered.add(neighbor);}
    queue.push(candidate(neighbor,[...path,neighbor],nextCost));
   }
  }
  const frontier=[...queue].sort((a,b)=>a.priority-b.priority||a.index-b.index).map(a=>a.node);
  snapshots.push({current:node,path:[...path],cost,expanded:[...expanded],frontier, candidates:[...queue].sort((a,b)=>a.priority-b.priority||a.index-b.index).map(c=>({...c,heuristic:heuristic[c.node],stale:['ucs','astar'].includes(strategy)&&c.cost!==best[c.node]}))});
  if(node===goal)return {path,cost,snapshots};
 }
 return {path:null,cost:null,snapshots};
}
