"""Compare browser fixtures with the actual Python functions and notebook A*."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'SearchingTechnique' / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

bfs = load('bfs', 'breadth_first_search.py').bfs_shortest_path
dfs = load('dfs', 'deapth_first_search.py').dfs
ucs = load('ucs', 'uniform_cost_search.py').uniform_cost_search
greedy = load('greedy', 'greedy_best_first_search.py').greedy_best_first_search
notebook = json.loads((ROOT / 'SearchingTechnique/Case_Study(BFS_&_A_).ipynb').read_text(encoding='utf-8'))
namespace = {}
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    tree = ast.parse(''.join(cell['source']))
    if any(isinstance(node, ast.FunctionDef) and node.name == 'a_star_shortest_path' for node in tree.body):
        # Execute only imports and the target function, never notebook demos/plots.
        selected = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) or isinstance(node, ast.FunctionDef) and node.name == 'a_star_shortest_path']
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<notebook-a-star>', 'exec'), namespace)
        break

fixtures = json.loads(Path(sys.argv[1]).read_text())
for case in fixtures:
    graph = case['graph']
    adjacency = {node: list(neighbors) for node, neighbors in graph.items()}
    strategy = case['strategy']
    heuristic = lambda node, goal: case['heuristic'][node]
    if strategy == 'bfs':
        path = bfs(adjacency, 'S', 'G')
    elif strategy == 'dfs':
        path = dfs(adjacency, 'S', 'G')
    elif strategy == 'greedy':
        path = greedy(adjacency, 'S', 'G', heuristic)
    elif strategy == 'ucs':
        result = ucs(graph, 'S', 'G')
        path = None if result is None else result[0]
    else:
        path = namespace['a_star_shortest_path'](graph, 'S', 'G', heuristic)
    cost = None if path is None else sum(graph[a][b] for a, b in zip(path, path[1:]))
    assert path == case['path'] and cost == case['cost'], (case['name'], strategy, path, case)
print(f'Python/browser parity: {len(fixtures)} fixtures across all five search strategies passed.')
