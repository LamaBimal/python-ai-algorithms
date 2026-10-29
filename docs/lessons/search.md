# Graph search and heuristics

[Learning path](../learning-path.md) | [Project README](../../README.md)

## Learning objectives and intuition

Represent a problem with states, allowed transitions, a start, and a goal.
Use BFS for minimum edge count, DFS to explore deeply, UCS for minimum cost,
and greedy search to follow a heuristic. A* combines cost so far and an
estimate of remaining cost.

## Mathematics and assumptions

BFS orders by depth; UCS orders by g(n), the cost already paid. Greedy
search orders by h(n); A* orders by f(n) = g(n) + h(n). An admissible h never
overestimates the remaining optimal cost and is zero at the goal. UCS/A*
require nonnegative edge costs here. A* must reopen improved states when an
admissible heuristic is inconsistent.

For adjacency-list graphs, conventional BFS/DFS can run in O(V + E), with
O(V) bookkeeping. These teaching implementations retain whole candidate
paths and visualization snapshots, so their memory usage can be larger.
Heuristic search performance depends on the graph and heuristic.

## Worked example

Use edges S->A:1, S->B:4, A->C:1, C->G:1, B->G:1.
BFS finds S-B-G (two edges, cost 5); UCS finds S-A-C-G (three edges,
cost 3). With h(A)=2, h(C)=1, h(B)=0, h(G)=0, greedy search can prefer
the expensive B route. A* still accounts for cost already paid.

## Implementation

Run `Learning/VisualFoundations.ipynb`. The instrumented search in
`Learning/search_walkthrough.py` records the expanded state, frontier,
current path, and accumulated cost for all five strategies. The existing
search modules remain available as focused implementations.

## Evaluation

Check every returned edge, total cost, start-equals-goal behavior, cycles,
and unreachable goals. Compare costs with UCS when assessing an admissible
A* heuristic. Frontier snapshots may contain stale candidates awaiting removal.

## Limitations and failure cases

DFS and greedy search do not guarantee the cheapest route. BFS ignores
unequal costs. An overestimating heuristic removes the A* optimality guarantee.
A heuristic is problem knowledge, not automatically an accurate distance.

## Exercises and checkpoint

1. Predict the expansion order before running each strategy.
2. Change S->B to cost 0.5 and explain the changed cheapest path.
3. Set h to zero and compare A* with UCS.

Checkpoint: explain why fewer edges can still mean a more expensive route.
