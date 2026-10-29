# Search algorithms

| Algorithm | Input | Result | Shortest-path guarantee |
| --- | --- | --- | --- |
| BFS | Node to list of neighbors | Path or `None` | Yes, for equal-cost edges |
| DFS | Node to list of neighbors | Path or `None` | No |
| Uniform-cost search | Node to dictionary of neighbor costs | `(path, cost)` or `None` | Yes, for finite, nonnegative costs |
| Greedy best-first search | Node to list of neighbors, plus a heuristic | Path or `None` | No; ignores cost already traveled |
| A* (case study notebook) | Adjacency lists or neighbor-cost dictionaries, optional heuristic | Path or `None` | Yes, for nonnegative costs and an admissible heuristic that is zero at the goal |

Run the new examples from the repository root; no third-party packages are needed:

```sh
python -m SearchingTechnique.uniform_cost_search
python -m SearchingTechnique.greedy_best_first_search
```

Uniform-cost search chooses the least expensive route even if it has more
edges. Its example returns `A -> C -> B -> D` with total cost 4 instead of the
route through the expensive `A -> B` edge. Negative or nonfinite costs are
rejected because the shortest-path guarantee would not apply.

Greedy best-first search chooses the next node using only the estimated
distance to the goal. Its example deliberately follows `A -> B -> D -> G`
although `A -> C -> G` has fewer edges. This shows why a promising heuristic
alone does not guarantee an optimal route. A discovered set prevents cycles
from causing repeated expansion.

Both new functions accept hashable node labels and use counters to break ties,
so labels with different types can share a frontier. Nodes missing from the
adjacency dictionary are treated as leaves.
