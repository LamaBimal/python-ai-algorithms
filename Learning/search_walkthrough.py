"""Instrumented search for displaying frontier and expansion decisions."""

from heapq import heappop, heappush
from itertools import count
from math import isfinite


def trace_search(graph, start, goal, strategy="bfs", heuristic=None):
    """Search a weighted dictionary graph and return path, cost, and snapshots.

    Supported strategies: bfs, dfs, ucs, greedy, astar. BFS minimizes edges;
    DFS follows input neighbor order. UCS minimizes nonnegative cost. A* needs
    an admissible heuristic, zero at the goal, for an optimality guarantee.
    Snapshot frontier entries are queued candidates and can include stale
    entries awaiting removal. The graph is finite and missing nodes are leaves.
    """
    if strategy not in {"bfs", "dfs", "ucs", "greedy", "astar"}:
        raise ValueError("Unknown strategy")
    for neighbors in graph.values():
        for cost in neighbors.values():
            if not isfinite(cost) or cost < 0:
                raise ValueError("Edge costs must be finite and nonnegative")
    heuristic = heuristic or (lambda node, goal: 0)
    sequence = count()

    def entry(node, path, cost):
        index = next(sequence)
        estimate = heuristic(node, goal) if strategy in {"greedy", "astar"} else 0
        if not isfinite(estimate):
            raise ValueError("Heuristic must be finite")
        priority = {"bfs": len(path), "dfs": -index, "ucs": cost,
                    "greedy": estimate, "astar": cost + estimate}[strategy]
        return priority, index, node, path, cost

    frontier = [entry(start, [start], 0)]
    best = {start: 0}
    discovered = {start}
    visited = set()
    expanded = []
    snapshots = []
    while frontier:
        _, _, node, path, cost = heappop(frontier)
        if strategy in {"ucs", "astar"} and cost != best[node]:
            continue
        if strategy == "dfs":
            if node in visited:
                continue
            visited.add(node)
        expanded.append(node)
        if node != goal:
            neighbors = list(graph.get(node, {}).items())
            if strategy == "dfs":
                neighbors.reverse()
            for neighbor, weight in neighbors:
                new_cost = cost + weight
                if strategy in {"ucs", "astar"}:
                    if new_cost >= best.get(neighbor, float("inf")):
                        continue
                    best[neighbor] = new_cost
                elif strategy == "dfs":
                    if neighbor in visited:
                        continue
                else:
                    if neighbor in discovered:
                        continue
                    discovered.add(neighbor)
                heappush(frontier, entry(neighbor, path + [neighbor], new_cost))
        snapshots.append({"current": node, "expanded": expanded.copy(),
                          "frontier": [item[2] for item in sorted(frontier)],
                          "path": path.copy(), "cost": cost})
        if node == goal:
            return {"path": path, "cost": cost, "snapshots": snapshots}
    return {"path": None, "cost": None, "snapshots": snapshots}
