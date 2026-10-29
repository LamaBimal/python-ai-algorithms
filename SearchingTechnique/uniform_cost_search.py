"""Uniform-cost search (Dijkstra) for graphs with nonnegative edge costs."""

from heapq import heappop, heappush
from itertools import count
from math import isfinite


def uniform_cost_search(graph, start, goal):
    """Return (shortest_path, cost), or None when no route exists.

    graph maps each node to {neighbor: cost}. Nodes must be hashable; a node
    without an entry is treated as a leaf. Costs must be finite and nonnegative.
    Unlike BFS, this minimizes total edge cost rather than number of edges.
    """
    for neighbors in graph.values():
        for cost in neighbors.values():
            if not isfinite(cost) or cost < 0:
                raise ValueError("Edge costs must be finite and nonnegative")

    sequence = count()
    frontier = [(0, next(sequence), start, [start])]
    best_cost = {start: 0}
    while frontier:
        cost, _, node, path = heappop(frontier)
        if cost != best_cost[node]:
            continue
        if node == goal:
            return path, cost
        for neighbor, edge_cost in graph.get(node, {}).items():
            new_cost = cost + edge_cost
            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                heappush(frontier, (new_cost, next(sequence), neighbor, path + [neighbor]))
    return None


def main():
    graph = {"A": {"B": 8, "C": 1}, "C": {"B": 2}, "B": {"D": 1}, "D": {}}
    print("Uniform-cost shortest path and cost:", uniform_cost_search(graph, "A", "D"))


if __name__ == "__main__":
    main()
