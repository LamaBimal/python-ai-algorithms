"""Greedy best-first search prioritizes estimated distance to the goal."""

from heapq import heappop, heappush
from itertools import count


def greedy_best_first_search(graph, start, goal, heuristic):
    """Return a path or None, using heuristic(node, goal) as priority.

    graph maps each node to an iterable of neighbors. This search terminates
    on finite graphs, including cyclic ones, but does not guarantee a shortest
    path. Unlike A*, the priority does not include the cost already traveled.
    """
    sequence = count()
    frontier = [(heuristic(start, goal), next(sequence), start, [start])]
    discovered = {start}
    while frontier:
        _, _, node, path = heappop(frontier)
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in discovered:
                discovered.add(neighbor)
                heappush(frontier, (
                    heuristic(neighbor, goal), next(sequence), neighbor, path + [neighbor]
                ))
    return None


def main():
    graph = {"A": ["B", "C"], "B": ["D"], "C": ["G"], "D": ["G"], "G": []}
    estimates = {"A": 3, "B": 1, "C": 2, "D": 1, "G": 0}
    path = greedy_best_first_search(graph, "A", "G", lambda node, goal: estimates[node])
    print("Greedy path (may be longer than the shortest route):", path)


if __name__ == "__main__":
    main()
