from queue import Queue

def bfs_shortest_path(graph, start, goal):
    """
    Finds the shortest path from the start node to the goal node in a graph using breadth-first search (BFS).

    Args:
        graph (dict): The graph represented as a dictionary where each node is a key, and its corresponding value is a list of its neighboring nodes.
        start: The starting node.
        goal: The goal node.

    Returns:
        list or None: The shortest path from the start node to the goal node as a list of nodes. If there is no path, None is returned.
    """

    # Set to keep track of visited nodes
    visited = set()

    # Queue to store paths
    queue = Queue()
    queue.put([start])

    while not queue.empty():
        path = queue.get()
        current_node = path[-1]
        if current_node == goal:
            return path

        if current_node not in visited:
            visited.add(current_node)
            neighbors = graph.get(current_node, [])
            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                queue.put(new_path)
    return None

def main():
    # Example usage
    graph = {
        'Hattisar': ['Location1', 'Location2'],
        'Location1': ['Location3', 'TribhuvanAirport'],
        'Location2': ['TribhuvanAirport'],
        'Location3': ['TribhuvanAirport'],
        'TribhuvanAirport': []
    }

    start_node = 'Hattisar'
    goal_node = 'TribhuvanAirport'

    shortest_path = bfs_shortest_path(graph, start_node, goal_node)
    print("Uninformed Search (BFS) Shortest Path:", shortest_path)


if __name__ == "__main__":
    main()
