def dfs(graph, start, goal):
    """
    Finds a path from the start node to the goal node in a graph using Depth-First Search (DFS).

    Args:
        graph (dict): The graph represented as a dictionary where each node is a key, and its corresponding value is a list of its neighboring nodes.
        start: The starting node.
        goal: The goal node.

    Returns:
        list or None: A path from the start node to the goal node as a list of nodes. If there is no path, None is returned.
    """

    visited = set()

    def dfs_recursive(current_node, path):
        if current_node == goal:
            return path

        visited.add(current_node)

        neighbors = graph.get(current_node, [])

        for neighbor in neighbors:
            if neighbor not in visited:
                new_path = list(path)
                new_path.append(neighbor)
                result = dfs_recursive(neighbor, new_path)
                if result is not None:
                    return result

        path.pop()  # Backtrack by removing the current node from the path

        return None

    return dfs_recursive(start, [start])

def main():
    # Example usage
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }

    start_node = 'A'
    goal_node = 'F'

    path = dfs(graph, start_node, goal_node)
    print("Depth-First Search Path:", path)


if __name__ == "__main__":
    main()
