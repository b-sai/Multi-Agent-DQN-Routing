import networkx as nx
import matplotlib.pyplot as plt
from random import choice, random
from copy import deepcopy
from functools import lru_cache

def set_max_neighbors(graph: nx.Graph, max_neighbors: int) -> nx.Graph:
    if max_neighbors is None:
        return graph
    node = get_node_with_max_neighbors(graph)
    while len(list(graph.neighbors(node))) < max_neighbors:
        for n2 in graph.nodes:
            if n2 not in graph.neighbors(node):
                graph.add_edge(node, n2, weight=(random() * 0.9) + 0.1, capacity=(random() * 0.9) + 0.1)
    return graph


# Randomly set the weights to be between [0.1,1.0]
def randomize_weights(graph: nx.Graph) -> nx.Graph:
    # Create a deep copy of the input graph
    random_edges = nx.Graph()
    for edge in graph.edges:
        random_edges.add_edge(edge[0], edge[1], weight=(random() * 0.9) + 0.1, capacity=(random() * 0.9) + 0.1)
    return random_edges


def get_neighbors(graph: nx.Graph, node: int) -> list:
    return list(graph.neighbors(node))


# Draws a graph
def draw_graph(g: nx.Graph) -> None:
    pos = nx.spring_layout(g)  # positions for all nodes

    nx.draw_networkx_nodes(g, pos, node_size=700)

    nx.draw_networkx_edges(g, pos, edgelist=g.edges,
                           width=6, alpha=0.5, edge_color='b', style='dashed')

    # labels
    nx.draw_networkx_labels(g, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    plt.show()


# Returns the node with the max neighbors. If there is more than one, randomly select one
def get_node_with_max_neighbors(graph: nx.Graph) -> int:
    max_neighbors = 0
    max_neighbor_nodes = []
    for node in graph.nodes:
        node_neighbors = list(graph.neighbors(node))
        if len(node_neighbors) > max_neighbors:
            max_neighbors = len(node_neighbors)
            max_neighbor_nodes = [node]
        elif len(node_neighbors) == max_neighbors:
            max_neighbor_nodes.append(node)
    return choice(max_neighbor_nodes)


# Return the max number of neighbors in the graph
def get_max_neighbors(graph: nx.Graph) -> int:
    max_neighbors = 0
    for node in graph.nodes:
        node_neighbors = list(graph.neighbors(node))
        if len(node_neighbors) > max_neighbors:
            max_neighbors = len(node_neighbors)
    return max_neighbors


# Computes the length path given the weights of the edges it goes along from graph
# @lru_cache(maxsize=None)
def compute_path_length(graph: nx.Graph, path: list) -> float:
    path_length = 0
    for i in range(len(path)):
        if path[i] != path[-1]:
            path_length += graph[path[i]][path[i+1]]["weight"]
    return path_length


# Iteratively remove the worst capacity link until no more paths exist between source and target
# Return the worst capacity along that path
def compute_best_flow(graph: nx.Graph, source: int, target: int) -> float:
    flow_graph = deepcopy(graph)
    worst_capacity = 1
    reachable = True
    #path = None
    while reachable:
        try:
            nx.astar_path_length(flow_graph, source, target)
            #path = nx.astar_path(flow_graph, source, target)
            min_edge = min(flow_graph.edges, key=lambda edge: flow_graph[edge[0]][edge[1]]["capacity"])
            worst_capacity = flow_graph[min_edge[0]][min_edge[1]]["capacity"]
            flow_graph.remove_edge(min_edge[0], min_edge[1])
        except nx.exception.NetworkXNoPath:
            reachable = False
    return worst_capacity


# @lru_cache(maxsize=None)
def compute_flow_value(graph: nx.Graph, path: list) -> float:
    worst_capacity = 1
    for i in range(len(path)):
        if path[i] != path[-1]:
            edge_capacity = graph[path[i]][path[i+1]]["capacity"]
            if worst_capacity > edge_capacity:
                worst_capacity = edge_capacity
    return worst_capacity


def adj_mat(graph: nx.Graph) -> list:
    return nx.adjacency_matrix(graph).todense()
