import networkx as nx
from random import choice
from helper.graph import compute_path_length,  compute_flow_value, get_max_neighbors

# reads data from a .brite file and generates graphs


def create_graph(numNodes=100, numEdges=200, fileName="Waxman.brite") -> nx.Graph:
    f = open(fileName)
    for i in range(1, 5):
        f.readline()
    g = nx.Graph()
    for i in range(numNodes):
        q = f.readline().strip()
        line = q.split("\t")
        g.add_node(i, pos=(int(line[1]), int(line[2])))
    for i in range(3):
        f.readline()
    for i in range(numEdges):
        q = f.readline().strip()
        line = q.split("\t")
        # 3 and 5
        g.add_edge(int(line[1]), int(line[2]),
                   weight=float(line[4]), capacity=float(line[5])/100)
    return g


# Create a random route in the network
def get_new_route(graph: nx.Graph) -> (int, int):
    nodes = list(graph.nodes)
    done = False
    node1 = -1
    node2 = -1
    while not done:
        try:
            node1 = choice(nodes)
            node2 = choice(nodes)
            while node1 == node2:
                node2 = choice(nodes)
            nx.shortest_path(graph, node1, node2)
            done = True
        except:
            done = False

    return node1, node2

# get new route try catch safe. To be used when nodes are removed from network


def _get_new_route(graph: nx.Graph) -> (int, int):
    nodes = list(graph.nodes)
    done = False
    node1 = -1
    node2 = -1
    while not done:
        try:
            node1 = choice(nodes)
            node2 = choice(nodes)
            while node1 == node2:
                node2 = choice(nodes)
            nx.shortest_path(graph, node1, node2)
            done = True
        except:
            done = False

    return node1, node2


def get_flows(graph: nx.Graph, num_flows: int) -> (int, int):
    paths = []
    for i in range(num_flows):
        s, t = _get_new_route(graph)
        paths.append(nx.shortest_path(graph, s, t))
    return paths

# adjust the latency and bandwidth to simulate traffic in the network


def adjust_lat_band(graph: nx.Graph, paths: list):
    edges = []
    for path in paths:
        for i in range(len(path)-1):
            if ((path[i], path[i+1])) not in edges and ((path[i+1], path[i])) not in edges:
                edges.append((path[i], path[i+1]))
            # increase latency
            graph[path[i]][path[i+1]]["weight"] = graph[path[i]
                                                        ][path[i+1]]["weight"] ** .1
            if graph[path[i]][path[i+1]]["weight"] > 0.999:
                graph[path[i]][path[i+1]]["weight"] = 0.999
            # decrease bandwidth
            graph[path[i]][path[i+1]]["capacity"] = (graph[path[i]
                                                           ][path[i+1]]["capacity"])**1.2
            if graph[path[i]][path[i+1]]["capacity"] < 0.001:
                graph[path[i]][path[i+1]]["capacity"] = 0.001
    print(len(edges))
    return graph


# @lru_cache(maxsize=None)
def cached_method(graph, source, target):
    return nx.astar_path_length(graph, source, target, weight="weight")


def compute_reward(graph: nx.Graph, target: int, path: list) -> (list, bool, float):
    c2 = cached_method(graph, path[-2], target)
    if path[-1] == target:
        # best_path_length = nx.astar_path_length(graph, path[0], target)
        actual_path_length = compute_path_length(graph, tuple(path))
        # latency_reward = best_path_length / actual_path_length
        # best_flow_value = compute_best_flow(graph, path[0], target)
        actual_flow_value = compute_flow_value(graph, tuple(path))
        # flow_reward = actual_flow_value / best_flow_value
        if c2 == compute_path_length(graph, path[-2:]):
            return [1.01, actual_path_length, actual_flow_value], True
        return [-1.51, actual_path_length, actual_flow_value], True
    if len(path) > 3 * len(list(graph.nodes)):

        c1 = cached_method(graph, path[-1], target)
        if c1 < c2:
            return [(c2 - c1), 0, 0], True
        return [-1, 0, 0], True
    else:
        c1 = cached_method(graph, path[-1], target)
        if c1 < c2:
            return [(c2 - c1)], False
        return [-1], False
