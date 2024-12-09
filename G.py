import random
import networkx as nx

def generate_and_save_graph(filename, num_vertices, min_degree, max_degree):
    graph = generate_graph(num_vertices, min_degree, max_degree)
    nx.write_gpickle(graph, filename)
    print(f"Graph saved to {filename}")

def generate_graph(num_vertices, min_degree, max_degree):
    while True:
        G = nx.Graph()
        for i in range(num_vertices):
            G.add_node(i)
        for node in range(num_vertices):
            current_degree = G.degree[node]
            if current_degree < min_degree:
                potential_edges = [
                    n for n in range(num_vertices)
                    if n != node and G.degree[n] < max_degree and not G.has_edge(node, n)
                ]
                if potential_edges:
                    edges_to_add = random.sample(potential_edges, min(len(potential_edges), min_degree - current_degree))
                    for neighbor in edges_to_add:
                        G.add_edge(node, neighbor)
        if nx.is_connected(G):
            return G

if __name__ == "__main__":
    filename = "predefined_graph.gpickle"
    num_vertices = 300
    min_degree = 2
    max_degree = 30
    generate_and_save_graph(filename, num_vertices, min_degree, max_degree)