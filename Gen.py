import random
import networkx as nx
import matplotlib.pyplot as plt

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

def is_valid_cover(solution, graph):
    for u, v in graph.edges():
        if u not in solution and v not in solution:
            return False
    return True

def initialize_population(graph, population_size):
    population = []
    nodes = list(graph.nodes)
    while len(population) < population_size:
        individual = random.sample(nodes, random.randint(1, len(nodes)))
        if is_valid_cover(individual, graph):
            population.append(individual)
    return population

def fitness(solution, graph):
    return len(solution) if is_valid_cover(solution, graph) else float('inf')

def proportional_selection(population, graph):
    fitness_values = [1 / fitness(ind, graph) for ind in population]
    total_fitness = sum(fitness_values)
    probabilities = [fv / total_fitness for fv in fitness_values]
    return random.choices(population, weights=probabilities, k=1)[0]

def crossover_single_offspring(parent1, parent2):
    split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child = parent1[:split_point] + [v for v in parent2[split_point:] if v not in parent1[:split_point]]
    return child

def crossover_with_validation(graph, population, max_attempts=10):
    for _ in range(max_attempts):
        parent1 = proportional_selection(population, graph)
        parent2 = proportional_selection(population, graph)
        while parent1 == parent2:
            parent2 = proportional_selection(population, graph)
        child = crossover_single_offspring(parent1, parent2)
        if is_valid_cover(child, graph):
            return child
    return crossover_with_validation(graph, population, max_attempts)

def local_improvement(individual, graph):
    improved = individual[:]
    if random.random() > 0.5:
        node = random.choice(list(graph.nodes))
        if node not in improved:
            improved.append(node)
    else:  
        if len(improved) > 1:
            improved.remove(random.choice(improved))
    return improved

def local_improvement_with_fallback(child, graph):
    improved = local_improvement(child, graph)
    return improved if is_valid_cover(improved, graph) else child

def swap_mutation(individual, swap_chance, min_swaps=2):
    swapped = individual[:]
    num_swaps = 0
    while num_swaps < min_swaps:
        if random.random() < swap_chance:
            if len(swapped) > 1:
                idx1, idx2 = random.sample(range(len(swapped)), 2)
                swapped[idx1], swapped[idx2] = swapped[idx2], swapped[idx1]
                num_swaps += 1
    return swapped


def genetic_algorithm(graph, population_size, num_generations, swap_chance):
    population = initialize_population(graph, population_size)
    best_solution = min(population, key=lambda ind: fitness(ind, graph))
    best_fitness_per_generation = [fitness(best_solution, graph)]

    visualize_graph(graph, best_solution)
    print(f"Generation 0: Best coverage = {fitness(best_solution, graph)}")

    for generation in range(num_generations):
        new_population = []
        child = crossover_with_validation(graph, population)
        swapped_child = swap_mutation(child, swap_chance)
        improved_child = local_improvement_with_fallback(swapped_child, graph)
        new_population.append(improved_child)

        population.extend(new_population)
        population = sorted(population, key=lambda ind: fitness(ind, graph))
        if len(population) > population_size:
            population.pop()

        best_candidate = min(population, key=lambda ind: fitness(ind, graph))
        if fitness(best_candidate, graph) < fitness(best_solution, graph):
            best_solution = best_candidate

        best_fitness_per_generation.append(fitness(best_solution, graph))
        print(f"Generation {generation + 1}: Best coverage = {fitness(best_solution, graph)}")

    plot_fitness(best_fitness_per_generation)
    return best_solution

def plot_fitness(best_fitness_per_generation):
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_per_generation, marker='o', label='Best coverage size')
    plt.xlabel("Generation")
    plt.ylabel("Size of best coverage")
    plt.title("Best coverage size over generations")
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_graph(graph, cover):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10)
    nx.draw_networkx_nodes(graph, pos, nodelist=cover, node_color="red", label="Best coverage")
    plt.legend(scatterpoints=1, loc="upper right", fontsize=12)
    plt.title("Graph and vertex coverage visualization")
    plt.show()

def load_graph(filename):
    graph = nx.read_gpickle(filename)
    print(f"Graph loaded from {filename}")
    return graph

if __name__ == "__main__":
    num_vertices = 300
    min_degree = 2
    max_degree = 30
    population_size = 25
    num_generations = 1500
    improvement_chance = 0.2
    # filename = "predefined_graph.gpickle"
    # graph = load_graph(filename)

    graph = generate_graph(num_vertices, min_degree, max_degree)
    best_solution = genetic_algorithm(graph, population_size, num_generations, improvement_chance)
    print("Best vertex coverage:", best_solution)
    print("Coverage Size:", len(best_solution))
    visualize_graph(graph, best_solution)
