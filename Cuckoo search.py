import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Problem (15 Cities with Random X, Y Coordinates)
np.random.seed(42) # For reproducible coordinates
num_cities = 15
city_coordinates = np.random.uniform(0, 100, (num_cities, 2))

# Calculate a distance matrix between all cities
dist_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        dist_matrix[i, j] = np.linalg.norm(city_coordinates[i] - city_coordinates[j])

# Fitness Function: Calculates total distance of a route
def calculate_fitness(route):
    distance = 0
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i], route[i+1]]
    distance += dist_matrix[route[-1], route[0]] # Return to starting city
    return distance

# Mutation Operator: 2-Opt Swap (Mimics Levy flight exploration in discrete spaces)
def two_opt_swap(route):
    new_route = route.copy()
    i, j = sorted(np.random.choice(range(1, len(route)), 2, replace=False))
    new_route[i:j] = new_route[i:j][::-1]
    return new_route

# Discrete Cuckoo Search Engine
def cuckoo_search_tsp(num_nests=20, max_iter=150, pa=0.25):
    nests = []
    for _ in range(num_nests):
        route = list(np.random.permutation(num_cities))
        nests.append(route)
        
    fitness = np.array([calculate_fitness(n) for n in nests])
    
    best_idx = np.argmin(fitness)
    best_route = nests[best_idx].copy()
    best_score = fitness[best_idx]
    
    # Track the optimization progress over generations
    fitness_history = [best_score]
    
    for iteration in range(max_iter):
        # Cuckoo Update Step
        for i in range(num_nests):
            new_route = two_opt_swap(nests[i])
            new_fitness = calculate_fitness(new_route)
            if new_fitness < fitness[i]:
                nests[i] = new_route
                fitness[i] = new_fitness
                
        # Abandon Worst Nests Step
        for i in range(num_nests):
            if np.random.rand() < pa:
                nests[i] = list(np.random.permutation(num_cities))
                fitness[i] = calculate_fitness(nests[i])
                
        # Update Best Solution
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_score:
            best_score = fitness[current_best_idx]
            best_route = nests[current_best_idx].copy()
            
        fitness_history.append(best_score)
            
        if (iteration + 1) % 30 == 0 or iteration == 0:
            print(f"Generation {iteration+1}/{max_iter} -> Shortest Route Distance: {best_score:.2f} km")
            
    return best_route, best_score, fitness_history

# --- Execution ---
print("Starting Discrete Cuckoo Search Optimization for 15-City TSP...")
best_path, optimal_distance, history = cuckoo_search_tsp(num_nests=25, max_iter=150, pa=0.25)

print("\n" + "="*45)
print("OPTIMIZATION RESULTS")
print(f"Best Route Sequence Found:\n{best_path} -> {best_path[0]}")
print(f"Total Shortest Distance: {optimal_distance:.4f} km")
print("="*45)

# --- Plotting Code ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left Subplot: Spatial TSP Path Layout
# Close the path ring loop back to the start city index
plotted_path = best_path + [best_path[0]]
coords = city_coordinates[plotted_path]

ax1.plot(coords[:, 0], coords[:, 1], 'o-', markersize=8, color='crimson', linewidth=2, label='Optimal Route')
ax1.scatter(city_coordinates[:, 0], city_coordinates[:, 1], color='darkblue', zorder=5)

# Label each individual city dot
for idx, (x, y) in enumerate(city_coordinates):
    ax1.text(x + 1.5, y + 1.5, f"C{idx}", fontsize=9, fontdict={'weight': 'bold'})

ax1.set_title("Final Optimized Spatial TSP Route Layout", fontsize=12, pad=10)
ax1.set_xlabel("X Coordinate Space (km)")
ax1.set_ylabel("Y Coordinate Space (km)")
ax1.grid(True, linestyle='--', alpha=0.5)

# Right Subplot: Convergence History Curve
ax2.plot(history, color='teal', linewidth=2.5, label='Best Nest Fitness')
ax2.set_title("Cuckoo Optimization Convergence History", fontsize=12, pad=10)
ax2.set_xlabel("Generation (Iteration Step)")
ax2.set_ylabel("Route Total Distance Score (km)")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()
