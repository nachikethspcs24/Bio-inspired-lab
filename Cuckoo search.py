import numpy as np

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
    # Pick two random points to slice the route
    i, j = sorted(np.random.choice(range(1, len(route)), 2, replace=False))
    # Reverse the segment between i and j to untangle the route
    new_route[i:j] = new_route[i:j][::-1]
    return new_route

# Discrete Cuckoo Search Engine
def cuckoo_search_tsp(num_nests=20, max_iter=150, pa=0.25):
    # Initialize Population: Generate random valid routes
    nests = []
    for _ in range(num_nests):
        # A route must contain all cities exactly once
        route = list(np.random.permutation(num_cities))
        nests.append(route)
        
    fitness = np.array([calculate_fitness(n) for n in nests])
    
    # Track Best Solution
    best_idx = np.argmin(fitness)
    best_route = nests[best_idx].copy()
    best_score = fitness[best_idx]
    
    # Generations Loop
    for iteration in range(max_iter):
        
        # Cuckoo Update Step (Exploration via 2-Opt Mutation)
        for i in range(num_nests):
            new_route = two_opt_swap(nests[i])
            new_fitness = calculate_fitness(new_route)
            
            # Selection: Greedy replacement
            if new_fitness < fitness[i]:
                nests[i] = new_route
                fitness[i] = new_fitness
                
        # Abandon Worst Nests Step (Discovery Probability 'pa')
        for i in range(num_nests):
            if np.random.rand() < pa:
                # Replace with a brand new random route permutation
                nests[i] = list(np.random.permutation(num_cities))
                fitness[i] = calculate_fitness(nests[i])
                
        # Update the Absolute Best Solution Found
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_score:
            best_score = fitness[current_best_idx]
            best_route = nests[current_best_idx].copy()
            
        if (iteration + 1) % 30 == 0 or iteration == 0:
            print(f"Generation {iteration+1}/{max_iter} -> Shortest Route Distance: {best_score:.2f} km")
            
    return best_route, best_score

# --- Execution ---
print("Starting Discrete Cuckoo Search Optimization for 15-City TSP...")
best_path, optimal_distance = cuckoo_search_tsp(num_nests=25, max_iter=150, pa=0.25)

print("\n" + "="*45)
print("OPTIMIZATION RESULTS")
print(f"Best Route Sequence Found:\n{best_path} -> {best_path[0]}")
print(f"Total Shortest Distance: {optimal_distance:.4f} km")
print("="*45)
