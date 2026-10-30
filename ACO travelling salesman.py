
import random

dist = [
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 20],
    [15, 35, 0, 30, 15],
    [20, 25, 30, 0, 10],
    [25, 20, 15, 10, 0]
]

n = len(dist)
ants = 10
iterations = 100

alpha = 1
beta = 2
rho = 0.5
Q = 100

pheromone = [[1.0] * n for _ in range(n)]

def distance(route):
    total = 0
    for i in range(n):
        total += dist[route[i]][route[(i + 1) % n]]
    return total

best_route = None
best_distance = float('inf')

for _ in range(iterations):

    all_routes = []

    for ant in range(ants):

        start = random.randrange(n)
        route = [start]
        unvisited = list(range(n))
        unvisited.remove(start)

        while unvisited:

            current = route[-1]
            weights = []

            for city in unvisited:
                p = pheromone[current][city] ** alpha
                h = (1 / dist[current][city]) ** beta
                weights.append(p * h)

            next_city = random.choices(
                unvisited, weights=weights
            )[0]

            route.append(next_city)
            unvisited.remove(next_city)

        d = distance(route)
        all_routes.append((route, d))

        if d < best_distance:
            best_distance = d
            best_route = route[:]

    # Pheromone evaporation
    for i in range(n):
        for j in range(n):
            pheromone[i][j] *= (1 - rho)

    # Pheromone update
    for route, d in all_routes:
        deposit = Q / d

        for i in range(n):
            a = route[i]
            b = route[(i + 1) % n]

            pheromone[a][b] += deposit
            pheromone[b][a] += deposit

print("ACO - TSP")
print("Best Route:", [x + 1 for x in best_route])
print("Shortest Distance:", best_distance)
