import numpy as np
from multiprocessing import Pool

job_times = np.array((
    (4, 3, 5),
    (2, 6, 3),
    (5, 2, 4),
    (3, 4, 2),
    (6, 3, 4),
    (2, 5, 3)
))

num_jobs, num_stations = job_times.shape

def calculate_makespan(priority_vector):
    order = np.argsort(priority_vector)
    completion_times = np.zeros((num_jobs, num_stations))
   
    for idx, job in enumerate(order):
        for station in range(num_stations):
            prev_job_time = completion_times[idx - 1, station] if idx > 0 else 0
            prev_station_time = completion_times[idx, station - 1] if station > 0 else 0
            completion_times[idx, station] = max(prev_job_time, prev_station_time) + job_times[job, station]
           
    return completion_times[-1, -1]

def update_cell(args):
    r, c, grid, fitness_grid, rows, cols, sigma = args
    neighbors = (
        (r, c),
        ((r - 1) % rows, c),
        ((r + 1) % rows, c),
        (r, (c - 1) % cols),
        (r, (c + 1) % cols)
    )
   
    best_r, best_c = min(neighbors, key=lambda pos: fitness_grid[pos[0], pos[1]])
    best_neighbor = grid[best_r, best_c]
    current_cell = grid[r, c]
   
    trial = current_cell + 0.5 * np.random.rand(num_jobs) * (best_neighbor - current_cell) + sigma * np.random.randn(num_jobs)
    trial_fitness = calculate_makespan(trial)
   
    if trial_fitness < fitness_grid[r, c]:
        return trial, trial_fitness
    return current_cell, fitness_grid[r, c]

def parallel_cellular_production(rows=4, cols=4, max_iter=30, seed=42):
    np.random.seed(seed)
    grid = np.random.uniform(-1, 1, (rows, cols, num_jobs))
    fitness_grid = np.zeros((rows, cols))
   
    for r in range(rows):
        for c in range(cols):
            fitness_grid[r, c] = calculate_makespan(grid[r, c])
           
    initial_best = np.min(fitness_grid)
   
    with Pool() as pool:
        for t in range(max_iter):
            sigma = 0.5 * (1 - t / max_iter)
            tasks = [(r, c, grid, fitness_grid, rows, cols, sigma) for r in range(rows) for c in range(cols)]
            results = pool.map(update_cell, tasks)
           
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    grid[r, c], fitness_grid[r, c] = results[idx]
                    idx += 1
                   
            if (t + 1) % 10 == 0 or t == 0:
                print(f"Iteration {t+1:02d} | Best Cycle Time: {np.min(fitness_grid):.1f} hrs")
               
    best_idx = np.unravel_index(np.argmin(fitness_grid), fitness_grid.shape)
    best_order = tuple(f"Job_{i+1}" for i in np.argsort(grid[best_idx]))
    return best_order, np.min(fitness_grid), initial_best

if __name__ == "__main__":
    sequential_time = calculate_makespan(np.arange(num_jobs))
    print(f"Sequential Schedule Cycle Time: {sequential_time:.1f} hrs\n")
   
    best_schedule, best_time, _ = parallel_cellular_production()
   
    print("\n--- Final Optimized Factory Schedule ---")
    print(f"Optimal Job Order:  {best_schedule}")
    print(f"Optimized Time:     {best_time:.1f} hrs")
    print(f"Time Saved:         {sequential_time - best_time:.1f} hrs")