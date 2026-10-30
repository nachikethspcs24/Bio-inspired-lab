
import random

# Cloud tasks (in Million Instructions)
tasks = [4000, 6000, 3000, 8000, 5000, 2000, 7000, 4000]

# Virtual machines (MIPS)
vms = [1000, 2000, 1500]

# Power consumption of VMs (Watts)
power = [100, 150, 120]

N = len(tasks)
M = len(vms)

# Calculate scheduling performance
def calculate(assignments):
    finish = [0] * M
    total_energy = 0
    waiting = 0
    busy = [0] * M
    completion = []

    for i in range(N):
        vm = assignments[i]
        execution = tasks[i] / vms[vm]

        waiting += finish[vm]
        finish[vm] += execution
        busy[vm] += execution

        energy = power[vm] * execution / 3600
        total_energy += energy
        completion.append(execution)

    makespan = max(finish)
    utilization = sum(busy) / (M * makespan) * 100
    avg_waiting = waiting / N
    avg_completion = sum(completion) / N

    return (makespan, utilization,
            total_energy, avg_waiting, avg_completion)


# FCFS Scheduling
def fcfs():
    finish = [0] * M
    assignments = []

    for task in tasks:
        vm = finish.index(min(finish))
        assignments.append(vm)
        finish[vm] += task / vms[vm]

    return assignments


# Round Robin Scheduling
def round_robin():
    return [i % M for i in range(N)]


# PSO Fitness Function
def fitness(position):
    assignments = [
        max(0, min(M - 1, round(x)))
        for x in position
    ]

    makespan, utilization, energy, waiting, completion = calculate(assignments)

    # Lower fitness is better
    score = (
        makespan / 100
        + waiting / 100
        + energy / 10
        + (100 - utilization) / 100
    )

    return score


# Particle Swarm Optimization
def pso():
    particles = 20
    iterations = 100

    positions = [
        [random.uniform(0, M - 1) for _ in range(N)]
        for _ in range(particles)
    ]

    velocities = [
        [0] * N for _ in range(particles)
    ]

    pbest = [p[:] for p in positions]
    pbest_score = [fitness(p) for p in positions]

    best_index = pbest_score.index(min(pbest_score))
    gbest = pbest[best_index][:]

    w = 0.7
    c1 = 1.5
    c2 = 1.5

    for _ in range(iterations):
        for i in range(particles):

            for j in range(N):
                r1 = random.random()
                r2 = random.random()

                velocities[i][j] = (
                    w * velocities[i][j]
                    + c1 * r1 * (pbest[i][j] - positions[i][j])
                    + c2 * r2 * (gbest[j] - positions[i][j])
                )

                positions[i][j] += velocities[i][j]

                positions[i][j] = max(
                    0, min(M - 1, positions[i][j])
                )

            score = fitness(positions[i])

            if score < pbest_score[i]:
                pbest[i] = positions[i][:]
                pbest_score[i] = score

                if score < fitness(gbest):
                    gbest = positions[i][:]

    return [
        max(0, min(M - 1, round(x)))
        for x in gbest
    ]


# Compare all algorithms
algorithms = {
    "FCFS": fcfs(),
    "Round Robin": round_robin(),
    "PSO": pso()
}

print("\nCLOUD TASK SCHEDULING RESULTS")
print("-" * 85)

print(f"{'Algorithm':<15}{'Completion':<15}"
      f"{'Utilization':<15}{'Energy':<12}"
      f"{'Makespan':<12}{'Waiting'}")

for name, assignment in algorithms.items():

    makespan, utilization, energy, waiting, completion = calculate(assignment)

    print(f"{name:<15}{completion:<15.2f}"
          f"{utilization:<15.2f}{energy:<12.2f}"
          f"{makespan:<12.2f}{waiting:.2f}")
