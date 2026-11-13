import numpy as np

# 1. Fault Detection Matrix (5 Test Cases x 4 Faults)
# Rows = Test Cases (T1 to T5), Columns = Faults (F1 to F4)
# 1 indicates the test case detects that fault, 0 otherwise
fault_matrix = np.zeros((5, 4), dtype=int)
fault_matrix[0, 0] = 1        # T1 detects F1
fault_matrix[1, 1] = 1        # T2 detects F2
fault_matrix[2, 0:3] = 1      # T3 detects F1, F2, F3
fault_matrix[3, 2:4] = 1      # T4 detects F3, F4
fault_matrix[4, 3] = 1        # T5 detects F4

test_names = ["T1", "T2", "T3", "T4", "T5"]
num_tests, num_faults = fault_matrix.shape

# 2. Fitness Function: Average Percentage of Faults Detected (APFD)
def calculate_apfd(order):
    tf_sum = 0
    for fault_idx in range(num_faults):
        for position, test_idx in enumerate(order):
            if fault_matrix[test_idx, fault_idx] == 1:
                tf_sum += (position + 1)  # 1-based index of first detection
                break
    apfd = 1 - (tf_sum / (num_tests * num_faults)) + (1 / (2 * num_tests))
    return apfd

# 3. Rank-Order Value (ROV) Mapping: Continuous Vector -> Discrete Test Order
def decode_wolf(position_vector):
    return np.argsort(position_vector)

# 4. Grey Wolf Optimization (GWO) Algorithm
def gwo_test_prioritization(num_wolves=10, max_iter=20, seed=42):
    np.random.seed(seed)
   
    # Initialize wolf pack positions randomly in [-1, 1]
    wolves = np.random.uniform(-1, 1, (num_wolves, num_tests))
   
    # Initialize Alpha, Beta, and Delta leaders
    alpha_pos = np.zeros(num_tests)
    alpha_score = -float("inf")
   
    beta_pos = np.zeros(num_tests)
    beta_score = -float("inf")
   
    delta_pos = np.zeros(num_tests)
    delta_score = -float("inf")
   
    # Main Hunting Loop
    for t in range(max_iter):
        for i in range(num_wolves):
            # Convert continuous wolf vector to discrete test sequence
            test_order = decode_wolf(wolves[i])
            fitness = calculate_apfd(test_order)
           
            # Update Alpha, Beta, and Delta hierarchy
            if fitness > alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, wolves[i].copy()
            elif fitness > beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, wolves[i].copy()
            elif fitness > delta_score:
                delta_score, delta_pos = fitness, wolves[i].copy()
       
        # Linearly decrease parameter 'a' from 2 to 0
        a = 2 - t * (2 / max_iter)
       
        # Update positions of all Omega wolves
        for i in range(num_wolves):
            for j in range(num_tests):
                # Guidance from Alpha
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - wolves[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha
               
                # Guidance from Beta
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - wolves[i, j])
                X2 = beta_pos[j] - A2 * D_beta
               
                # Guidance from Delta
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - wolves[i, j])
                X3 = delta_pos[j] - A3 * D_delta
               
                # Average position update and keep within [-1, 1]
                wolves[i, j] = np.clip((X1 + X2 + X3) / 3.0, -1.0, 1.0)

        if (t + 1) % 5 == 0 or t == 0:
            current_seq = [test_names[idx] for idx in decode_wolf(alpha_pos)]
            print(f"Iteration {t+1:02d} | Best APFD: {alpha_score:.4f} | Order: {current_seq}")

    return decode_wolf(alpha_pos), alpha_score

# Run Comparison
if __name__ == "__main__":
    unprioritized_order = list(range(num_tests))
    initial_apfd = calculate_apfd(unprioritized_order)

    print("--- Unprioritized Execution ---")
    print(f"Order: {test_names} | APFD: {initial_apfd:.4f} ({initial_apfd*100:.2f}%)\n")

    print("--- GWO Optimization Progress ---")
    best_order_idx, best_apfd = gwo_test_prioritization()
    best_order_names = [test_names[idx] for idx in best_order_idx]

    print("\n--- Final Prioritized Result ---")
    print(f"Optimal Alpha Order: {best_order_names}")
    print(f"Optimized APFD:      {best_apfd:.4f} ({best_apfd*100:.2f}%)")
    print(f"Improvement:         +{(best_apfd - initial_apfd)*100:.2f}%")
