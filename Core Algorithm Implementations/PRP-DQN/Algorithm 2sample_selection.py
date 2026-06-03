import os
import numpy as np
import random
import time


X_MIN, X_MAX = 1, 50
Y_MIN, Y_MAX = 1, 50
Z_MIN, Z_MAX = 1, 50

STATE_MIN = np.array([X_MIN, Y_MIN, Z_MIN])
STATE_MAX = np.array([X_MAX, Y_MAX, Z_MAX])


def Tr(state):

    dx, dy, dz = state
    MAX_GRID_SIZE = 500.0
    MIN_PLANNING_X, MIN_PLANNING_Y, MIN_PLANNING_Z = 10.0, 15.0, 8.0
    CRITICAL_X_VELOCITY, CRITICAL_Y_VELOCITY, CRITICAL_Z_VELOCITY = 20.0, 25.0, 15.0
    TARGET_X, TARGET_Y, TARGET_Z = 450.0, 450.0, 200.0

    triggered = set()
    current_x = random.uniform(0.0, MAX_GRID_SIZE)
    current_y = random.uniform(0.0, MAX_GRID_SIZE)
    current_z = random.uniform(0.0, MAX_GRID_SIZE)
    simulated_y = current_y

    if abs(dx) < MIN_PLANNING_X != abs(dy) < MIN_PLANNING_X: triggered.add(1)
    if abs(dx) < MIN_PLANNING_X != abs(dz) < MIN_PLANNING_X: triggered.add(2)
    if abs(dx) < MIN_PLANNING_X != abs(dx) < MIN_PLANNING_Y: triggered.add(3)
    if abs(dx) < MIN_PLANNING_X != abs(dx) < MIN_PLANNING_Z: triggered.add(4)

    if abs(dz) > MIN_PLANNING_Z * 2 != abs(dx) > MIN_PLANNING_Z * 2: triggered.add(5)
    if abs(dz) > MIN_PLANNING_Z * 2 != abs(dy) > MIN_PLANNING_Z * 2: triggered.add(6)
    if abs(dz) > MIN_PLANNING_Z * 2 != abs(dz) > MIN_PLANNING_X * 2: triggered.add(7)
    if abs(dz) > MIN_PLANNING_Z * 2 != abs(dz) > MIN_PLANNING_Y * 2: triggered.add(8)
    if abs(dz) > MIN_PLANNING_Z * 2 != abs(dz) > MIN_PLANNING_Z: triggered.add(9)

    if TARGET_Y > simulated_y and dy < 20 != TARGET_Y > simulated_y and dy < 10: triggered.add(10)
    if TARGET_Y > simulated_y and dy < 20 != TARGET_Y > simulated_y and dy < 30: triggered.add(11)
    if TARGET_Y > simulated_y and dy < 20 != TARGET_Y > simulated_y and dy < 40: triggered.add(12)
    if TARGET_Y > simulated_y and dy < 20 != TARGET_Y > simulated_y and dy < 50: triggered.add(13)
    if TARGET_Y > simulated_y and dy < 20 != TARGET_Y > simulated_y and dx < 20: triggered.add(14)
    if TARGET_Y > simulated_y and dy < 20 != TARGET_Y > simulated_y and dz < 20: triggered.add(15)

    if abs(dy) > CRITICAL_X_VELOCITY * 1.5 != abs(dx) > CRITICAL_X_VELOCITY * 1.5: triggered.add(16)
    if abs(dy) > CRITICAL_X_VELOCITY * 1.5 != abs(dz) > CRITICAL_X_VELOCITY * 1.5: triggered.add(17)
    if abs(dy) > CRITICAL_X_VELOCITY * 1.5 != abs(dy) > CRITICAL_X_VELOCITY: triggered.add(18)
    if abs(dy) > CRITICAL_X_VELOCITY * 1.5 != abs(dy) > CRITICAL_X_VELOCITY * 2: triggered.add(19)
    if abs(dy) > CRITICAL_X_VELOCITY * 1.5 != abs(dy) > CRITICAL_Z_VELOCITY * 1.5: triggered.add(20)
    if abs(dy) > CRITICAL_X_VELOCITY * 1.5 != abs(dy) > CRITICAL_Y_VELOCITY * 1.5: triggered.add(21)

    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_X < current_z and dz > CRITICAL_Z_VELOCITY: triggered.add(
        22)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Y < current_z and dz > CRITICAL_Z_VELOCITY: triggered.add(
        23)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Z < current_x and dz > CRITICAL_Z_VELOCITY: triggered.add(
        24)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Z < current_y and dz > CRITICAL_Z_VELOCITY: triggered.add(
        25)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Z < current_z and dx > CRITICAL_Z_VELOCITY: triggered.add(
        26)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Z < current_z and dy > CRITICAL_Z_VELOCITY: triggered.add(
        27)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Z < current_z and dz > CRITICAL_X_VELOCITY: triggered.add(
        28)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Z < current_z and dz > CRITICAL_Y_VELOCITY: triggered.add(
        29)

    return triggered


def jaccard_similarity(a, b):
    if not a and not b: return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def compute_robustness(state, path):
    base_triggered = Tr(state)
    if not base_triggered: return 0.0
    rob, neighbors = 0.0, 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0: continue
                neighbor_state = np.clip(state + np.array([dx, dy, dz]), STATE_MIN, STATE_MAX)
                n_trig = Tr(neighbor_state)
                if not n_trig: continue
                rob += jaccard_similarity(base_triggered, n_trig)
                neighbors += 1
    return rob / neighbors if neighbors > 0 else 0.0


def select_excellent_states(path_indices, all_paths, num_total=2000, top_k=200, weights=(0.33, 0.33, 0.34)):
    selected_samples = {}
    for path_idx in path_indices:
        path = all_paths[path_idx]
        samples = []
        for _ in range(num_total):
            state = np.random.randint(1, 129, size=3)
            triggered = Tr(state)
            sim = jaccard_similarity(triggered, path)
            pl = 1 - abs(len(triggered) - len(path)) / max(len(triggered), len(path)) if triggered else 0.0
            rob = compute_robustness(state, path)
            score = weights[0] * sim + weights[1] * pl + weights[2] * rob
            samples.append((state, score, sim, pl, rob))
        samples.sort(key=lambda x: x[1], reverse=True)
        selected_samples[path_idx] = samples[:top_k]
    return selected_samples

def save_samples_per_path(samples_dict, base_dir, prefix="path", extension=".txt"):
    os.makedirs(base_dir, exist_ok=True)
    for path_id, samples in samples_dict.items():
        filepath = os.path.join(base_dir, f"{prefix}_{path_id + 1}{extension}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Path {path_id + 1} Selected Samples\n")
            f.write("x y z\tScore\tSim\tPl\tRob\n")
            for s in samples:
                state_str = ' '.join(map(str, s[0]))
                f.write(f"{state_str}\t{s[1]:.4f}\t{s[2]:.4f}\t{s[3]:.4f}\t{s[4]:.4f}\n")

if __name__ == '__main__':
    targetPaths = [
        {1, 2, 4, 11, 12, 13, 14, 15},
        {5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29},
        {5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
    ]
    G_high_indices = [0, 1]

    start_time = time.time()
    best_samples = select_excellent_states(G_high_indices, targetPaths, num_total=1000, top_k=50)
    output_dir = "./path_samples"
    save_samples_per_path(best_samples, base_dir=output_dir)