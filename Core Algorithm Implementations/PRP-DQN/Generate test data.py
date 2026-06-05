import random
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Set, Dict

X_MIN, X_MAX = 1, 50
Y_MIN, Y_MAX = 1, 50
Z_MIN, Z_MAX = 1, 50

def Tr(state: List[float]) -> Set[int]:

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

    # Branch 1-29 (keep original logic)
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
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_X < current_z and dz > CRITICAL_Z_VELOCITY: triggered.add(22)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Y < current_z and dz > CRITICAL_Z_VELOCITY: triggered.add(23)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Z < current_x and dz > CRITICAL_Z_VELOCITY: triggered.add(24)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Z < current_y and dz > CRITICAL_Z_VELOCITY: triggered.add(25)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Z < current_z and dx > CRITICAL_Z_VELOCITY: triggered.add(26)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Z < current_z and dy > CRITICAL_Z_VELOCITY: triggered.add(27)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Z < current_z and dz > CRITICAL_X_VELOCITY: triggered.add(28)
    if TARGET_Z < current_z and dz > CRITICAL_Z_VELOCITY != TARGET_Z < current_z and dz > CRITICAL_Y_VELOCITY: triggered.add(29)

    return triggered

def generate_input() -> List[float]:

    return [
        round(random.uniform(X_MIN, X_MAX), 1),
        random.randint(Y_MIN, Y_MAX),
        round(random.uniform(Z_MIN, Z_MAX), 1)
    ]


def jaccard_similarity(set1: Set[int], set2: Set[int]) -> float:
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union else 0

def length_score(path: Set[int], target: Set[int]) -> float:
    return 1 - abs(len(path) - len(target)) / max(len(path), len(target), 1)


def get_26_neighbors(x: float, y: int, z: float) -> List[Tuple[float, int, float]]:
    neighbors = []
    for dx in [-0.1,0,0.1]:
        for dy in [-1,0,1]:
            for dz in [-0.1,0,0.1]:
                if dx==dy==dz==0: continue
                new_x = max(X_MIN, min(X_MAX, round(x+dx,1)))
                new_y = max(Y_MIN, min(Y_MAX, y+dy))
                new_z = max(Z_MIN, min(Z_MAX, round(z+dz,1)))
                neighbors.append((new_x,new_y,new_z))
    return neighbors


def average_neighbor_similarity(x: float, y: int, z: float,
                                path_map: Dict[Tuple[float,int,float], Set[int]]) -> float:
    center_path = path_map[(x,y,z)]
    neighbors = get_26_neighbors(x,y,z)
    similarities = [jaccard_similarity(center_path, path_map[nb]) for nb in neighbors if nb in path_map]
    return sum(similarities)/len(similarities) if similarities else 0.0


def run_scoring_per_path_separate(samples_per_path=1000, top_k=200):
    targetPaths = [
        {1,2,4,11,12,13,14,15},
        {5,6,7,8,9,17,18,19,20,21,24,25,26,27,28,29},
        {5,6,7,8,9,16,17,18,19,20,21,22,23,24,25},
    ]

    results_summary = []

    for tid, target_path in enumerate(targetPaths):
        print(f"Processing target path {tid+1}: {sorted(target_path)}")
        coords = []
        path_map = {}

        print(f"Generating {samples_per_path} samples...")
        for _ in range(samples_per_path):
            x,y,z = generate_input()
            coords.append((x,y,z))
            path_map[(x,y,z)] = Tr([x,y,z])

        rows=[]
        print("Computing scores...")
        for (x,y,z) in tqdm(coords):
            path = path_map[(x,y,z)]
            robust = average_neighbor_similarity(x,y,z,path_map)
            sim = jaccard_similarity(path,set(target_path))
            l_score = length_score(path,set(target_path))
            final = 0.5*sim + 0.5*l_score + 0.0*robust
            rows.append({
                'target_path_id':tid,
                'x':x,'y':y,'z':z,
                'path_len':len(path),
                'actual_path':", ".join(str(i) for i in sorted(path)),
                'similarity':sim,
                'length_score':l_score,
                'robustness':robust,
                'final_score':final
            })

        df_current = pd.DataFrame(rows)
        df_current.sort_values(by="final_score",ascending=False,inplace=True)
        df_top_k = df_current.head(top_k)

        avg_score = df_top_k["final_score"].mean()
        max_score = df_top_k["final_score"].max()
        total_score = df_top_k["final_score"].sum()
        min_score = df_top_k["final_score"].min()

        results_summary.append({
            'path_id':tid+1,
            'target_path':sorted(target_path),
            'samples_generated':samples_per_path,
            'top_k_selected':top_k,
            'average_score':avg_score,
            'total_score':total_score,
            'max_score':max_score,
            'min_score':min_score
        })

        df_top_k.to_csv(f"path_{tid+1}_top{top_k}.csv",index=False)
        print(f"Path {tid+1} completed: top {top_k} sample average score: {avg_score:.6f}, maximum score: {max_score:.6f}, total score: {total_score:.6f}")

    return results_summary

if __name__ == '__main__':
    print(f"Starting scored-sample generation, x/y/z value range: {X_MIN}-{X_MAX}")
    summary = run_scoring_per_path_separate(samples_per_path=1000, top_k=200)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("paths_summary.csv",index=False)
    print("Summary results saved to paths_summary.csv")