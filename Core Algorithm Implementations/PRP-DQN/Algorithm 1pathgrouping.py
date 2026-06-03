import numpy as np

targetPaths = [
    {1, 2, 4, 11, 12, 13, 14, 15},
    {5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29},
    {5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
]

def jaccard_similarity(set_a, set_b):

    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)
def algorithm_1_path_grouping(paths):

    N = len(paths)
    Lambda_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            Lambda_matrix[i][j] = jaccard_similarity(paths[i], paths[j])

    relevance_degrees = []
    for i in range(N):

        sim_sum = sum(Lambda_matrix[i][j] for j in range(N) if i != j)
        relevance_degrees.append(sim_sum / (N - 1))

    Th = sum(relevance_degrees) / N
    base_idx = np.argmax(relevance_degrees)

    G_high = [base_idx]
    G_low = []

    for k in range(N):
        if k == base_idx:
            continue
        s_k = Lambda_matrix[base_idx][k]
        if s_k >= Th:
            G_high.append(k)
        else:
            G_low.append(k)

    return G_high, G_low


if __name__ == '__main__':
    G_high_indices, G_low_indices = algorithm_1_path_grouping(targetPaths)

    print(f"G_high: {[i + 1 for i in G_high_indices]}")
    print(f"G_low: {[i + 1 for i in G_low_indices]}")