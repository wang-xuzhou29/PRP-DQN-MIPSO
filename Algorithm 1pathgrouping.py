import numpy as np

targetPaths = [
    {1, 2, 4, 11, 12, 13, 14, 15},
    {5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29},
    {5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
]

def jaccard_similarity(set_a, set_b):
    """计算两个集合的 Jaccard 相似度 """
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)
def algorithm_1_path_grouping(paths):
    """
    执行 Algorithm 1: 基于高低相关性的路径分组
    """
    N = len(paths)
    Lambda_matrix = np.zeros((N, N))

    # 1. 构建 N x N 相似度矩阵
    for i in range(N):
        for j in range(N):
            Lambda_matrix[i][j] = jaccard_similarity(paths[i], paths[j])

    # 2. 计算每条路径的相关度
    relevance_degrees = []
    for i in range(N):
        # 排除 i==j 的情况计算平均值
        sim_sum = sum(Lambda_matrix[i][j] for j in range(N) if i != j)
        relevance_degrees.append(sim_sum / (N - 1))

    # 3. 计算阈值 Th
    Th = sum(relevance_degrees) / N
    print(f"[*] 计算得到的全局阈值 Th = {Th:.4f}")

    # 4. 选择基准路径 P_base
    base_idx = np.argmax(relevance_degrees)
    print(f"[*] 选择的基准路径 P_base 是 Path {base_idx + 1}")

    # 5. 分组逻辑
    G_high = [base_idx]
    G_low = []

    for k in range(N):
        if k == base_idx:
            continue
        # 计算与基准路径的相似度
        s_k = Lambda_matrix[base_idx][k]
        if s_k >= Th:
            G_high.append(k)
        else:
            G_low.append(k)

    return G_high, G_low


if __name__ == '__main__':
    G_high_indices, G_low_indices = algorithm_1_path_grouping(targetPaths)

    print("\n✅ 分组结果:")
    print(f"高相关性组 (G_high): {[i + 1 for i in G_high_indices]}")
    print(f"低相关性组 (G_low): {[i + 1 for i in G_low_indices]}")