import torch.nn as nn
import os
import torch.optim as optim
import random
from collections import deque
import numpy as np
import torch
from datetime import datetime
import time
import psutil
from statistics import mean
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

# === 设备设置 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# === 归一化/反归一化工具类 ===
# === 状态变量配置：必须与 execute_Tr(dx, dy, dz) 的入参一致 ===
# 根据 Tr 函数中的阈值，dx/dy/dz 至少要覆盖 -50~50 这一带。
# 如果你的真实物理范围不同，只需要改这里，后面归一化、采样、clip 会自动同步。
STATE_RANGES = {
    'dx': (-60, 60),
    'dy': (-60, 60),
    'dz': (-60, 60),
}
STATE_NAMES = ('dx', 'dy', 'dz')
STATE_MIN = np.array([STATE_RANGES[name][0] for name in STATE_NAMES], dtype=np.int32)
STATE_MAX = np.array([STATE_RANGES[name][1] for name in STATE_NAMES], dtype=np.int32)


def clip_state(state):
    """把状态限制在 dx/dy/dz 的有效范围内，并转成 int tuple。"""
    return tuple(np.clip(np.array(state, dtype=np.int32), STATE_MIN, STATE_MAX).astype(int))


def random_state():
    """按照 dx/dy/dz 的有效范围随机生成一个原始状态。"""
    return tuple(random.randint(STATE_RANGES[name][0], STATE_RANGES[name][1]) for name in STATE_NAMES)


class StateNormalizer:
    """状态归一化器：将 dx/dy/dz 映射到 [0, 1]。"""

    def __init__(self, ranges=None):
        self.ranges = ranges or STATE_RANGES
        self.names = STATE_NAMES

    def normalize(self, state):
        """原始状态 -> 归一化状态。"""
        state = np.array(state, dtype=np.float32)
        normalized = np.zeros_like(state, dtype=np.float32)
        for i, name in enumerate(self.names):
            low, high = self.ranges[name]
            normalized[i] = (state[i] - low) / (high - low)
        return normalized

    def denormalize(self, normalized_state):
        """归一化状态 -> 原始 dx/dy/dz，并限制在有效范围内。"""
        normalized_state = np.array(normalized_state, dtype=np.float32)
        denormalized = np.zeros_like(normalized_state, dtype=np.float32)
        for i, name in enumerate(self.names):
            low, high = self.ranges[name]
            denormalized[i] = normalized_state[i] * (high - low) + low
            denormalized[i] = np.clip(np.round(denormalized[i]), low, high).astype(int)
        return denormalized


# 全局归一化器
normalizer = StateNormalizer()


# === 简化的奖励函数 ===
def compute_reward(state, target_path, triggered, prev_triggered=None, prev_state=None):
    sim = jaccard_similarity(triggered, target_path)
    reward = sim * 10
    if target_path.issubset(triggered):
        reward += 1
    return reward


def execute_Tr(dx: int, dy: int, dz: int):
    """执行 Tr 规则。DQN 的状态变量就是这里的 dx、dy、dz。

    注意：下面 current_x/current_y/current_z 仍然是内部模拟变量，不属于 DQN 状态。
    如果你的真实 Tr 规则需要 current_x/current_y/current_z，建议把状态维度扩展为 6 维。
    """
    # --- 1. 常量与配置 ---
    MAX_GRID_SIZE = 500.0
    TARGET_X, TARGET_Y, TARGET_Z = 450.0, 450.0, 200.0

    MIN_PLANNING_X = 10.0
    MIN_PLANNING_Y = 15.0
    MIN_PLANNING_Z = 8.0
    CRITICAL_X_VELOCITY = 20.0
    CRITICAL_Y_VELOCITY = 25.0
    CRITICAL_Z_VELOCITY = 15.0

    triggered = set()

    # 原代码中 current_x/current_y/current_z 未定义。这里仍按原思路用内部模拟值。
    # 但这会导致同一个 (dx, dy, dz) 每次触发结果可能不同。若要求严格可复现，应改成外部输入或确定性计算。
    current_x = random.uniform(0.0, MAX_GRID_SIZE)
    current_y = random.uniform(0.0, MAX_GRID_SIZE)
    current_z = random.uniform(0.0, MAX_GRID_SIZE)
    simulated_y = current_y

    # --- 分支 1-4 ---
    if (abs(dx) < MIN_PLANNING_X) != (abs(dy) < MIN_PLANNING_X): triggered.add(1)
    if (abs(dx) < MIN_PLANNING_X) != (abs(dz) < MIN_PLANNING_X): triggered.add(2)
    if (abs(dx) < MIN_PLANNING_X) != (abs(dx) < MIN_PLANNING_Y): triggered.add(3)
    if (abs(dx) < MIN_PLANNING_X) != (abs(dx) < MIN_PLANNING_Z): triggered.add(4)

    # --- 分支 5-9 ---
    if (abs(dz) > MIN_PLANNING_Z * 2) != (abs(dx) > MIN_PLANNING_Z * 2): triggered.add(5)
    if (abs(dz) > MIN_PLANNING_Z * 2) != (abs(dy) > MIN_PLANNING_Z * 2): triggered.add(6)
    if (abs(dz) > MIN_PLANNING_Z * 2) != (abs(dz) > MIN_PLANNING_X * 2): triggered.add(7)
    if (abs(dz) > MIN_PLANNING_Z * 2) != (abs(dz) > MIN_PLANNING_Y * 2): triggered.add(8)
    if (abs(dz) > MIN_PLANNING_Z * 2) != (abs(dz) > MIN_PLANNING_Z): triggered.add(9)

    # --- 分支 10-15 ---
    if ((TARGET_Y > simulated_y) and (dy < 20)) != ((TARGET_Y > simulated_y) and (dy < 10)): triggered.add(10)
    if ((TARGET_Y > simulated_y) and (dy < 20)) != ((TARGET_Y > simulated_y) and (dy < 30)): triggered.add(11)
    if ((TARGET_Y > simulated_y) and (dy < 20)) != ((TARGET_Y > simulated_y) and (dy < 40)): triggered.add(12)
    if ((TARGET_Y > simulated_y) and (dy < 20)) != ((TARGET_Y > simulated_y) and (dy < 50)): triggered.add(13)
    if ((TARGET_Y > simulated_y) and (dy < 20)) != ((TARGET_Y > simulated_y) and (dx < 20)): triggered.add(14)
    if ((TARGET_Y > simulated_y) and (dy < 20)) != ((TARGET_Y > simulated_y) and (dz < 20)): triggered.add(15)

    # --- 分支 16-21 ---
    if (abs(dy) > CRITICAL_X_VELOCITY * 1.5) != (abs(dx) > CRITICAL_X_VELOCITY * 1.5): triggered.add(16)
    if (abs(dy) > CRITICAL_X_VELOCITY * 1.5) != (abs(dz) > CRITICAL_X_VELOCITY * 1.5): triggered.add(17)
    if (abs(dy) > CRITICAL_X_VELOCITY * 1.5) != (abs(dy) > CRITICAL_X_VELOCITY): triggered.add(18)
    if (abs(dy) > CRITICAL_X_VELOCITY * 1.5) != (abs(dy) > CRITICAL_X_VELOCITY * 2): triggered.add(19)
    if (abs(dy) > CRITICAL_X_VELOCITY * 1.5) != (abs(dy) > CRITICAL_Z_VELOCITY * 1.5): triggered.add(20)
    if (abs(dy) > CRITICAL_X_VELOCITY * 1.5) != (abs(dy) > CRITICAL_Y_VELOCITY * 1.5): triggered.add(21)

    # --- 分支 22-29 ---
    if ((TARGET_Z < current_z) and (dz > CRITICAL_Z_VELOCITY)) != ((TARGET_X < current_z) and (dz > CRITICAL_Z_VELOCITY)): triggered.add(22)
    if ((TARGET_Z < current_z) and (dz > CRITICAL_Z_VELOCITY)) != ((TARGET_Y < current_z) and (dz > CRITICAL_Z_VELOCITY)): triggered.add(23)
    if ((TARGET_Z < current_z) and (dz > CRITICAL_Z_VELOCITY)) != ((TARGET_Z < current_x) and (dz > CRITICAL_Z_VELOCITY)): triggered.add(24)
    if ((TARGET_Z < current_z) and (dz > CRITICAL_Z_VELOCITY)) != ((TARGET_Z < current_y) and (dz > CRITICAL_Z_VELOCITY)): triggered.add(25)
    if ((TARGET_Z < current_z) and (dz > CRITICAL_Z_VELOCITY)) != ((TARGET_Z < current_z) and (dx > CRITICAL_Z_VELOCITY)): triggered.add(26)
    if ((TARGET_Z < current_z) and (dz > CRITICAL_Z_VELOCITY)) != ((TARGET_Z < current_z) and (dy > CRITICAL_Z_VELOCITY)): triggered.add(27)
    if ((TARGET_Z < current_z) and (dz > CRITICAL_Z_VELOCITY)) != ((TARGET_Z < current_z) and (dz > CRITICAL_X_VELOCITY)): triggered.add(28)
    if ((TARGET_Z < current_z) and (dz > CRITICAL_Z_VELOCITY)) != ((TARGET_Z < current_z) and (dz > CRITICAL_Y_VELOCITY)): triggered.add(29)

    return triggered


target_paths = [
    {1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29},
    {5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
{5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29}
]

# 转换为集合
target_paths = [set(path) for path in target_paths]


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if set2.issubset(set1):
        return 1.0
    return intersection / union if union != 0 else 0.0


def compute_path_similarity_matrix(paths):
    n = len(paths)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = len(paths[i] & paths[j])
            union = len(paths[i] | paths[j])
            matrix[i][j] = inter / union if union > 0 else 0.0
    return matrix


def group_paths_by_similarity(paths):
    sim_matrix = compute_path_similarity_matrix(paths)
    avg_sim_scores = np.mean(sim_matrix, axis=1)
    threshold = np.mean(avg_sim_scores)

    center_idx = np.argmax(avg_sim_scores)
    similar_group = [center_idx]
    for i in range(len(paths)):
        if i != center_idx and sim_matrix[center_idx][i] > threshold:
            similar_group.append(i)

    isolated_group = [i for i in range(len(paths)) if i not in similar_group]
    return similar_group, isolated_group


# === 随机分组配置 ===
# 默认：随机分成两个组，第一组数量等于原“相似度分组”方法得到的相似组数量。
# 如果希望每次运行时通过键盘输入第一组数量，将 USE_KEYBOARD_INPUT_GROUP_SIZE 改为 True。
# 如果希望随机结果可复现，将 RANDOM_GROUP_SEED 设置为固定整数，例如 2026；如果希望每次随机不同，保持 None。
USE_KEYBOARD_INPUT_GROUP_SIZE = False
RANDOM_GROUP_SEED = None


def group_paths_randomly(paths, use_keyboard_input=False, seed=None):
    """
    随机分组函数：只分两个组。
    - 随机组1：先训练模型；
    - 随机组2：复用随机组1模型进行样本评分和迁移训练。

    默认随机组1数量 = 原 group_paths_by_similarity(paths) 得到的第一组数量，
    这样可以保证随机分组与原方法的两组规模一致，只改变路径归属方式。
    """
    n_paths = len(paths)

    # 用原相似度分组方法仅获取组规模，不再按相似度决定路径归属。
    original_group1, original_group2 = group_paths_by_similarity(paths)
    default_group1_size = len(original_group1)

    group1_size = default_group1_size

    if use_keyboard_input:
        while True:
            user_input = input(
                f"请输入随机组1的路径数量，范围 1~{n_paths - 1}；"
                f"直接回车则使用原方法数量 {default_group1_size}："
            ).strip()

            if user_input == "":
                group1_size = default_group1_size
                break

            try:
                group1_size = int(user_input)
                if 1 <= group1_size <= n_paths - 1:
                    break
                print(f"输入无效：随机组1数量必须在 1~{n_paths - 1} 之间。")
            except ValueError:
                print("输入无效：请输入整数，或直接回车使用默认数量。")

    if not (1 <= group1_size <= n_paths - 1):
        raise ValueError(f"随机组1数量必须在 1~{n_paths - 1} 之间，当前为 {group1_size}")

    all_indices = list(range(n_paths))
    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(all_indices)

    random_group1 = sorted(all_indices[:group1_size])
    random_group2 = sorted(all_indices[group1_size:])

    return random_group1, random_group2, default_group1_size, len(original_group2)


def compute_robustness(state, path):
    """计算鲁棒性（输入为原始值）"""
    dx, dy, dz = state
    base = execute_Tr(dx, dy, dz)
    if not base:
        return 0.0

    rob, neighbors = 0.0, 0
    for dw in [-1, 0, 1]:
        for dt in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dw == dt == dz == 0:
                    continue
                # 使用新的变量范围进行限制
                neighbor = np.clip(np.array(state) + np.array([dw, dt, dz]),
                                   STATE_MIN, STATE_MAX)
                neighbor = tuple(neighbor)
                ndx, ndy, ndz = neighbor
                n_trig = execute_Tr(ndx, ndy, ndz)
                if not n_trig:
                    continue
                rob += jaccard_similarity(base, n_trig)
                neighbors += 1
    return rob / neighbors if neighbors > 0 else 0.0


def compute_q_value_score(state, similar_model):
    """计算Q值分数（输入为原始值）"""
    if similar_model is None:
        return 0.0

    try:
        # 归一化后输入模型
        normalized_state = normalizer.normalize(state)
        state_tensor = torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = similar_model(state_tensor)
            max_q_value = torch.max(q_values).item()
            normalized_q = max_q_value / 20.0
            normalized_q = max(0.0, min(1.0, normalized_q))
            return 1.0 - normalized_q
    except:
        return 0.0


def generate_samples_for_similar_paths(similar_group, num_candidates=2000, top_k=200, run_id=1):
    SIMILAR_WEIGHTS = [0.55, 0.39, 0.06]

    def save_samples(path_id, samples, base_dir, group_type="similar"):
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"path{path_id}_{group_type}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{group_type.title()} Group Path {path_id} - Run {run_id}\n")
            f.write("dx dy dz\tScore\tSimilarity\tLengthDiff\tRobustness\n")
            for s in samples:
                dx, dy, dz = s['state']
                f.write(
                    f"{dx} {dy} {dz}\t{s['score']:.4f}\t{s['similarity']:.4f}\t{s['length_diff']:.4f}\t{s['robustness']:.4f}\n")

    base_dir = r"D:\实验\CNN\DQNNEW\path_samples_grouped"

    for path_idx in similar_group:
        path = target_paths[path_idx]
        path_id = path_idx + 1
        candidate_samples = []
        attempts = 0

        while len(candidate_samples) < num_candidates and attempts < num_candidates * 10:
            attempts += 1
            # 使用 dx/dy/dz 的变量范围生成状态
            state = random_state()
            dx, dy, dz = state

            triggered = execute_Tr(dx, dy, dz)

            if not triggered:
                continue

            sim = jaccard_similarity(triggered, path)
            len_diff = 1 - abs(len(triggered) - len(path)) / max(len(triggered), len(path))
            rob = compute_robustness(state, path)

            candidate_samples.append({
                'state': state,
                'similarity': sim,
                'length_diff': len_diff,
                'robustness': rob,
                'triggered': triggered
            })

        if candidate_samples:
            for sample in candidate_samples:
                score = (SIMILAR_WEIGHTS[0] * sample['similarity'] +
                         SIMILAR_WEIGHTS[1] * sample['length_diff'] +
                         SIMILAR_WEIGHTS[2] * sample['robustness'])
                sample['score'] = score

            candidate_samples.sort(key=lambda x: x['score'], reverse=True)
            selected_samples = candidate_samples[:top_k]
            save_samples(path_id=path_id, samples=selected_samples, base_dir=base_dir, group_type="similar")


def generate_samples_for_isolated_paths(isolated_group, similar_model, num_candidates=2000, top_k=200, run_id=1):
    ISOLATED_WEIGHTS = [0.18, 0.21, 0.32, 0.29]

    def save_samples(path_id, samples, base_dir, group_type="isolated"):
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"path{path_id}_{group_type}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{group_type.title()} Group Path {path_id} - Run {run_id}\n")
            f.write("dx dy dz\tScore\tSimilarity\tLengthDiff\tRobustness\tQValueScore\n")
            for s in samples:
                dx, dy, dz = s['state']
                f.write(
                    f"{dx} {dy} {dz}\t{s['score']:.4f}\t{s['similarity']:.4f}\t{s['length_diff']:.4f}\t{s['robustness']:.4f}\t{s['q_value_score']:.4f}\n")

    base_dir = r"D:\实验\CNN\DQNNEW\path_samples_grouped"

    for path_idx in isolated_group:
        path = target_paths[path_idx]
        path_id = path_idx + 1
        candidate_samples = []
        attempts = 0

        while len(candidate_samples) < num_candidates and attempts < num_candidates * 10:
            attempts += 1
            # 使用 dx/dy/dz 的变量范围生成状态
            state = random_state()
            dx, dy, dz = state

            triggered = execute_Tr(dx, dy, dz)

            if not triggered:
                continue

            sim = jaccard_similarity(triggered, path)
            len_diff = 1 - abs(len(triggered) - len(path)) / max(len(triggered), len(path))
            rob = compute_robustness(state, path)
            q_score = compute_q_value_score(state, similar_model)

            candidate_samples.append({
                'state': state,
                'similarity': sim,
                'length_diff': len_diff,
                'robustness': rob,
                'q_value_score': q_score,
                'triggered': triggered
            })

        if candidate_samples:
            for sample in candidate_samples:
                score = (ISOLATED_WEIGHTS[0] * sample['similarity'] +
                         ISOLATED_WEIGHTS[1] * sample['length_diff'] +
                         ISOLATED_WEIGHTS[2] * sample['robustness'] +
                         ISOLATED_WEIGHTS[3] * sample['q_value_score'])
                sample['score'] = score

            candidate_samples.sort(key=lambda x: x['score'], reverse=True)
            selected_samples = candidate_samples[:top_k]
            save_samples(path_id=path_id, samples=selected_samples, base_dir=base_dir, group_type="isolated")


class GroupExperienceReplay:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.priorities = deque(maxlen=self.capacity)
        self.sampled_indices = set()  # 记录已抽取的索引

    def append(self, experience):
        self.buffer.append(experience)
        self.priorities.append(experience[-1])

    def sample(self, batch_size, alpha=0.6):
        priorities = np.array(self.priorities) ** alpha
        probabilities = priorities / np.sum(priorities)
        batch_indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in batch_indices]
        return batch, batch_indices, probabilities[batch_indices]

    def update_priorities(self, batch_indices, td_errors):
        for idx, td_error in zip(batch_indices, td_errors):
            if idx < len(self.priorities):
                self.priorities[idx] = max(td_error, 1e-6)

    def __len__(self):
        return len(self.buffer)

    def get_high_reward_samples(self, target_path, num_samples=20):
        """获取高奖励样本（不重复抽取，返回原始值）"""
        if len(self.buffer) == 0:
            return []

        samples_with_recalculated_scores = []
        for idx, experience in enumerate(self.buffer):
            # 跳过已抽取的样本
            if idx in self.sampled_indices:
                continue

            # 从归一化状态反归一化
            normalized_state_tensor = experience[0]
            normalized_state = normalized_state_tensor.cpu().numpy().flatten()
            state_tuple = tuple(normalizer.denormalize(normalized_state))

            dx, dy, dz = state_tuple
            triggered = execute_Tr(dx, dy, dz)
            new_reward = compute_reward(state_tuple, target_path, triggered, None, None)
            sim = jaccard_similarity(triggered, target_path)
            samples_with_recalculated_scores.append((idx, state_tuple, new_reward, sim, triggered))

        # 按奖励排序
        samples_with_recalculated_scores.sort(key=lambda x: x[2], reverse=True)

        # 取前num_samples个
        selected = samples_with_recalculated_scores[:num_samples]

        # 记录已抽取的索引
        for item in selected:
            self.sampled_indices.add(item[0])

        # 返回格式：(state_tuple, reward, sim, triggered)
        return [(s[1], s[2], s[3], s[4]) for s in selected]

    def reset_sampled_indices(self):
        """重置已抽取索引记录"""
        self.sampled_indices.clear()


def load_path_data(file_path):
    """加载路径数据（原始值）"""
    path_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[2:]:
            parts = line.strip().split("\t")
            state = tuple(map(int, parts[0].split()))
            path_data.append(state)
    return path_data


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgentWithPER:
    def __init__(self, state_dim, action_dim, replay_buffer, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.1, learning_rate=0.001, alpha=0.6, beta=0.4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.replay_buffer = replay_buffer
        self.alpha = alpha
        self.beta = beta

        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.target_model.load_state_dict(self.model.state_dict())

    def decode_action(self, action_idx):
        """动作解码（适配新变量范围）"""
        delta_values = [1, -1]  # 调整为较小的步长，适应新的变量范围
        dim = action_idx // 2
        delta_idx = action_idx % 2
        delta = delta_values[delta_idx]
        if dim == 0:
            return (delta, 0, 0)
        elif dim == 1:
            return (0, delta, 0)
        elif dim == 2:
            return (0, 0, delta)

    def act(self, normalized_state):
        """选择动作（输入为归一化状态）"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values, dim=1).item()

    def store_transition(self, normalized_state, action, reward, normalized_next_state, done):
        """存储经验（输入为归一化状态）"""
        state = torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0).to(device)
        next_state = torch.tensor(normalized_next_state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.model(state)
            next_q_values = self.target_model(next_state)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = reward + (self.gamma * max_next_q_values * (1 - done))
            td_error = torch.abs(q_values[0][action] - target_q_values).item()

        self.replay_buffer.append((state, action, reward, next_state, done, td_error))
        return td_error

    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        batch, batch_indices, probabilities = self.replay_buffer.sample(batch_size, alpha=self.alpha)
        states, actions, rewards, next_states, dones, _ = zip(*batch)

        weights = (len(self.replay_buffer) * probabilities) ** (-self.beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        states = torch.tensor(np.array([s.cpu().numpy().flatten() for s in states]), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array([ns.cpu().numpy().flatten() for ns in next_states]),
                                   dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_max_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_max_q_values * (1 - dones))

        td_errors = current_q_values - target_q_values
        weighted_loss = (td_errors.pow(2) * weights).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        new_priorities = torch.abs(td_errors).detach().cpu().numpy()
        self.replay_buffer.update_priorities(batch_indices, new_priorities)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


def train_group(group_paths, path_documents, replay_buffer, batch_size=32, group_name="", pretrained_model=None,
                sample_group_type=None):
    """训练函数（使用归一化）- 调整为3分钟版本规模"""
    # sample_group_type 只控制读取的样本文件后缀：similar / isolated。
    # 这样可以把显示名称改为“随机组1/随机组2”，但不破坏原来的样本文件命名逻辑。
    if sample_group_type is None:
        sample_group_type = 'similar' if group_name == '相似组' else 'isolated'
    state_dim = 3
    action_dim = 6  # 调整为2*3 (2个delta值 * 3个维度)

    agent = DQNAgentWithPER(state_dim, action_dim, replay_buffer)

    if pretrained_model is not None:
        print(f"  {group_name}：加载预训练模型权重（迁移学习）...")
        agent.model.load_state_dict(pretrained_model.state_dict())
        agent.target_model.load_state_dict(pretrained_model.state_dict())
        print(f"  {group_name}：模型权重迁移完成！")

    path_rewards = {}

    print(f"开始训练{group_name}，包含路径: {[idx + 1 for idx in group_paths]}")
    start_time = time.time()

    # === 调整为3分钟版本规模 ===
    BATCH_SIZE = 50  # 批次大小
    N_SAMPLES = 200  # 样本数量
    N_STEPS = 3  # 步数
    N_ROUNDS = 5  # 重复轮次
    N_BATCHES = 4  # 批次数量

    replay_count = 0

    for path_idx in group_paths:
        file_path = os.path.join(path_documents,
                                 f"path{path_idx + 1}_{sample_group_type}.txt")
        if not os.path.exists(file_path):
            print(f"    警告：路径{path_idx + 1}文件不存在，跳过")
            continue

        path_data = load_path_data(file_path)  # 原始值
        target_path = target_paths[path_idx]

        if path_idx not in path_rewards:
            path_rewards[path_idx] = 0

        print(f"\n  开始训练路径 {path_idx + 1}，将进行 {N_ROUNDS} 轮完整训练")

        for round_idx in range(N_ROUNDS):
            print(f"    路径 {path_idx + 1} - 第 {round_idx + 1}/{N_ROUNDS} 轮训练")

            for batch_idx in range(N_BATCHES):
                batch_start = batch_idx * BATCH_SIZE
                batch_end = min(batch_start + BATCH_SIZE, N_SAMPLES)

                # 如果样本不足，跳过该批次
                if batch_start >= len(path_data):
                    print(f"      批次 {batch_idx + 1}: 样本不足，跳过")
                    break

                print(f"      批次 {batch_idx + 1}/{N_BATCHES} (样本 {batch_start}-{batch_end})")

                for sample_idx in range(batch_start, batch_end):
                    if sample_idx >= len(path_data):
                        break

                    state = path_data[sample_idx]  # 原始值
                    prev_state = None
                    prev_triggered = None

                    for step in range(N_STEPS):
                        # 归一化状态用于模型
                        normalized_state = normalizer.normalize(state)

                        # 获取合法动作
                        legal_actions = []
                        for a in range(agent.action_dim):
                            dw, dt, dz = agent.decode_action(a)
                            # 使用新的变量范围进行限制
                            cand_next = tuple(np.clip(np.array(state) + np.array([dw, dt, dz]),
                                                      STATE_MIN, STATE_MAX))
                            legal_actions.append(a)

                        if not legal_actions:
                            break

                        # 选择动作
                        if random.random() < agent.epsilon:
                            action = random.choice(legal_actions)
                        else:
                            state_tensor = torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0).to(device)
                            with torch.no_grad():
                                q_values = agent.model(state_tensor)[0]
                            action = legal_actions[torch.argmax(q_values[legal_actions]).item()]

                        # 执行动作（原始空间）
                        dw, dt, dz = agent.decode_action(action)
                        next_state = tuple(np.clip(np.array(state) + np.array([dw, dt, dz]),
                                                   STATE_MIN, STATE_MAX))

                        # 归一化下一状态
                        normalized_next_state = normalizer.normalize(next_state)

                        # 计算奖励（使用原始值）
                        dx, dy, dz = next_state
                        triggered = execute_Tr(dx, dy, dz)
                        reward = compute_reward(next_state, target_path, triggered,
                                                prev_triggered, prev_state)
                        done = (step == N_STEPS - 1)

                        # 存储经验（归一化状态）
                        agent.store_transition(normalized_state, action, reward, normalized_next_state, done)

                        # 更新状态
                        prev_state = state
                        prev_triggered = triggered
                        state = next_state
                        path_rewards[path_idx] += reward

                # 回放训练
                if len(agent.replay_buffer) >= batch_size:
                    agent.train(batch_size)
                    replay_count += 1

                    if replay_count % 2 == 0:
                        agent.update_target_model()

            print(f"      路径 {path_idx + 1} - 第 {round_idx + 1} 轮完成")

        print(f"  路径 {path_idx + 1} 的 {N_ROUNDS} 轮训练全部完成 ✅")

    training_time = time.time() - start_time
    print(f"\n{group_name}训练完成!")
    print(f"  训练方式: 每条路径独立完成{N_ROUNDS}轮训练（归一化）✅")
    print(f"  总步数: {replay_count}次回放训练")
    print(f"  用时: {training_time:.2f}秒")
    print(f"  经验池大小: {len(replay_buffer)}")

    return agent, path_rewards, training_time


def generate_and_train_grouped_paths_staged(path_documents, random_group1, random_group2, batch_size=32, run_id=1):
    """分阶段训练（归一化版本）- 随机分组 + 模型复用"""
    print(f"\n=== 运行 {run_id}/20 分阶段训练开始（3分钟版本规模，随机分组+模型复用） ===")
    random_group1_paths = [idx + 1 for idx in random_group1]
    random_group2_paths = [idx + 1 for idx in random_group2]

    print(f"随机组1路径（先训练组）: {random_group1_paths}")
    print(f"随机组2路径（模型复用组）: {random_group2_paths}")

    total_start_time = time.time()

    print(f"\n[阶段1] 为随机组1生成样本...")
    # 样本生成规模保持不变；文件后缀仍保存为 similar，便于复用原读取逻辑。
    generate_samples_for_similar_paths(random_group1, num_candidates=2000, top_k=200, run_id=run_id)

    print(f"\n[阶段2] 训练随机组1（归一化，{5}轮完整训练）...")
    group1_replay_buffer = GroupExperienceReplay(capacity=20000)
    group1_agent, group1_path_rewards, group1_training_time = train_group(
        random_group1, path_documents, group1_replay_buffer, batch_size,
        group_name="随机组1（先训练组）", pretrained_model=None, sample_group_type="similar"
    )

    print(f"\n[阶段3] 使用随机组1模型为随机组2生成样本...")
    # 模型复用点1：随机组2样本评分中的 QValueScore 使用随机组1训练后的模型。
    generate_samples_for_isolated_paths(random_group2, group1_agent.model,
                                        num_candidates=2000, top_k=200, run_id=run_id)

    print(f"\n[阶段4] 训练随机组2（继承随机组1模型，{5}轮完整训练）...")
    group2_replay_buffer = GroupExperienceReplay(capacity=20000)
    # 模型复用点2：随机组2训练前加载随机组1模型权重，相当于迁移学习。
    group2_agent, group2_path_rewards, group2_training_time = train_group(
        random_group2, path_documents, group2_replay_buffer, batch_size,
        group_name="随机组2（模型复用组）", pretrained_model=group1_agent.model, sample_group_type="isolated"
    )

    total_path_rewards = {**group1_path_rewards, **group2_path_rewards}
    total_cumulative_reward = sum(total_path_rewards.values())
    total_training_time = time.time() - total_start_time

    print(f"\n=== 运行 {run_id}/20 完成，总用时: {total_training_time:.2f}秒 ===")
    print(f"随机组1训练用时: {group1_training_time:.2f}秒")
    print(f"随机组2训练用时: {group2_training_time:.2f}秒")
    print(f"经验池统计 - 随机组1: {len(group1_replay_buffer)}, 随机组2: {len(group2_replay_buffer)}")

    return group1_agent, group2_agent, group1_replay_buffer, group2_replay_buffer, \
        total_cumulative_reward, total_path_rewards, total_training_time

def create_consolidated_excel_report(all_runs_data, similar_group, isolated_group, output_dir):
    """创建Excel报告（保持原有功能）"""
    os.makedirs(output_dir, exist_ok=True)

    similar_group_paths = [idx + 1 for idx in similar_group]
    isolated_group_paths = [idx + 1 for idx in isolated_group]

    wb = Workbook()

    thin_border = Border(
        left=Side(style='thin', color='000000'),
        right=Side(style='thin', color='000000'),
        top=Side(style='thin', color='000000'),
        bottom=Side(style='thin', color='000000')
    )

    header_color = "4472C4"
    similar_group_color = "E2EFDA"
    isolated_group_color = "FCE4D6"
    stats_color = "FFF2CC"

    ws_paths = wb.active
    ws_paths.title = "各路径详细表现"

    path_headers = ['路径编号', '分组类型'] + [f'第{i}次' for i in range(1, 21)] + ['平均相似度', '最高相似度',
                                                                                    '最低相似度', '标准差']
    for col, header in enumerate(path_headers, 1):
        cell = ws_paths.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = PatternFill(start_color=header_color, end_color=header_color, fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    ws_paths.row_dimensions[1].height = 30

    for path_id in range(1, len(target_paths) + 1):
        row = path_id + 1

        if path_id in similar_group_paths:
            group_type = "随机组1（先训练组）"
            row_color = similar_group_color
        elif path_id in isolated_group_paths:
            group_type = "随机组2（模型复用组）"
            row_color = isolated_group_color
        else:
            group_type = "未分组"
            row_color = "FFFFFF"

        cell = ws_paths.cell(row=row, column=1, value=f"路径{path_id}")
        cell.font = Font(bold=True, size=10)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.fill = PatternFill(start_color=row_color, end_color=row_color, fill_type="solid")
        cell.border = thin_border

        cell = ws_paths.cell(row=row, column=2, value=group_type)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.fill = PatternFill(start_color=row_color, end_color=row_color, fill_type="solid")
        cell.border = thin_border

        path_similarities = []
        for run_idx, run_data in enumerate(all_runs_data):
            sim = run_data['path_similarities'].get(path_id, {}).get('avg_similarity', 0.0)
            path_similarities.append(sim)

            cell = ws_paths.cell(row=row, column=3 + run_idx, value=round(sim, 4))
            cell.number_format = '0.0000'
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border

        stats_values = [
            round(np.mean(path_similarities), 4),
            round(np.max(path_similarities), 4),
            round(np.min(path_similarities), 4),
            round(np.std(path_similarities), 4)
        ]

        for i, value in enumerate(stats_values):
            cell = ws_paths.cell(row=row, column=23 + i, value=value)
            cell.number_format = '0.0000'
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.fill = PatternFill(start_color=stats_color, end_color=stats_color, fill_type="solid")
            cell.font = Font(bold=True, size=10)
            cell.border = thin_border

    ws_paths.column_dimensions['A'].width = 13
    ws_paths.column_dimensions['B'].width = 16
    for col in range(3, 23):
        ws_paths.column_dimensions[get_column_letter(col)].width = 10
    for col in range(23, 27):
        ws_paths.column_dimensions[get_column_letter(col)].width = 13

    # === 工作表2: 分组统计 ===
    ws_groups = wb.create_sheet("分组统计")

    # 设置标题（删除了"筛选标准"列）
    group_headers = ['分组名称', '包含路径'] + [f'第{i}次' for i in range(1, 21)] + ['平均相似度', '标准差']
    for col, header in enumerate(group_headers, 1):
        cell = ws_groups.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = PatternFill(start_color=header_color, end_color=header_color, fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    ws_groups.row_dimensions[1].height = 30

    row = 2

    # 相似路径组
    cell = ws_groups.cell(row=row, column=1, value="随机组1（先训练组）")
    cell.font = Font(bold=True, size=11)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.fill = PatternFill(start_color=similar_group_color, end_color=similar_group_color, fill_type="solid")
    cell.border = thin_border

    cell = ws_groups.cell(row=row, column=2, value=','.join(map(str, similar_group_paths)))
    cell.alignment = Alignment(horizontal="left", vertical="center")
    cell.fill = PatternFill(start_color=similar_group_color, end_color=similar_group_color, fill_type="solid")
    cell.border = thin_border

    group_similarities = []
    for run_idx, run_data in enumerate(all_runs_data):
        group_sim = np.mean(
            [run_data['path_similarities'].get(p, {}).get('avg_similarity', 0.0) for p in similar_group_paths])
        group_similarities.append(group_sim)

        cell = ws_groups.cell(row=row, column=3 + run_idx, value=round(group_sim, 4))
        cell.number_format = '0.0000'
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    # 统计列
    cell = ws_groups.cell(row=row, column=23, value=round(np.mean(group_similarities), 4))
    cell.number_format = '0.0000'
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.fill = PatternFill(start_color=stats_color, end_color=stats_color, fill_type="solid")
    cell.font = Font(bold=True, size=11)
    cell.border = thin_border

    cell = ws_groups.cell(row=row, column=24, value=round(np.std(group_similarities), 4))
    cell.number_format = '0.0000'
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.fill = PatternFill(start_color=stats_color, end_color=stats_color, fill_type="solid")
    cell.font = Font(bold=True, size=11)
    cell.border = thin_border

    row += 1

    # 孤岛路径组
    if isolated_group_paths:
        cell = ws_groups.cell(row=row, column=1, value="随机组2（模型复用组）")
        cell.font = Font(bold=True, size=11)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.fill = PatternFill(start_color=isolated_group_color, end_color=isolated_group_color, fill_type="solid")
        cell.border = thin_border

        cell = ws_groups.cell(row=row, column=2, value=','.join(map(str, isolated_group_paths)))
        cell.alignment = Alignment(horizontal="left", vertical="center")
        cell.fill = PatternFill(start_color=isolated_group_color, end_color=isolated_group_color, fill_type="solid")
        cell.border = thin_border

        isolated_similarities = []
        for run_idx, run_data in enumerate(all_runs_data):
            iso_sim = np.mean(
                [run_data['path_similarities'].get(p, {}).get('avg_similarity', 0.0) for p in isolated_group_paths])
            isolated_similarities.append(iso_sim)

            cell = ws_groups.cell(row=row, column=3 + run_idx, value=round(iso_sim, 4))
            cell.number_format = '0.0000'
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border

        # 统计列
        cell = ws_groups.cell(row=row, column=23, value=round(np.mean(isolated_similarities), 4))
        cell.number_format = '0.0000'
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.fill = PatternFill(start_color=stats_color, end_color=stats_color, fill_type="solid")
        cell.font = Font(bold=True, size=11)
        cell.border = thin_border

        cell = ws_groups.cell(row=row, column=24, value=round(np.std(isolated_similarities), 4))
        cell.number_format = '0.0000'
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.fill = PatternFill(start_color=stats_color, end_color=stats_color, fill_type="solid")
        cell.font = Font(bold=True, size=11)
        cell.border = thin_border

    # 调整列宽
    ws_groups.column_dimensions['A'].width = 16
    ws_groups.column_dimensions['B'].width = 22
    for col in range(3, 23):
        ws_groups.column_dimensions[get_column_letter(col)].width = 10
    ws_groups.column_dimensions[get_column_letter(23)].width = 14
    ws_groups.column_dimensions[get_column_letter(24)].width = 12

    # === 工作表3: 详细样本数据 ===
    ws_samples = wb.create_sheet("详细样本数据")

    # 样本数据标题（删除了"分组类型"列）
    sample_headers = ['运行次数', '路径编号', '样本序号', 'Dx', 'Dy', 'Dz', '相似度', '触发规则集合']
    for col, header in enumerate(sample_headers, 1):
        cell = ws_samples.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = PatternFill(start_color=header_color, end_color=header_color, fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    ws_samples.row_dimensions[1].height = 30

    sample_row = 2
    # 输出每次运行每条路径的所有样本数据
    for run_idx, run_data in enumerate(all_runs_data, 1):
        for path_id in range(1, len(target_paths) + 1):
            samples = run_data['path_samples'].get(path_id, [])

            # 确定路径背景色
            if path_id in similar_group_paths:
                path_color = similar_group_color
            elif path_id in isolated_group_paths:
                path_color = isolated_group_color
            else:
                path_color = "FFFFFF"

            for sample_idx, (state_tuple, reward, sim, triggered) in enumerate(samples, 1):
                dx, dy, dz = state_tuple
                triggered_str = ','.join(map(str, sorted(triggered)))

                # 运行次数
                cell = ws_samples.cell(row=sample_row, column=1, value=f"第{run_idx}次")
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.fill = PatternFill(start_color=path_color, end_color=path_color, fill_type="solid")
                cell.border = thin_border

                # 路径编号
                cell = ws_samples.cell(row=sample_row, column=2, value=f"路径{path_id}")
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.fill = PatternFill(start_color=path_color, end_color=path_color, fill_type="solid")
                cell.border = thin_border

                # 样本序号
                cell = ws_samples.cell(row=sample_row, column=3, value=sample_idx)
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = thin_border

                # Dx, Dy, Dz值
                for col_offset, value in enumerate([dx, dy, dz]):
                    cell = ws_samples.cell(row=sample_row, column=4 + col_offset, value=value)
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = thin_border

                # 相似度
                cell = ws_samples.cell(row=sample_row, column=7, value=round(sim, 4))
                cell.number_format = '0.0000'
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = thin_border

                # 触发规则集合
                cell = ws_samples.cell(row=sample_row, column=8, value=f"{{{triggered_str}}}")
                cell.alignment = Alignment(horizontal="left", vertical="center")
                cell.border = thin_border

                sample_row += 1

    # 调整列宽
    sample_widths = [13, 13, 11, 10, 12, 8, 12, 45]
    for i, width in enumerate(sample_widths, 1):
        ws_samples.column_dimensions[get_column_letter(i)].width = width

    # 保存文件
    output_path = os.path.join(output_dir, "20次运行综合报告_随机分组_模型复用_3分钟版本.xlsx")
    wb.save(output_path)
    print(f"\n✅ 综合Excel报告已生成: {output_path}")


def run_20_times_training():
    """运行20次训练（调整为3分钟版本规模）- 优化保存方案"""
    model_path_base = r"D:\实验\CNN\DQNNEW\saved_models_random_reuse_3min_version"
    path_documents = r"D:\实验\CNN\DQNNEW\path_samples_grouped"
    output_dir = r"D:\实验\CNN\对比实验二\excel_reports_random_reuse_3min_version"

    os.makedirs(model_path_base, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 随机分组：只分两个组。
    # 默认第一组数量与原相似度分组方法得到的第一组数量一致；也可以开启键盘输入。
    similar_group, isolated_group, default_group1_size, default_group2_size = group_paths_randomly(
        target_paths,
        use_keyboard_input=USE_KEYBOARD_INPUT_GROUP_SIZE,
        seed=RANDOM_GROUP_SEED
    )

    similar_group_display = [idx + 1 for idx in similar_group]
    isolated_group_display = [idx + 1 for idx in isolated_group]

    print("=" * 60)
    print("开始20次连续训练实验 - 随机分组 + 模型复用 - 3分钟版本规模")
    print("=" * 60)
    print("训练规模配置:")
    print("  ✅ 每路径: 5轮重复训练")
    print("  ✅ 每轮: 4个批次")
    print("  ✅ 每批次: 50个样本")
    print("  ✅ 每样本: 3步探索")
    print("  ✅ 样本生成: 2000候选 → 200最终")
    print("  ✅ 模型保存: 只保存模型参数（优化版本）")
    print("  ✅ 分组方式: 随机分成两个组")
    print(f"  ✅ 默认组规模: 随机组1={default_group1_size}条路径，随机组2={default_group2_size}条路径")
    print(f"  ✅ 键盘输入组规模: {'开启' if USE_KEYBOARD_INPUT_GROUP_SIZE else '关闭'}")
    print(f"  ✅ 随机种子: {RANDOM_GROUP_SEED if RANDOM_GROUP_SEED is not None else 'None，每次运行随机不同'}")
    print("=" * 60)
    print(f"\n自动分组结果:")
    print(f"相似路径组: {similar_group_display}")
    print(f"孤岛路径组: {isolated_group_display}")
    print("\n" + "=" * 60)

    all_runs_data = []
    total_start_time = time.time()

    for run_id in range(1, 21):
        print(f"\n{'=' * 60}")
        print(f"开始第 {run_id}/20 次运行")
        print(f"{'=' * 60}")

        group1_agent, group2_agent, group1_buffer, group2_buffer, total_cumulative_reward, path_rewards, training_time = \
            generate_and_train_grouped_paths_staged(path_documents, similar_group, isolated_group, batch_size=32,
                                                    run_id=run_id)

        # === 优化保存：只保存模型参数，不保存优化器状态等 ===
        group1_model_path = os.path.join(model_path_base, f"random_group1_model_run_{run_id}.pth")
        group2_model_path = os.path.join(model_path_base, f"random_group2_model_run_{run_id}.pth")

        # 只保存模型状态字典，大幅减少文件大小和保存时间
        torch.save(group1_agent.model.state_dict(), group1_model_path)
        torch.save(group2_agent.model.state_dict(), group2_model_path)

        print(f"[第{run_id}次] 模型已保存（优化版本 - 只保存参数）")

        # 重置已抽取索引
        group1_buffer.reset_sampled_indices()
        group2_buffer.reset_sampled_indices()

        run_data = {
            'run_id': run_id,
            'training_time': training_time,
            'total_reward': total_cumulative_reward,
            'path_rewards': path_rewards,
            'path_similarities': {},
            'path_samples': {}
        }

        all_similarities = []
        for path_idx in range(len(target_paths)):
            target_path = target_paths[path_idx]
            path_id = path_idx + 1

            if path_id in similar_group_display:
                buffer = group1_buffer
            elif path_id in isolated_group_display:
                buffer = group2_buffer
            else:
                continue

            high_reward_samples = buffer.get_high_reward_samples(target_path, num_samples=20)

            if high_reward_samples:
                similarities = [sim for _, _, sim, _ in high_reward_samples]
                run_data['path_similarities'][path_idx + 1] = {
                    'avg_similarity': np.mean(similarities),
                    'max_similarity': np.max(similarities),
                    'min_similarity': np.min(similarities),
                    'sample_count': len(similarities)
                }
                run_data['path_samples'][path_idx + 1] = high_reward_samples
                all_similarities.extend(similarities)
            else:
                run_data['path_similarities'][path_idx + 1] = {
                    'avg_similarity': 0.0,
                    'max_similarity': 0.0,
                    'min_similarity': 0.0,
                    'sample_count': 0
                }
                run_data['path_samples'][path_idx + 1] = []

        if all_similarities:
            run_data['overall_avg_similarity'] = np.mean(all_similarities)
            run_data['max_similarity'] = np.max(all_similarities)
            run_data['min_similarity'] = np.min(all_similarities)
        else:
            run_data['overall_avg_similarity'] = 0.0
            run_data['max_similarity'] = 0.0
            run_data['min_similarity'] = 0.0

        all_runs_data.append(run_data)

        print(f"[第{run_id}次] 完成! 总体平均相似度: {run_data['overall_avg_similarity']:.4f}")
        print(f"{'=' * 60}\n")

    total_time = time.time() - total_start_time

    print("\n正在生成综合Excel报告...")
    create_consolidated_excel_report(all_runs_data, similar_group, isolated_group, output_dir)

    print("\n" + "=" * 60)
    print("20次训练全部完成! - 随机分组 + 模型复用 - 3分钟版本规模")
    print("=" * 60)
    print(f"训练规模总结:")
    print(f"  每路径: 5轮 × 4批次 × 50样本 × 3步 = 3000步/路径")
    print(f"  样本生成: 2000候选 → 200最终")
    print(f"  模型保存: 只保存模型参数（优化版本）")
    print(f"  总耗时: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)")
    print(f"  平均每次耗时: {total_time / 20:.2f}秒")
    print(f"\n平均相似度统计:")
    avg_similarities = [r['overall_avg_similarity'] for r in all_runs_data]
    print(f"  总体平均: {np.mean(avg_similarities):.4f}")
    print(f"  最高: {np.max(avg_similarities):.4f}")
    print(f"  最低: {np.min(avg_similarities):.4f}")
    print(f"  标准差: {np.std(avg_similarities):.4f}")
    print(f"\n所有结果已保存到: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_20_times_training()