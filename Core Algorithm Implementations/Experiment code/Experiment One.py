import random
from typing import Set, List, Tuple, Dict
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import os
from collections import deque
import torch.optim as optim
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def execute_orchestration_rules(a):
    dx, dy, dz = a
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


targetPaths_sets = [set(path) for path in [
        {1, 2, 4, 11, 12, 13, 14, 15},
        {5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29},
        {5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
    ]]


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

def compute_reward(state, target_path, triggered, prev_triggered=None, prev_state=None):
    sim = jaccard_similarity(triggered, target_path)
    reward = sim * 10
    if target_path.issubset(triggered):
        reward += 1
    return reward

class SharedExperienceReplay:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.priorities = deque(maxlen=self.capacity)

    def append(self, experience):
        self.buffer.append(experience)
        self.priorities.append(experience[-1])

    def sample(self, batch_size, alpha=0.6):
        priorities = np.array(self.priorities) ** alpha
        probabilities = priorities / np.sum(priorities)
        batch_indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in batch_indices]
        return batch, batch_indices, probabilities[batch_indices]

    def __len__(self):
        return len(self.buffer)

    def get_high_reward_samples(self, target_path, num_samples=20):
        if len(self.buffer) == 0:
            return []

        samples_with_recalculated_scores = []
        for experience in self.buffer:
            state_tensor = experience[0]
            state_tuple = tuple(state_tensor.cpu().numpy().flatten().astype(int))
            triggered = execute_orchestration_rules(state_tuple)
            new_reward = compute_reward(state_tuple, target_path, triggered, None, None)
            sim = jaccard_similarity(triggered, target_path)
            samples_with_recalculated_scores.append((state_tuple, new_reward, sim, triggered))

        samples_with_recalculated_scores.sort(key=lambda x: x[1], reverse=True)
        return samples_with_recalculated_scores[:num_samples]

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
        delta_values = [10, 5, 3, 2, 1, -1, -2, -3, -5, -10]
        dim = action_idx // 10
        delta_idx = action_idx % 10
        delta = delta_values[delta_idx]
        if dim == 0:
            return (delta, 0, 0)
        elif dim == 1:
            return (0, delta, 0)
        elif dim == 2:
            return (0, 0, delta)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values, dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

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

        states = torch.tensor(np.array([s.cpu().numpy().flatten() for s in states]), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array([ns.cpu().numpy().flatten() for ns in next_states]),
                                   dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_max_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_max_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_q_value(self, state):
        """获取状态的Q值"""
        if isinstance(state, (list, tuple)):
            state = np.array(state, dtype=np.float32)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.max().item()

def generate_samples_for_similar_paths(agent, similar_group, episodes=100):


    for episode in tqdm(range(episodes)):
        for path_idx in similar_group:
            target_path = targetPaths_sets[path_idx]
            state = (random.randint(2, 100), random.randint(2, 100), random.randint(2, 100))

            for step in range(5):
                legal_actions = []
                for a in range(agent.action_dim):
                    dx, dy, dz = agent.decode_action(a)
                    cand_next = (state[0] + dx, state[1] + dy, state[2] + dz)
                    if all(2 <= x <= 100 for x in cand_next):
                        legal_actions.append(a)

                if not legal_actions:
                    break

                if random.random() < agent.epsilon:
                    action = random.choice(legal_actions)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = agent.model(state_tensor)[0]
                    action = legal_actions[torch.argmax(q_values[legal_actions]).item()]
                dx, dy, dz = agent.decode_action(action)
                next_state = (state[0] + dx, state[1] + dy, state[2] + dz)

                next_state = (
                    max(2, min(100, next_state[0])),
                    max(2, min(100, next_state[1])),
                    max(2, min(100, next_state[2]))
                )

                triggered = execute_orchestration_rules(next_state)
                reward = compute_reward(next_state, target_path, triggered)
                done = (step == 4)

                agent.store_transition(state, action, reward, next_state, done)
                state = next_state

                if len(agent.replay_buffer) >= 32:
                    agent.train(32)

        if episode % 50 == 0:
            agent.update_target_model()



class QValueNormalizer:
    def __init__(self):
        self.min_q = None
        self.max_q = None
        self.q_values = []

    def collect_q_values(self, agent, sample_states):
        for state in sample_states:
            q_val = agent.get_q_value(state)
            self.q_values.append(q_val)

        self.min_q = min(self.q_values)
        self.max_q = max(self.q_values)
        print(f"Q值范围: [{self.min_q:.4f}, {self.max_q:.4f}]")

    def normalize_q_value(self, q_value):
        if self.min_q is None or self.max_q is None:
            return 0.5

        if self.max_q == self.min_q:
            return 0.5

        normalized = (q_value - self.min_q) / (self.max_q - self.min_q)
        return max(0.0, min(1.0, normalized))


# === 评分函数 ===
def length_score(path: Set[int], target: Set[int]) -> float:
    return 1 - abs(len(path) - len(target)) / max(len(path), len(target), 1)


def compute_robustness(state: Tuple[int, int, int], path_map: Dict[Tuple[int, int, int], Set[int]]) -> float:
    x, y, z = state
    center_path = path_map[state]
    neighbors = get_6_neighbors(x, y, z)
    similarities = []

    for nb in neighbors:
        if nb in path_map:
            sim = jaccard_similarity(center_path, path_map[nb])
            similarities.append(sim)

    return sum(similarities) / len(similarities) if similarities else 0.0

def get_6_neighbors(x: int, y: int, z: int) -> List[Tuple[int, int, int]]:
    neighbors = []
    for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        new_x = max(2, min(100, x + dx))
        new_y = max(2, min(100, y + dy))
        new_z = max(2, min(100, z + dz))
        neighbors.append((new_x, new_y, new_z))
    return neighbors

def compute_four_criteria_score(state: Tuple[int, int, int], target_path: Set[int],
                                path_map: Dict[Tuple[int, int, int], Set[int]],
                                agent: DQNAgentWithPER,
                                q_normalizer: QValueNormalizer,
                                weights: List[float] = [0.33, 0.33, 0.33, 0.0]) -> float:

    triggered = path_map[state]
    similarity = jaccard_similarity(triggered, target_path)
    length_diff = length_score(triggered, target_path)
    robustness = compute_robustness(state, path_map)
    raw_q_value = agent.get_q_value(state)
    normalized_q_value = q_normalizer.normalize_q_value(raw_q_value)
    isolation_score = 1 - normalized_q_value
    final_score = (weights[0] * similarity +
                   weights[1] * length_diff +
                   weights[2] * robustness +
                   weights[3] * isolation_score)

    return final_score

def generate_random_state() -> Tuple[int, int, int]:
    return (
        random.randint(2, 100),
        random.randint(2, 100),
        random.randint(2, 100)
    )

def run_isolated_path_scoring(samples_per_path: int = 500):  # 减少到500个样本
    print("=" * 60)
    print("孤岛路径四标准评分系统（优化版-减少运行时间）")
    print("=" * 60)

    similar_group, isolated_group = group_paths_by_similarity(targetPaths_sets)
    print(f"相似路径组: {similar_group}")
    print(f"孤岛路径组: {isolated_group}")

    print("\n步骤2: 为相似路径训练DQN模型")
    replay_buffer = SharedExperienceReplay(capacity=5000)  # 减少经验池容量
    state_dim = 3
    action_dim = 30
    agent = DQNAgentWithPER(state_dim, action_dim, replay_buffer)

    generate_samples_for_similar_paths(agent, similar_group, episodes=100)  # 减少到100 episodes

    print("\n步骤3: 初始化Q值归一化器")
    q_normalizer = QValueNormalizer()
    sample_states = [generate_random_state() for _ in range(300)]  # 减少到300个样本
    q_normalizer.collect_q_values(agent, sample_states)

    print("\n步骤4: 孤岛路径四标准评分")
    isolated_path_scores = {}

    for path_idx in isolated_group:
        target_path = targetPaths_sets[path_idx]
        print(f"\n处理孤岛路径 {path_idx + 1}: {sorted(target_path)}")

        coords = []
        path_map = {}
        scores = []

        print(f"生成 {samples_per_path} 个测试样本...")
        for _ in tqdm(range(samples_per_path)):
            state = generate_random_state()
            coords.append(state)
            path_map[state] = execute_orchestration_rules(state)

            score = compute_four_criteria_score(
                state, target_path, path_map, agent, q_normalizer
            )
            scores.append(score)

        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)

        isolated_path_scores[path_idx] = {
            'path_id': path_idx + 1,
            'target_path': sorted(target_path),
            'avg_score': avg_score,
            'max_score': max_score,
            'min_score': min_score,
            'sample_count': samples_per_path
        }

        print(f"路径 {path_idx + 1} 平均得分: {avg_score:.6f}")
        print(f"得分范围: [{min_score:.6f}, {max_score:.6f}]")

    print("\n" + "=" * 60)
    print("孤岛路径四标准评分汇总结果")
    print("=" * 60)

    all_avg_scores = []
    for path_idx, result in isolated_path_scores.items():
        print(f"路径 {result['path_id']:2d}: 平均得分 = {result['avg_score']:.6f}")
        all_avg_scores.append(result['avg_score'])

    overall_avg = np.mean(all_avg_scores)
    print(f"\n所有孤岛路径的整体平均得分: {overall_avg:.6f}")

    # 6. 保存结果
    results_df = pd.DataFrame([
        {
            'path_id': result['path_id'],
            'target_path': str(result['target_path']),
            'avg_score': result['avg_score'],
            'max_score': result['max_score'],
            'min_score': result['min_score'],
            'sample_count': result['sample_count']
        }
        for result in isolated_path_scores.values()
    ])

    results_df.to_csv("isolated_path_four_criteria_scores.csv", index=False)
    print(f"\n结果已保存至: isolated_path_four_criteria_scores.csv")

    return isolated_path_scores

if __name__ == '__main__':
    print("开始执行孤岛路径四标准评分...")

    # 执行评分
    results = run_isolated_path_scoring(samples_per_path=500)

    print("\n程序执行完成！")