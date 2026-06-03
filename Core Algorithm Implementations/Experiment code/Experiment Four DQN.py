
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from datetime import datetime
import os

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 统一实验配置 ===
EXPERIMENT_CONFIG = {
    'STATE_DIM': 3,
    'MIN_VALUE': 1,
    'MAX_VALUE': 128,  # 修改为50
    'SAMPLES_PER_PATH': 200,
    'BATCH_SIZE_SAMPLES': 50,
    'STEPS_PER_SAMPLE': 5,
    'NUM_ROUNDS': 5,
    'REPLAY_BATCH_SIZE': 64,
    'SIMILARITY_WEIGHT': 10.0,
    'COVERAGE_BONUS': 5.0,
    'TRIGGER_BONUS': 1.0,
    'HIDDEN_DIM': 256,
    'LEARNING_RATE': 3e-4,
    'NUM_RUNS': 20,  # 修改为20次运行
    'TOP_K_SAMPLES': 20,
    'REPLAY_BUFFER_CAPACITY': 20000,  # 单个路径的经验池容量
    'TARGET_PATHS': [
        {1, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 35},
        {3, 6, 7, 8, 11, 12, 13, 14, 15, 17, 25, 26, 29, 30, 31, 33, 35},
        {1, 2, 6, 9, 10, 11, 12, 14, 15, 25, 26, 27, 30, 31, 33, 34, 36, 37, 39},
        {30, 1, 2, 4, 5, 33, 7, 8, 35, 16, 17, 38, 39, 26, 29},
        {3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 25, 26, 27, 28, 32, 33, 35},
        {1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 18, 25, 26, 27, 28, 30, 32, 33, 34},
        {1, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 29, 30, 32, 33, 35},
        {3, 6, 7, 8, 11, 12, 13, 15, 17, 25, 27, 28, 31, 32, 33, 35},
        {3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 27, 28, 30, 31, 33, 35},
        {1, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 18, 27, 30, 33, 35},
        {30, 31, 32, 3, 4, 5, 33, 7, 8, 35, 16, 17, 26, 27, 28},
        {1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 18, 25, 27, 28, 30, 31, 33, 35},
        {3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 25, 28, 30, 31, 33, 35},
        {1, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 25, 26, 27, 28, 30, 31, 32, 33, 34},
        {30, 31, 32, 3, 6, 7, 8, 33, 35, 11, 12, 14, 15, 27, 28}
    ],
}


# === 工具函数 ===
def clip_state(state):
    return np.clip(state, EXPERIMENT_CONFIG['MIN_VALUE'], EXPERIMENT_CONFIG['MAX_VALUE'])


def denormalize_state(normalized_state):
    """将标准化状态转换回原始状态"""
    min_val = EXPERIMENT_CONFIG['MIN_VALUE']
    max_val = EXPERIMENT_CONFIG['MAX_VALUE']
    return normalized_state * (max_val - min_val) / 2 + (min_val + max_val) / 2


def coverage_similarity(triggered, target_path):
    """
    新的相似度计算方式:交集 / 目标路径长度
    """
    if len(target_path) == 0:
        return 1.0 if len(triggered) == 0 else 0.0

    intersection = target_path.intersection(triggered)
    return len(intersection) / len(target_path)


def unified_reward_function(triggered, target_path):
    config = EXPERIMENT_CONFIG
    similarity = coverage_similarity(triggered, target_path)
    reward = similarity * config['SIMILARITY_WEIGHT']

    if target_path.issubset(triggered):
        reward += config['COVERAGE_BONUS']

    if len(triggered) > 0:
        reward += config['TRIGGER_BONUS']

    return reward


# === 执行规则函数  ===
def execute_Tr(a):
    x, y, z = int(a[0]), int(a[1]), int(a[2])
    triggered = set()

    # Rule Group 1: (x > y) related
    if (x > y) != (x > 5):
        triggered.add(1)
    if (x > y) != (x * x > y):
        triggered.add(2)
    if (x > y) != (x > y * y):
        triggered.add(3)

        # Rule Group 2: (x > z) related
    if (x > z) != (x > 10):
        triggered.add(4)
    if (x > z) != (x * x > z):
        triggered.add(5)
    if (x > z) != (x > z * z):
        triggered.add(6)

        # Rule Group 3: (y > z) related
    if (y > z) != (y > 8):
        triggered.add(7)
    if (y > z) != (y * y > z):
        triggered.add(8)
    if (y > z) != (y > z * z):
        triggered.add(9)
    if (y > z) != (10 > z):
        triggered.add(10)

        # Rule Group 4: (x + y <= z) related
    if (x + y <= z) != (x + y <= z * x):
        triggered.add(11)
    if (x + y <= z) != (x + y <= z * y):
        triggered.add(12)
    if (x + y <= z) != (x * y <= z * z):
        triggered.add(13)
    if (x + y <= z) != (x - y <= z):
        triggered.add(14)

        # 修正后的规则 15：安全处理除以零
    cond_xy_le_z = (x + y <= z)
    cond_x_div_y_le_z = False
    if y != 0:
        cond_x_div_y_le_z = (x / y <= z)

    if cond_xy_le_z != cond_x_div_y_le_z:
        triggered.add(15)

    if (x + y <= z) != (x + y <= 15):
        triggered.add(16)
    if (x + y <= z) != (x + y <= 20):
        triggered.add(17)
    if (x + y <= z) != (x + 5 <= z):
        triggered.add(18)
    if (x + y <= z) != (10 + y <= z):
        triggered.add(19)
    if (x + y <= z) != (x + 8 <= z):
        triggered.add(20)

        # Rule Group 5: (x == y == z) related
    if (x == y == z) != (x <= y == z):
        triggered.add(21)
    if (x == y == z) != (x == y != z):
        triggered.add(22)
    if (x == y == z) != (x != y == z):
        triggered.add(23)

    if (x == y == z) != (x == y <= z):
        triggered.add(24)

        # Rule Group 6: Modulo operations
    if ((x % 2 + y % 2 + z % 2) >= 2) != ((x % 3 + y % 2 + z % 2) >= 2):
        triggered.add(25)
    if ((x % 2 + y % 2 + z % 2) >= 2) != ((x % 2 + y % 3 + z % 2) >= 2):
        triggered.add(26)
    if ((x % 2 + y % 2 + z % 2) >= 2) != ((x % 2 + y % 2 + z % 3) >= 2):
        triggered.add(27)
    if ((x % 2 + y % 2 + z % 2) >= 2) != ((x % 2 + y % 2 + z % 2) >= 1):
        triggered.add(28)
    if ((x % 2 + y % 2 + z % 2) >= 2) != ((x % 2 + y % 2 + z % 2) >= 3):
        triggered.add(29)
    if ((x % 2 + y % 2 + z % 2) >= 2) != ((x % 2 + y % 5 + z % 2) >= 2):
        triggered.add(30)
    if ((x % 2 + y % 2 + z % 2) >= 2) != ((x % 5 + y % 2 + z % 2) >= 2):
        triggered.add(31)
    if ((x % 2 + y % 2 + z % 2) >= 2) != ((x % 2 + y % 2 + z % 5) >= 2):
        triggered.add(32)

        # Rule Group 7: Quadratic equation discriminant like conditions
    cond_main_part = (x != 0 and (y * y - 4 * x * z == 0))

    if cond_main_part != (x != 0 and (y * y - 4 * x * z != 0)):
        triggered.add(33)
    if cond_main_part != (x != 0 and (y * y - 4 * x * z >= 0)):
        triggered.add(34)
    if cond_main_part != (x != 0 and (y * y - 4 * x * z <= 0)):
        triggered.add(35)

    # Rule Group 8: System of equations like conditions
    cond_eq_main_part = (x + y == z and y + z == 2 * x)

    if cond_eq_main_part != (x + y != z and y + z == 2 * x):
        triggered.add(36)

    if cond_eq_main_part != (x + y >= z and y + z == 2 * x):
        triggered.add(37)

    if cond_eq_main_part != (x + y == z and y + z != 2 * x):
        triggered.add(38)
    if cond_eq_main_part != (x + y == z or y + z == 2 * x):
        triggered.add(39)

    return triggered


# 修改函数名以匹配调用
execute_Tr = execute_Tr

# === DQN网络 ===
class DQNNetwork(nn.Module):
    def __init__(self, action_size=30):
        super(DQNNetwork, self).__init__()
        hidden_dim = EXPERIMENT_CONFIG['HIDDEN_DIM']

        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output(x)


# === 单个路径的经验回放缓冲区 ===
class PathReplayBuffer:
    """单个路径专用的经验回放池"""

    def __init__(self, path_idx, capacity=20000):
        self.path_idx = path_idx
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.similarities = deque(maxlen=capacity)  # 只存储相似度

    def push(self, state, action, reward, next_state, done, similarity):
        self.buffer.append((state, action, reward, next_state, done))
        self.similarities.append(similarity)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        return (
            torch.FloatTensor(state).to(device),
            torch.LongTensor(action).to(device),
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.BoolTensor(done).to(device)
        )

    def get_top_k(self, k=20):
        """获取当前路径的Top-K样本"""
        if len(self.buffer) == 0:
            return []

        # 将buffer和similarities组合并排序
        samples_with_sim = list(zip(self.buffer, self.similarities))
        samples_with_sim.sort(key=lambda x: x[1], reverse=True)

        # 获取Top-K
        top_k = samples_with_sim[:k]

        results = []
        for (state, _, _, _, _), similarity in top_k:
            original_state = denormalize_state(state)
            original_state_int = np.round(original_state).astype(int)

            results.append({
                'state': original_state_int,
                'similarity': similarity,
                'triggered': execute_Tr(original_state_int)
            })

        return results

    def __len__(self):
        return len(self.buffer)


# === DQN智能体(修改版:使用多个独立经验池)===
class ImprovedDQNAgent:
    def __init__(self, num_paths, action_size=30):
        self.action_size = action_size
        self.num_paths = num_paths
        self.epsilon = 0.9
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        self.q_network = DQNNetwork(action_size).to(device)
        self.target_network = DQNNetwork(action_size).to(device)

        lr = EXPERIMENT_CONFIG['LEARNING_RATE']
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 为每条路径创建独立的经验回放池
        capacity = EXPERIMENT_CONFIG['REPLAY_BUFFER_CAPACITY']
        self.replay_buffers = {}
        for path_idx in range(num_paths):
            self.replay_buffers[path_idx] = PathReplayBuffer(path_idx, capacity)

        self.replay_train_count = 0
        self.update_target_network()

    def discrete_to_action_delta(self, action_idx):
        # 减小动作幅度以适应新的取值范围
        delta_values = [5, 3, 2, 1, 0.5, -0.5, -1, -2, -3, -5]

        if action_idx >= 30:
            action_idx = action_idx % 30

        dim = action_idx // 10
        delta_idx = action_idx % 10
        delta = delta_values[delta_idx]

        action_delta = np.zeros(3)
        action_delta[dim] = delta

        return action_delta

    def get_action(self, state):
        min_val = EXPERIMENT_CONFIG['MIN_VALUE']
        max_val = EXPERIMENT_CONFIG['MAX_VALUE']
        normalized_state = (state - (min_val + max_val) / 2) / ((max_val - min_val) / 2)

        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()

        action_delta = self.discrete_to_action_delta(action_idx)
        return action_delta, action_idx

    def store_experience(self, path_idx, state, action_idx, reward, next_state, done, similarity):
        """存储经验到对应路径的经验池"""
        min_val = EXPERIMENT_CONFIG['MIN_VALUE']
        max_val = EXPERIMENT_CONFIG['MAX_VALUE']
        normalized_state = (state - (min_val + max_val) / 2) / ((max_val - min_val) / 2)
        normalized_next_state = (next_state - (min_val + max_val) / 2) / ((max_val - min_val) / 2)

        self.replay_buffers[path_idx].push(
            normalized_state, action_idx, reward,
            normalized_next_state, done, similarity
        )

    def replay_train(self, path_idx):
        """从指定路径的经验池中训练"""
        batch_size = EXPERIMENT_CONFIG['REPLAY_BATCH_SIZE']
        batch = self.replay_buffers[path_idx].sample(batch_size)

        if batch is None:
            return

        states, actions, rewards, next_states, dones = batch

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * ~dones)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.replay_train_count += 1

        if self.replay_train_count % 2 == 0:
            self.update_target_network()
            print(f"    -> 目标网络已更新 (第{self.replay_train_count}次回放训练)")

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_all_top_k(self, k=20):
        """获取所有路径的Top-K样本"""
        results = {}
        for path_idx in range(self.num_paths):
            results[path_idx] = self.replay_buffers[path_idx].get_top_k(k)
        return results

    def get_buffer_stats(self):
        """获取所有经验池的统计信息"""
        stats = {}
        for path_idx in range(self.num_paths):
            stats[path_idx] = len(self.replay_buffers[path_idx])
        return stats


# === 核心性能指标统计函数 ===
def calculate_run_performance(run_idx, dqn_results, training_time, total_steps, update_count, agent):
    """计算单次运行的全面性能指标"""
    target_paths = EXPERIMENT_CONFIG['TARGET_PATHS']
    num_paths = len(target_paths)

    # 1. 总奖励（Total Reward）
    total_reward = 0
    # 2. 平均奖励（Average Reward）
    average_reward = 0
    # 5. 收敛性（Convergence）
    convergence = 0
    # 12. 环境适应性（Environment Adaptability）
    environment_adaptability = 0
    # 13. 策略的泛化能力（Generalization Ability）
    generalization_ability = 0
    # 15. 计算效率（Computational Efficiency）
    computational_efficiency = 0
    # 16. 策略更新频率（Policy Update Frequency）
    policy_update_frequency = 0

    # 样本相似度统计
    all_similarities = []

    # 计算指标
    total_samples = 0
    all_rewards = []

    for path_idx in range(num_paths):
        samples = dqn_results[path_idx]
        for sample in samples:
            triggered = sample['triggered']
            target_path = target_paths[path_idx]
            reward = unified_reward_function(triggered, target_path)
            similarity = sample['similarity']

            total_reward += reward
            all_rewards.append(reward)
            all_similarities.append(similarity)
            total_samples += 1

    # 1. 总奖励
    total_reward = total_reward

    # 2. 平均奖励
    if total_samples > 0:
        average_reward = total_reward / total_samples

    # 5. 收敛性（平均相似度）
    if all_similarities:
        convergence = np.mean(all_similarities)

    # 12. 环境适应性（相似度方差）
    if len(all_similarities) > 1:
        environment_adaptability = 1 / (np.std(all_similarities) + 1e-8)

    # 13. 策略的泛化能力（平均相似度）
    generalization_ability = convergence

    # 15. 计算效率（步数/秒）
    if training_time > 0:
        computational_efficiency = total_steps / training_time

    # 16. 策略更新频率
    if training_time > 0:
        policy_update_frequency = update_count / training_time

    # 样本相似度统计
    avg_similarity = np.mean(all_similarities) if all_similarities else 0
    max_similarity = np.max(all_similarities) if all_similarities else 0
    min_similarity = np.min(all_similarities) if all_similarities else 0

    return {
        '运行编号': run_idx + 1,

        # 保留的核心指标
        '总奖励': round(total_reward, 2),
        '平均奖励': round(average_reward, 4),
        '收敛性': round(convergence, 4),
        '环境适应性': round(environment_adaptability, 4),
        '泛化能力': round(generalization_ability, 4),
        '计算效率': round(computational_efficiency, 2),
        '策略更新频率': round(policy_update_frequency, 4),

        # 样本相似度统计
        '平均相似度': round(avg_similarity, 4),
        '最大相似度': round(max_similarity, 4),
        '最小相似度': round(min_similarity, 4),
    }


# === Excel导出函数 ===
def export_to_excel(all_dqn_results, all_performance_data, target_paths, output_path="DQN测试结果_20次运行.xlsx"):
    """导出20次运行的DQN结果到Excel"""
    print("\n正在生成Excel报告...")

    # 初始化数据列表
    all_dqn_summary_data = []
    all_dqn_detailed_data = []

    # 处理每次运行的数据
    for run_idx, (dqn_results, performance_data) in enumerate(zip(all_dqn_results, all_performance_data)):
        # ===== Sheet1: DQN路径汇总统计 =====
        dqn_summary_data = []
        for path_idx in range(len(target_paths)):
            target_path = target_paths[path_idx]
            samples = dqn_results[path_idx]

            if len(samples) == 0:
                dqn_summary_data.append({
                    '运行编号': run_idx + 1,
                    '路径编号': path_idx + 1,
                    '目标规则数': len(target_path),
                    '样本数量': 0,
                    '平均相似度': 0,
                    '最大相似度': 0,
                    '最小相似度': 0,
                    '相似度标准差': 0,
                    '是否完美匹配': '否',
                    '目标路径': ', '.join(map(str, sorted(target_path)))
                })
                continue

            similarities = [s['similarity'] for s in samples]
            perfect_count = sum(1 for s in similarities if abs(s - 1.0) < 0.001)
            is_perfect = '是' if perfect_count > 0 else '否'

            dqn_summary_data.append({
                '运行编号': run_idx + 1,
                '路径编号': path_idx + 1,
                '目标规则数': len(target_path),
                '样本数量': len(samples),
                '平均相似度': round(np.mean(similarities), 4),
                '最大相似度': round(max(similarities), 4),
                '最小相似度': round(min(similarities), 4),
                '相似度标准差': round(np.std(similarities), 4),
                '是否完美匹配': is_perfect,
                '目标路径': ', '.join(map(str, sorted(target_path)))
            })

        all_dqn_summary_data.extend(dqn_summary_data)

        # ===== Sheet2: DQN详细样本数据 =====
        dqn_detailed_data = []
        for path_idx in range(len(target_paths)):
            target_path = target_paths[path_idx]
            samples = dqn_results[path_idx]

            for sample_idx, sample in enumerate(samples):
                state = sample['state']
                similarity = sample['similarity']
                triggered = sample['triggered']

                dqn_detailed_data.append({
                    '运行编号': run_idx + 1,
                    '路径编号': path_idx + 1,
                    '样本序号': sample_idx + 1,
                    'X值': int(state[0]),
                    'Y值': int(state[1]),
                    'Z值': int(state[2]),
                    '相似度': round(similarity, 4),
                    '是否完美匹配': '是' if abs(similarity - 1.0) < 0.001 else '否',
                    '目标路径': ', '.join(map(str, sorted(target_path))),
                    '触发规则': ', '.join(map(str, sorted(triggered))),
                    '匹配规则数': len(target_path.intersection(triggered)),
                    '目标规则数': len(target_path)
                })

        all_dqn_detailed_data.extend(dqn_detailed_data)

    # 创建Excel文件
    dqn_summary_df = pd.DataFrame(all_dqn_summary_data)
    dqn_detailed_df = pd.DataFrame(all_dqn_detailed_data)
    performance_df = pd.DataFrame(all_performance_data)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet1: DQN路径汇总统计
        dqn_summary_df.to_excel(writer, sheet_name='DQN路径汇总统计', index=False)

        # Sheet2: DQN详细样本数据
        dqn_detailed_df.to_excel(writer, sheet_name='DQN详细样本数据', index=False)

        # Sheet3: 全面性能指标统计 - 只保留指定列
        selected_columns = [
            '运行编号',
            '总奖励', '平均奖励', '收敛性', '环境适应性',
            '泛化能力', '计算效率', '策略更新频率',
            '平均相似度', '最大相似度', '最小相似度'
        ]
        performance_df_selected = performance_df[selected_columns]
        performance_df_selected.to_excel(writer, sheet_name='全面性能指标统计', index=False)

        # 美化样式
        workbook = writer.book

        # 通用样式
        header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        header_font = Font(name='微软雅黑', size=11, bold=True, color='FFFFFF')
        perfect_fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')  # 浅绿色

        # === 设置Sheet1样式 ===
        ws1 = writer.sheets['DQN路径汇总统计']
        for cell in ws1[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')

        # 高亮完美匹配行
        for row_idx in range(2, ws1.max_row + 1):
            if ws1.cell(row_idx, 9).value == '是':  # 第9列是"是否完美匹配"
                for col_idx in range(1, ws1.max_column + 1):
                    ws1.cell(row_idx, col_idx).fill = perfect_fill

        ws1.column_dimensions['A'].width = 12
        ws1.column_dimensions['B'].width = 12
        ws1.column_dimensions['C'].width = 12
        ws1.column_dimensions['D'].width = 12
        ws1.column_dimensions['E'].width = 15
        ws1.column_dimensions['F'].width = 15
        ws1.column_dimensions['G'].width = 15
        ws1.column_dimensions['H'].width = 15
        ws1.column_dimensions['I'].width = 15
        ws1.column_dimensions['J'].width = 50

        # === 设置Sheet2样式 ===
        ws2 = writer.sheets['DQN详细样本数据']
        for cell in ws2[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')

        ws2.column_dimensions['A'].width = 12
        ws2.column_dimensions['B'].width = 12
        ws2.column_dimensions['C'].width = 12
        ws2.column_dimensions['D'].width = 10
        ws2.column_dimensions['E'].width = 10
        ws2.column_dimensions['F'].width = 10
        ws2.column_dimensions['G'].width = 12
        ws2.column_dimensions['H'].width = 15
        ws2.column_dimensions['I'].width = 40
        ws2.column_dimensions['J'].width = 40
        ws2.column_dimensions['K'].width = 15
        ws2.column_dimensions['L'].width = 15

        # === 设置Sheet3样式 ===
        ws3 = writer.sheets['全面性能指标统计']
        for cell in ws3[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')

        # 设置列宽
        columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        for col in columns:
            ws3.column_dimensions[col].width = 18

    print(f"Excel报告已保存到: {output_path}")
    print(f"  - Sheet1: DQN路径汇总统计 ({len(all_dqn_summary_data)}行)")
    print(f"  - Sheet2: DQN详细样本数据 ({len(all_dqn_detailed_data)}行)")
    print(f"  - Sheet3: 全面性能指标统计 ({len(all_performance_data)}行)")


# === DQN训练流程(修改版:使用独立经验池)===
def train_dqn_workflow():
    print("=" * 80)
    print("开始DQN训练流程 (独立经验池版)")
    print("相似度计算方式: 交集 / 目标路径长度")
    print(
        f"训练策略: 每条路径重复{EXPERIMENT_CONFIG['NUM_ROUNDS']}轮,每个样本走{EXPERIMENT_CONFIG['STEPS_PER_SAMPLE']}步")
    print(f"经验池策略: 每条路径独立经验池,容量={EXPERIMENT_CONFIG['REPLAY_BUFFER_CAPACITY']}")
    print("=" * 80)

    target_paths = EXPERIMENT_CONFIG['TARGET_PATHS']
    num_paths = len(target_paths)

    # 创建智能体(传入路径数量)
    agent = ImprovedDQNAgent(num_paths=num_paths)

    start_time = time.time()
    total_steps = 0

    # 生成样本
    print(f"\n生成样本: 每条路径{EXPERIMENT_CONFIG['SAMPLES_PER_PATH']}个")
    path_samples = {}
    for path_idx in range(num_paths):
        samples = []
        for _ in range(EXPERIMENT_CONFIG['SAMPLES_PER_PATH']):
            state = np.random.randint(
                EXPERIMENT_CONFIG['MIN_VALUE'],
                EXPERIMENT_CONFIG['MAX_VALUE'] + 1,
                EXPERIMENT_CONFIG['STATE_DIM']
            ).astype(np.float32)
            samples.append(state)
        path_samples[path_idx] = samples
        print(f"  路径 {path_idx + 1}/{num_paths}: 生成 {len(samples)} 个样本")

    # 批次训练配置
    batch_size = EXPERIMENT_CONFIG['BATCH_SIZE_SAMPLES']
    num_batches = EXPERIMENT_CONFIG['SAMPLES_PER_PATH'] // batch_size
    num_rounds = EXPERIMENT_CONFIG['NUM_ROUNDS']

    print(f"\n训练配置:")
    print(f"  - 每批样本数: {batch_size}")
    print(f"  - 每条路径批次数: {num_batches}")
    print(f"  - 每条路径重复轮数: {num_rounds}")
    print(f"  - 每个样本步数: {EXPERIMENT_CONFIG['STEPS_PER_SAMPLE']}")
    print(
        f"  - 总训练批次: {num_paths} 路径 × {num_rounds} 轮 × {num_batches} 批 = {num_paths * num_rounds * num_batches} 批")
    print("-" * 80)

    # 训练流程:先完成一条路径的所有轮次,再进行下一条路径
    for path_idx in range(num_paths):
        target_path = target_paths[path_idx]
        print(f"\n{'=' * 80}")
        print(f"开始训练路径 {path_idx + 1}/{num_paths}")
        print(f"目标规则: {sorted(target_path)}")
        print(f"使用独立经验池: replay_buffers[{path_idx}]")
        print(f"{'=' * 80}")

        # 对当前路径进行NUM_ROUNDS轮训练
        for round_idx in range(num_rounds):
            print(f"\n{'─' * 80}")
            print(f"路径 {path_idx + 1} - 第 {round_idx + 1}/{num_rounds} 轮")
            print(f"{'─' * 80}")

            # 每轮包含num_batches个批次
            for batch_idx in range(num_batches):
                print(f"\n  批次 {batch_idx + 1}/{num_batches} (路径{path_idx + 1}, 第{round_idx + 1}轮)")

                # 获取当前批次的样本
                batch_samples = path_samples[path_idx][batch_idx * batch_size:(batch_idx + 1) * batch_size]

                batch_rewards = []
                batch_similarities = []

                # 处理当前批次的每个样本
                for sample_idx, initial_state in enumerate(batch_samples):
                    state = initial_state.copy()
                    episode_reward = 0
                    final_similarity = 0

                    # 每个样本执行STEPS_PER_SAMPLE步
                    for step in range(EXPERIMENT_CONFIG['STEPS_PER_SAMPLE']):
                        action_delta, action_idx = agent.get_action(state)

                        next_state = state + action_delta
                        next_state = clip_state(next_state)

                        triggered = execute_Tr(next_state)  # 现在只需要传递一个参数
                        reward = unified_reward_function(triggered, target_path)
                        similarity = coverage_similarity(triggered, target_path)

                        done = (step == EXPERIMENT_CONFIG['STEPS_PER_SAMPLE'] - 1)

                        # 存储到对应路径的经验池
                        agent.store_experience(
                            path_idx, state, action_idx, reward, next_state, done, similarity
                        )

                        state = next_state
                        episode_reward += reward
                        final_similarity = similarity
                        total_steps += 1

                    batch_rewards.append(episode_reward)
                    batch_similarities.append(final_similarity)

                # 批次统计
                avg_reward = np.mean(batch_rewards)
                avg_similarity = np.mean(batch_similarities)
                max_similarity = np.max(batch_similarities)
                print(f"    平均奖励={avg_reward:.2f}, 平均相似度={avg_similarity:.4f}, "
                      f"最大相似度={max_similarity:.4f}, epsilon={agent.epsilon:.3f}")

                # 每个批次结束后从当前路径的经验池训练
                print(f"    执行回放训练(从路径{path_idx}的经验池)...")
                agent.replay_train(path_idx)

                # 显示当前路径经验池大小
                buffer_size = len(agent.replay_buffers[path_idx])
                print(f"    路径{path_idx}经验池大小: {buffer_size}, 总回放训练次数: {agent.replay_train_count}")

    training_time = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"DQN训练完成! 总耗时: {training_time:.2f}秒, 总步数: {total_steps}")
    print(f"总回放训练次数: {agent.replay_train_count}")
    print(f"目标网络更新次数: {agent.replay_train_count // 2}")

    # 显示所有路径的经验池统计
    print("\n各路径经验池大小:")
    buffer_stats = agent.get_buffer_stats()
    for path_idx, size in buffer_stats.items():
        print(f"  路径{path_idx + 1}: {size} 条经验")

    print("=" * 80)

    # 获取Top-K样本
    print(f"\n从各路径经验池中挑选相似度最高的{EXPERIMENT_CONFIG['TOP_K_SAMPLES']}个样本...")
    dqn_top_k_results = agent.get_all_top_k(EXPERIMENT_CONFIG['TOP_K_SAMPLES'])

    return agent, dqn_top_k_results, training_time, total_steps, agent.replay_train_count


# === 主流程 ===
def main():
    print("\n" + "=" * 80)
    print("DQN算法测试 - 20次运行版本")
    print("全面性能指标评估")
    print("=" * 80)

    all_dqn_results = []
    all_performance_data = []
    target_paths = EXPERIMENT_CONFIG['TARGET_PATHS']

    # 运行20次实验
    for run_idx in range(EXPERIMENT_CONFIG['NUM_RUNS']):
        print(f"\n{'='*80}")
        print(f"开始第 {run_idx + 1}/{EXPERIMENT_CONFIG['NUM_RUNS']} 次运行")
        print(f"{'='*80}")

        # DQN训练
        dqn_agent, dqn_results, training_time, total_steps, update_count = train_dqn_workflow()

        # 计算性能指标
        performance_data = calculate_run_performance(
            run_idx, dqn_results, training_time, total_steps, update_count, dqn_agent
        )

        # 保存结果
        all_dqn_results.append(dqn_results)
        all_performance_data.append(performance_data)

        print(f"\n第 {run_idx + 1} 次运行完成!")
        print(f"  总奖励: {performance_data['总奖励']}")
        print(f"  平均奖励: {performance_data['平均奖励']}")
        print(f"  收敛性: {performance_data['收敛性']}")

    # 导出Excel结果（整合20次运行数据）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"DQN测试结果_20次运行_{timestamp}.xlsx"
    export_to_excel(all_dqn_results, all_performance_data, target_paths, output_path)

    # 打印总体统计摘要
    print("\n" + "=" * 80)
    print("20次运行总体统计摘要")
    print("=" * 80)

    # 计算关键指标统计
    total_rewards = [p['总奖励'] for p in all_performance_data]
    average_rewards = [p['平均奖励'] for p in all_performance_data]
    convergences = [p['收敛性'] for p in all_performance_data]
    environment_adaptabilities = [p['环境适应性'] for p in all_performance_data]
    generalization_abilities = [p['泛化能力'] for p in all_performance_data]
    computational_efficiencies = [p['计算效率'] for p in all_performance_data]
    policy_update_frequencies = [p['策略更新频率'] for p in all_performance_data]
    avg_similarities = [p['平均相似度'] for p in all_performance_data]

    print(f"总奖励统计:")
    print(f"  平均值: {np.mean(total_rewards):.2f}")
    print(f"  标准差: {np.std(total_rewards):.2f}")

    print(f"\n平均奖励统计:")
    print(f"  平均值: {np.mean(average_rewards):.4f}")
    print(f"  标准差: {np.std(average_rewards):.4f}")

    print(f"\n收敛性统计:")
    print(f"  平均值: {np.mean(convergences):.4f}")
    print(f"  标准差: {np.std(convergences):.4f}")

    print(f"\n环境适应性统计:")
    print(f"  平均值: {np.mean(environment_adaptabilities):.4f}")
    print(f"  标准差: {np.std(environment_adaptabilities):.4f}")

    print(f"\n泛化能力统计:")
    print(f"  平均值: {np.mean(generalization_abilities):.4f}")
    print(f"  标准差: {np.std(generalization_abilities):.4f}")

    print(f"\n计算效率统计:")
    print(f"  平均值: {np.mean(computational_efficiencies):.2f}")
    print(f"  标准差: {np.std(computational_efficiencies):.2f}")

    print(f"\n策略更新频率统计:")
    print(f"  平均值: {np.mean(policy_update_frequencies):.4f}")
    print(f"  标准差: {np.std(policy_update_frequencies):.4f}")

    print(f"\n平均相似度统计:")
    print(f"  平均值: {np.mean(avg_similarities):.4f}")
    print(f"  标准差: {np.std(avg_similarities):.4f}")

    print("\n" + "=" * 80)
    print(f"所有 {EXPERIMENT_CONFIG['NUM_RUNS']} 次优化流程完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()