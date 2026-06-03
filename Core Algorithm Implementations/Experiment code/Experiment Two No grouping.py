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

# === 定义各维度的取值范围（新范围）===
WEATHER_MIN = 1    # 天气最小值
WEATHER_MAX = 6    # 天气最大值
TIME_MIN = 1       # 时间段最小值
TIME_MAX = 6       # 时间段最大值
Z_MIN = 1          # 行人数量最小值
Z_MAX = 60         # 行人数量最大值

# === 归一化/反归一化函数 ===
def normalize_state(state):
    """
    将状态归一化到[0, 1]
    Args:
        state: (weather, time_period, z) 原始状态
    Returns:
        归一化后的状态
    """
    weather_norm = (state[0] - WEATHER_MIN) / (WEATHER_MAX - WEATHER_MIN)
    time_norm = (state[1] - TIME_MIN) / (TIME_MAX - TIME_MIN)
    z_norm = (state[2] - Z_MIN) / (Z_MAX - Z_MIN)
    return (weather_norm, time_norm, z_norm)

def denormalize_state(state_norm):
    """
    将状态从[0, 1]反归一化到原始范围
    Args:
        state_norm: (weather_norm, time_norm, z_norm) 归一化状态
    Returns:
        原始状态
    """
    weather = int(round(state_norm[0] * (WEATHER_MAX - WEATHER_MIN) + WEATHER_MIN))
    time_period = int(round(state_norm[1] * (TIME_MAX - TIME_MIN) + TIME_MIN))
    z = int(round(state_norm[2] * (Z_MAX - Z_MIN) + Z_MIN))

    # 确保在有效范围内
    weather = np.clip(weather, WEATHER_MIN, WEATHER_MAX)
    time_period = np.clip(time_period, TIME_MIN, TIME_MAX)
    z = np.clip(z, Z_MIN, Z_MAX)

    return (weather, time_period, z)

def normalize_value(value, min_val, max_val):
    """归一化单个值"""
    return (value - min_val) / (max_val - min_val)

def denormalize_value(value_norm, min_val, max_val):
    """反归一化单个值"""
    return int(round(value_norm * (max_val - min_val) + min_val))

# === 安全除法函数 ===
def safe_divide(numerator, denominator, default=0.0):
    """安全除法，避免除零错误"""
    if denominator == 0:
        return default
    return numerator / denominator

# === 简化的奖励函数 ===
def compute_reward(state, target_path, triggered, prev_triggered=None, prev_state=None):
    """计算奖励值"""
    sim = jaccard_similarity(triggered, target_path)
    reward = sim * 10

    if target_path.issubset(triggered):
        reward += 1

    if prev_triggered is not None:
        prev_sim = jaccard_similarity(prev_triggered, target_path)
        improvement = sim - prev_sim
        reward += improvement * 5

    return reward

def execute_validation_rules_block4(weather, time_period, z):
    """验证规则函数 - weather, time_period, z组合"""
    triggered = set()

    # 使用新的变量映射逻辑
    x = z  # 直接使用z作为x
    y = (weather * time_period * 10 + z) % 100 + 1  # 基于输入参数计算y值

    # 1-7: 早高峰组合（time_period == 1）
    if time_period == 1:
        if x < 60 and y > 75:
            triggered.add(1)
        if x > 60 and y > 70:
            triggered.add(2)
        if x < 50 and y < 40:
            triggered.add(3)
        if x > 78 and 45 < y < 70:
            triggered.add(4)
        if 45 < x < 70 and y > 78:
            triggered.add(5)
        if x < 55 and 50 < y < 75:
            triggered.add(6)
        if 50 < x < 75 and y < 55:
            triggered.add(7)

    # 8-14: 晚高峰组合（time_period == 2）
    if time_period == 2:
        if x < 60 and y > 75:
            triggered.add(8)
        if x > 60 and y > 70:
            triggered.add(9)
        if x < 55 and y < 45:
            triggered.add(10)
        if 45 < x < 70 and y > 78:
            triggered.add(11)
        if x > 78 and 45 < y < 70:
            triggered.add(12)
        if 55 < x < 75 and y < 50:
            triggered.add(13)
        if x < 50 and 55 < y < 75:
            triggered.add(14)

    # 15-19: 午餐时间组合（time_period == 3）
    if time_period == 3:
        if x > 60 and 40 < y < 65:
            triggered.add(15)
        if 40 < x < 65 and y > 60:
            triggered.add(16)
        if 45 < x < 70 and 45 < y < 60:
            triggered.add(17)
        if x < 50 and y < 40:
            triggered.add(18)
        if x > 65 and y < 45:
            triggered.add(19)

    # 20-25: 夜间组合（time_period == 4）
    if time_period == 4:
        if x < 45 and y < 35:
            triggered.add(20)
        if x > 60 and y < 40:
            triggered.add(21)
        if x < 50 and y > 70:
            triggered.add(22)
        if 45 < x < 70 and 45 < y < 60:
            triggered.add(23)
        if x < 35 and y < 25:
            triggered.add(24)
        if 40 < x < 65 and y < 45:
            triggered.add(25)

    # 26-28: 周末组合（time_period == 5）
    if time_period == 5:
        if x < 60 and y < 50:
            triggered.add(26)
        if x > 65 and y > 75:
            triggered.add(27)
        if x > 60 and y < 45:
            triggered.add(28)

    # 29-33: 假日组合（time_period == 6）
    if time_period == 6:
        if 40 < x < 70 and 40 < y < 60:
            triggered.add(29)
        if x < 55 and y < 45:
            triggered.add(30)
        if x > 60 and y < 50:
            triggered.add(31)
        if x < 60 and y > 70:
            triggered.add(32)
        if x > 65 and y > 75:
            triggered.add(33)

    # 34-68: 天气相关扩展规则
    if weather == 1:  # 晴天
        if time_period in [1, 2] and x > 70:
            triggered.add(34)
        if time_period in [1, 2] and y > 70:
            triggered.add(35)
        if time_period in [3, 4] and x < 50:
            triggered.add(36)
        if time_period in [3, 4] and y < 50:
            triggered.add(37)
        if time_period in [5, 6] and 40 < x < 80:
            triggered.add(38)
        if time_period in [5, 6] and 40 < y < 80:
            triggered.add(39)

    if weather == 2:  # 雨天
        if time_period in [1, 2] and x > 75:
            triggered.add(40)
        if time_period in [1, 2] and y < 60:
            triggered.add(41)
        if time_period in [3, 4] and x < 45:
            triggered.add(42)
        if time_period in [3, 4] and y > 65:
            triggered.add(43)
        if time_period in [5, 6] and 35 < x < 75:
            triggered.add(44)
        if time_period in [5, 6] and 35 < y < 75:
            triggered.add(45)

    if weather == 3:  # 雾天
        if time_period in [1, 2] and x > 60:
            triggered.add(46)
        if time_period in [1, 2] and y > 65:
            triggered.add(47)
        if time_period in [3, 4] and x < 55:
            triggered.add(48)
        if time_period in [3, 4] and y < 55:
            triggered.add(49)
        if time_period in [5, 6] and 30 < x < 70:
            triggered.add(50)
        if time_period in [5, 6] and 30 < y < 70:
            triggered.add(51)

    if weather == 4:  # 雪天
        if time_period in [1, 2] and x > 65:
            triggered.add(52)
        if time_period in [1, 2] and y < 55:
            triggered.add(53)
        if time_period in [3, 4] and x < 40:
            triggered.add(54)
        if time_period in [3, 4] and y > 60:
            triggered.add(55)
        if time_period in [5, 6] and 25 < x < 65:
            triggered.add(56)
        if time_period in [5, 6] and 25 < y < 65:
            triggered.add(57)

    if weather == 5:  # 风天
        if time_period in [1, 2] and x > 70:
            triggered.add(58)
        if time_period in [1, 2] and y > 60:
            triggered.add(59)
        if time_period in [3, 4] and x < 35:
            triggered.add(60)
        if time_period in [3, 4] and y < 40:
            triggered.add(61)
        if time_period in [5, 6] and 20 < x < 60:
            triggered.add(62)
        if time_period in [5, 6] and 20 < y < 60:
            triggered.add(63)

    if weather == 6:  # 暴雨
        if time_period in [1, 2] and x > 55:
            triggered.add(64)
        if time_period in [1, 2] and y > 55:
            triggered.add(65)
        if time_period in [3, 4] and x < 45:
            triggered.add(66)
        if time_period in [3, 4] and y < 45:
            triggered.add(67)
        if time_period in [5, 6] and 15 < x < 55:
            triggered.add(68)

    # 69-78: 复合条件（多参数组合）
    if weather + time_period > 6:
        if x > 50 and y > 50:
            triggered.add(69)
        if x < 50 and y < 50:
            triggered.add(70)
        if x > y:
            triggered.add(71)
        if x < y:
            triggered.add(72)
        if abs(x - y) < 20:
            triggered.add(73)

    if weather + time_period <= 6:
        if x > 60 or y > 60:
            triggered.add(74)
        if x < 40 or y < 40:
            triggered.add(75)
        if x + y > 100:
            triggered.add(76)
        if x + y < 80:
            triggered.add(77)
        if abs(x - y) > 30:
            triggered.add(78)

    # 79-88: 数值关系条件
    if weather % 2 == time_period % 2:  # 同奇偶性
        if x % 10 < 5:
            triggered.add(79)
        if y % 10 >= 5:
            triggered.add(80)
        if (x + y) % 3 == 0:
            triggered.add(81)
        if (x * y) % 7 == 0:
            triggered.add(82)
        if x // 10 == y // 10:
            triggered.add(83)

    if weather % 2 != time_period % 2:  # 不同奇偶性
        if x > 75 or y > 75:
            triggered.add(84)
        if x < 25 or y < 25:
            triggered.add(85)
        if max(x, y) - min(x, y) > 40:
            triggered.add(86)
        if (x + y) // 2 > 50:
            triggered.add(87)
        if weather * time_period > 15:
            triggered.add(88)

    # 89-95: 高级组合条件（奇数天气）
    if weather in [1, 3, 5]:  # 奇数天气
        if time_period in [1, 3, 5] and x > 40:
            triggered.add(89)
        if time_period in [2, 4, 6] and y > 40:
            triggered.add(90)
        if x % 20 < 10 and y % 20 < 10:
            triggered.add(91)
        if x + weather * 10 > 50:
            triggered.add(92)
        if y + time_period * 10 > 50:
            triggered.add(93)
        if time_period in [1, 3, 5] and x < 60:
            triggered.add(94)
        if time_period in [2, 4, 6] and y < 60:
            triggered.add(95)

    # 96-98: 偶数天气条件
    if weather in [2, 4, 6]:  # 偶数天气
        if (x + y) % weather == 0:
            triggered.add(96)
        if x * weather > 100:
            triggered.add(97)
        if y * time_period > 100:
            triggered.add(98)

    # 99-100: 最后的复杂条件
    if (weather * time_period + z) % 7 == 0:
        triggered.add(99)
    if max(weather, time_period) * min(x, y) > 150:
        triggered.add(100)

    return triggered

def execute_Tr(weather, time_period, z):
    """执行验证规则的包装函数"""
    return execute_validation_rules_block4(weather, time_period, z)

# === 目标路径定义 ===
target_paths = [
    [15, 16, 48, 49, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 91, 92, 93, 94, 99, 100],
    [16, 18, 19, 60, 61, 70, 71, 72, 73, 80, 81, 82, 83, 89, 91, 92, 93, 94, 99, 100],
    [1, 4, 6, 46, 47, 74, 75, 76, 77, 78, 80, 81, 82, 83, 89, 92, 93, 94, 99, 100],
    [30, 31, 50, 51, 70, 71, 72, 73, 84, 85, 86, 87, 88, 91, 92, 93, 95, 99, 100],
    [18, 19, 36, 37, 74, 76, 77, 78, 79, 80, 81, 82, 83, 89, 92, 93, 94, 99, 100],
    [20, 24, 25, 36, 37, 76, 77, 78, 84, 86, 87, 88, 90, 91, 92, 93, 95, 99, 100],
    [8, 12, 34, 35, 74, 75, 76, 77, 78, 84, 86, 87, 88, 90, 91, 92, 93, 95, 100],
    [8, 10, 58, 59, 70, 71, 72, 73, 84, 85, 86, 87, 88, 91, 92, 93, 95, 99, 100],
    [8, 14, 46, 47, 75, 76, 77, 78, 84, 85, 86, 87, 88, 90, 92, 93, 95, 99, 100],
    [1, 2, 6, 46, 47, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 92, 93, 94, 100],
    [39, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 91, 92, 93, 94, 99, 100],
    [20, 21, 60, 61, 70, 71, 72, 73, 84, 85, 86, 87, 88, 90, 92, 93, 95, 99],
    [8, 9, 11, 13, 40, 41, 75, 76, 77, 78, 79, 80, 81, 83, 96, 97, 98, 100],
    [18, 19, 54, 55, 70, 71, 72, 73, 84, 86, 87, 88, 96, 97, 98, 99, 100],
    [27, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 91, 92, 93, 94, 99, 100],
    [25, 48, 49, 69, 71, 72, 73, 84, 85, 86, 87, 88, 90, 92, 93, 95, 100],
    [26, 28, 62, 70, 71, 72, 73, 80, 81, 82, 83, 89, 91, 92, 93, 94, 100],
    [32, 33, 68, 69, 71, 72, 73, 79, 80, 81, 82, 83, 96, 97, 98, 99, 100],
    [1, 52, 53, 74, 75, 76, 77, 78, 84, 85, 86, 87, 88, 97, 98, 99, 100],
    [8, 12, 14, 64, 65, 69, 71, 72, 73, 80, 81, 82, 83, 96, 97, 98, 100],
    [1, 3, 64, 65, 70, 71, 72, 73, 84, 86, 87, 88, 96, 97, 98, 99, 100],
    [22, 36, 37, 76, 77, 78, 85, 86, 87, 88, 90, 91, 93, 95, 100],
    [31, 45, 70, 71, 72, 73, 79, 80, 81, 83, 96, 97, 98, 99, 100],
    [22, 66, 67, 69, 71, 72, 73, 79, 80, 82, 83, 97, 98, 100],
    [44, 45, 69, 71, 72, 73, 79, 80, 83, 96, 97, 98, 99, 100],
    [57, 71, 72, 73, 79, 80, 83, 97, 98, 100],
    [15, 16, 17, 48, 49, 74, 75, 76, 77, 78, 79, 80, 82, 83, 89, 91, 92, 93, 94, 100],
    [1, 2, 5, 46, 47, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 91, 92, 93, 94, 100],
    [20, 21, 25, 42, 43, 74, 76, 77, 78, 79, 80, 81, 82, 83, 96, 97, 98, 99, 100],
    [2, 5, 7, 40, 41, 75, 76, 77, 78, 84, 85, 86, 87, 88, 96, 97, 98, 99, 100],
    [26, 28, 56, 57, 70, 71, 72, 73, 84, 85, 86, 87, 88, 96, 97, 98, 99, 100],
    [26, 28, 38, 74, 76, 77, 78, 80, 81, 82, 83, 89, 91, 92, 93, 94, 100],
    [30, 31, 62, 63, 70, 71, 72, 73, 84, 86, 87, 88, 90, 91, 92, 93, 95],
    [29, 62, 63, 71, 72, 73, 84, 85, 86, 87, 88, 90, 92, 93, 95, 100],
    [23, 25, 60, 61, 71, 72, 73, 84, 85, 86, 87, 88, 90, 92, 93, 95, 100]
]

# 转换为集合
target_paths = [set(path) for path in target_paths]

targetPaths = [set(path) for path in target_paths]
NUM_PATHS = len(targetPaths)

def jaccard_similarity(set1, set2):
    """计算Jaccard相似度"""
    if not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if set2.issubset(set1):
        return 1.0

    return intersection / union if union != 0 else 0.0

# === 路径相似度矩阵计算和自动分组算法 ===
def compute_path_similarity_matrix(paths):
    """计算路径相似度矩阵"""
    n = len(paths)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = len(paths[i] & paths[j])
            union = len(paths[i] | paths[j])
            matrix[i][j] = inter / union if union > 0 else 0.0
    return matrix

def group_paths_by_similarity(paths, threshold_percentile=50):
    """基于相似度自动分组路径"""
    sim_matrix = compute_path_similarity_matrix(paths)
    avg_sim_scores = np.mean(sim_matrix, axis=1)
    threshold = np.percentile(avg_sim_scores, threshold_percentile)

    center_idx = np.argmax(avg_sim_scores)
    similar_group = [center_idx]

    for i in range(len(paths)):
        if i != center_idx and sim_matrix[center_idx][i] > threshold:
            similar_group.append(i)

    isolated_group = [i for i in range(len(paths)) if i not in similar_group]
    return similar_group, isolated_group

# === 优化的样本生成 ===
def compute_robustness(state, path, sample_size=9):
    """计算鲁棒性（调整为新范围的邻域）"""
    base = execute_Tr(state[0], state[1], state[2])
    if not base:
        return 0.0

    rob, neighbors = 0.0, 0

    # 调整邻域步长：根据新范围调整
    # weather范围1-6，步长±1
    # time_period范围1-6，步长±1
    # z范围1-60，步长±6(约10%)，±3(约5%)
    deltas = [
        (-1, -1, -6), (0, -1, 0), (1, -1, 6),
        (-1, 0, -6), (1, 0, 6),
        (-1, 1, -6), (0, 1, 0), (1, 1, 6),
        (0, 0, 0)
    ]

    for dw, dt, dz in deltas[:sample_size]:
        if dw == dt == dz == 0:
            continue

        neighbor_weather = int(np.clip(state[0] + dw, WEATHER_MIN, WEATHER_MAX))
        neighbor_time = int(np.clip(state[1] + dt, TIME_MIN, TIME_MAX))
        neighbor_z = int(np.clip(state[2] + dz, Z_MIN, Z_MAX))
        neighbor = (neighbor_weather, neighbor_time, neighbor_z)

        n_trig = execute_Tr(neighbor[0], neighbor[1], neighbor[2])
        if not n_trig:
            continue

        rob += jaccard_similarity(n_trig, base)
        neighbors += 1

    return rob / neighbors if neighbors > 0 else 0.0

def generate_samples_for_all_paths(num_candidates=2000, top_k=200, run_id=1):
    """为所有路径生成优化样本"""
    BEST_WEIGHTS = [0.55, 0.25, 0.2]

    def save_samples(path_id, samples, base_dir):
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"path{path_id}_individual.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Individual Path {path_id} (Weighted Screening) - Run {run_id}\n")
            f.write("weather time_period z\tScore\tSimilarity\tRobustness\tLengthDiff\n")
            for s in samples:
                weather, time_period, z = s['state']
                f.write(
                    f"{weather} {time_period} {z}\t{s['score']:.4f}\t{s['similarity']:.4f}\t"
                    f"{s['robustness']:.4f}\t{s['length_diff']:.4f}\n"
                )

    base_dir = r"D:\实验\CNN\DQNNEW\path_samples_individual"

    for path_idx in range(len(targetPaths)):
        path = targetPaths[path_idx]
        candidate_samples = []
        attempts = 0
        max_attempts = num_candidates * 10

        while len(candidate_samples) < num_candidates and attempts < max_attempts:
            attempts += 1

            weather = np.random.randint(WEATHER_MIN, WEATHER_MAX + 1)
            time_period = np.random.randint(TIME_MIN, TIME_MAX + 1)
            z = np.random.randint(Z_MIN, Z_MAX + 1)
            state = (weather, time_period, z)

            triggered = execute_Tr(weather, time_period, z)
            if not triggered:
                continue

            sim = jaccard_similarity(triggered, path)
            rob = compute_robustness(state, path)
            len_diff = 1 - abs(len(triggered) - len(path)) / max(len(triggered), len(path))

            candidate_samples.append({
                'state': state,
                'similarity': sim,
                'robustness': rob,
                'length_diff': len_diff,
                'triggered': triggered
            })

        if candidate_samples:
            for sample in candidate_samples:
                score = (BEST_WEIGHTS[0] * sample['similarity'] +
                         BEST_WEIGHTS[1] * sample['robustness'] +
                         BEST_WEIGHTS[2] * sample['length_diff'])
                sample['score'] = score

            candidate_samples.sort(key=lambda x: x['score'], reverse=True)
            selected_samples = candidate_samples[:top_k]
            save_samples(path_id=path_idx + 1, samples=selected_samples, base_dir=base_dir)

# === 共享经验回放池（改进版：不重复采样）===
class SharedExperienceReplay:
    """共享经验回放池（带优先级采样和不重复抽取）"""

    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.priorities = deque(maxlen=self.capacity)

    def append(self, experience):
        """添加经验"""
        self.buffer.append(experience)
        self.priorities.append(experience[-1])

    def sample(self, batch_size, alpha=0.6):
        """优先级采样"""
        if len(self.buffer) < batch_size:
            return [], [], []

        priorities = np.array(self.priorities, dtype=np.float64)
        priorities = np.power(priorities, alpha)
        probabilities = priorities / np.sum(priorities)

        batch_indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        batch = [self.buffer[idx] for idx in batch_indices]

        return batch, batch_indices, probabilities[batch_indices]

    def update_priorities(self, batch_indices, td_errors):
        """更新优先级"""
        for idx, td_error in zip(batch_indices, td_errors):
            if idx < len(self.priorities):
                self.priorities[idx] = max(abs(td_error), 1e-6)

    def __len__(self):
        return len(self.buffer)

    def get_high_reward_samples(self, target_path, num_samples=20):
        """获取高奖励样本（不重复抽取，返回反归一化后的状态）"""
        if len(self.buffer) == 0:
            return []

        samples_with_scores = []
        seen_states = set()  # 用于去重

        for experience in self.buffer:
            state_tensor = experience[0]
            state_norm = state_tensor.cpu().numpy().flatten()

            # 反归一化状态
            state_tuple = denormalize_state((state_norm[0], state_norm[1], state_norm[2]))

            # 去重检查
            if state_tuple in seen_states:
                continue
            seen_states.add(state_tuple)

            triggered = execute_Tr(state_tuple[0], state_tuple[1], state_tuple[2])
            reward = compute_reward(state_tuple, target_path, triggered, None, None)
            sim = jaccard_similarity(triggered, target_path)

            samples_with_scores.append((state_tuple, reward, sim, triggered))

        samples_with_scores.sort(key=lambda x: x[1], reverse=True)
        return samples_with_scores[:num_samples]

def load_path_data(file_path):
    """加载筛选后的路径数据"""
    path_data = []

    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return path_data

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[2:]:
                parts = line.strip().split("\t")
                if parts:
                    values = parts[0].split()
                    if len(values) >= 3:
                        state = (int(values[0]), int(values[1]), int(values[2]))
                        path_data.append(state)
    except Exception as e:
        print(f"读取文件出错 {file_path}: {e}")

    return path_data

# === DQN网络 ===
class DQN(nn.Module):
    """深度Q网络"""

    def __init__(self, state_dim, action_dim, hidden_dims=[128, 64]):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], action_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# === DQN Agent with PER（改进版：使用归一化）===
class DQNAgentWithPER:
    """带优先经验回放的DQN智能体（使用归一化数据）"""

    def __init__(self, state_dim, action_dim, replay_buffer,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
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
        """
        解码动作索引为状态变化量（调整为新范围）
        范围30个动作，设计动作：
        - weather: ±1, 0(x2)
        - time_period: ±1, 0(x2)
        - z: ±12(约20%), ±6(约10%), ±3(约5%), 0(x2)
        """
        delta_values_weather_time = [1, 0, 0, -1]  # 天气和时间段的变化
        delta_values_z = [12, 6, 3, 0, 0, -3, -6, -12]  # z的变化

        dim = action_idx // 10
        delta_idx = action_idx % 10

        if dim == 0:  # weather
            if delta_idx >= 4:
                delta_idx = 3
            return (delta_values_weather_time[delta_idx], 0, 0)
        elif dim == 1:  # time_period
            if delta_idx >= 4:
                delta_idx = 3
            return (0, delta_values_weather_time[delta_idx], 0)
        elif dim == 2:  # z
            if delta_idx >= 8:
                delta_idx = 7
            return (0, 0, delta_values_z[delta_idx])

    def act(self, state_norm, legal_actions=None):
        """选择动作（使用归一化状态）"""
        if legal_actions is None:
            legal_actions = list(range(self.action_dim))

        if not legal_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        state_tensor = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)[0]

        legal_q_values = q_values[legal_actions]
        best_legal_idx = torch.argmax(legal_q_values).item()
        return legal_actions[best_legal_idx]

    def get_legal_actions(self, state):
        """获取当前状态下的合法动作（使用原始状态）"""
        legal_actions = []

        for action_idx in range(self.action_dim):
            dw, dt, dz = self.decode_action(action_idx)

            next_weather = state[0] + dw
            next_time = state[1] + dt
            next_z = state[2] + dz

            if (WEATHER_MIN <= next_weather <= WEATHER_MAX and
                    TIME_MIN <= next_time <= TIME_MAX and
                    Z_MIN <= next_z <= Z_MAX):
                legal_actions.append(action_idx)

        return legal_actions

    def store_transition(self, state, action, reward, next_state, done):
        """存储转换（使用归一化状态）"""
        # 归一化状态
        state_norm = normalize_state(state)
        next_state_norm = normalize_state(next_state)

        state_tensor = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0).to(device)
        next_state_tensor = torch.tensor(next_state_norm, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.model(state_tensor)
            next_q_values = self.target_model(next_state_tensor)
            max_next_q = next_q_values.max(1)[0]
            target_q = reward + (self.gamma * max_next_q * (1 - done))
            td_error = abs(q_values[0][action].item() - target_q.item())

        self.replay_buffer.append((state_tensor, action, reward, next_state_tensor, done, td_error))
        return td_error

    def train(self, batch_size=32):
        """训练网络"""
        if len(self.replay_buffer) < batch_size:
            return 0.0

        batch, batch_indices, probabilities = self.replay_buffer.sample(batch_size, alpha=self.alpha)

        if not batch:
            return 0.0

        states, actions, rewards, next_states, dones, _ = zip(*batch)

        weights = (len(self.replay_buffer) * probabilities) ** (-self.beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        states = torch.cat(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.cat(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_max_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_max_q * (1 - dones))

        td_errors = current_q - target_q
        weighted_loss = (td_errors.pow(2) * weights).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        new_priorities = abs(td_errors.detach().cpu().numpy())
        self.replay_buffer.update_priorities(batch_indices, new_priorities)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return weighted_loss.item()

    def update_target_model(self):
        """更新目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())

# === 训练流程 ===
def generate_and_train_for_individual_paths(path_documents, repeats=5, batch_size=32, run_id=1):
    """使用归一化和共享经验池的训练流程"""
    state_dim = 3
    action_dim = 30

    shared_replay_buffer = SharedExperienceReplay(capacity=20000)
    agent = DQNAgentWithPER(state_dim, action_dim, shared_replay_buffer)

    total_cumulative_reward = 0
    path_rewards = {}

    print(f"\n=== 运行 {run_id}/20 开始训练（归一化+不重复采样版本）===")
    start_time = time.time()

    SAMPLES_PER_BATCH = 50
    NUM_BATCHES = 4
    STEPS_PER_SAMPLE = 3

    for path_idx in range(len(targetPaths)):
        path_id = path_idx + 1
        print(f"\n{'=' * 60}")
        print(f"训练路径 {path_id}/{NUM_PATHS}")
        print(f"{'=' * 60}")

        file_path = os.path.join(path_documents, f"path{path_id}_individual.txt")
        if not os.path.exists(file_path):
            print(f"  警告: 文件不存在 {file_path}")
            continue

        path_data = load_path_data(file_path)
        if not path_data:
            print(f"  警告: 路径 {path_id} 无有效数据")
            continue

        target_path = targetPaths[path_idx]

        if path_idx not in path_rewards:
            path_rewards[path_idx] = 0

        for repeat_idx in range(repeats):
            print(f"\n  第 {repeat_idx + 1}/{repeats} 轮")

            for batch_idx in range(NUM_BATCHES):
                batch_start = batch_idx * SAMPLES_PER_BATCH
                batch_end = min(batch_start + SAMPLES_PER_BATCH, len(path_data))

                print(f"    批次 {batch_idx + 1}/{NUM_BATCHES} (样本 {batch_start}-{batch_end})")

                for sample_idx in range(batch_start, batch_end):
                    state = path_data[sample_idx]
                    prev_state = None
                    prev_triggered = None

                    for step in range(STEPS_PER_SAMPLE):
                        legal_actions = agent.get_legal_actions(state)

                        if not legal_actions:
                            break

                        # 使用归一化状态选择动作
                        state_norm = normalize_state(state)
                        action = agent.act(state_norm, legal_actions)
                        if action is None:
                            break

                        dw, dt, dz = agent.decode_action(action)
                        next_state = (
                            int(np.clip(state[0] + dw, WEATHER_MIN, WEATHER_MAX)),
                            int(np.clip(state[1] + dt, TIME_MIN, TIME_MAX)),
                            int(np.clip(state[2] + dz, Z_MIN, Z_MAX))
                        )

                        triggered = execute_Tr(next_state[0], next_state[1], next_state[2])
                        reward = compute_reward(next_state, target_path, triggered,
                                                prev_triggered, prev_state)
                        done = (step == STEPS_PER_SAMPLE - 1)

                        # 存储时自动归一化
                        agent.store_transition(state, action, reward, next_state, done)

                        prev_state = state
                        prev_triggered = triggered
                        state = next_state

                        total_cumulative_reward += reward
                        path_rewards[path_idx] += reward

                if len(agent.replay_buffer) >= batch_size:
                    loss = agent.train(batch_size)
                    print(f"      批次 {batch_idx + 1} 完成，损失: {loss:.4f}")

                if (batch_idx + 1) % 2 == 0:
                    agent.update_target_model()
                    print(f"      已完成 {batch_idx + 1} 批，更新目标网络")

        print(f"\n路径 {path_id} 完成，累积奖励: {path_rewards[path_idx]:.2f}")
        print(f"共享池大小: {len(shared_replay_buffer)}")

    training_time = time.time() - start_time
    print(f"\n=== 运行 {run_id}/20 完成，用时: {training_time:.2f}秒 ===")

    return agent, shared_replay_buffer, total_cumulative_reward, path_rewards, training_time

# === 创建Excel报告函数 ===
def create_consolidated_excel_report(all_runs_data, similar_group, isolated_group, output_dir):
    """创建Excel综合报告"""
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

    # === 工作表1: 各路径详细表现 ===
    ws_paths = wb.active
    ws_paths.title = "各路径详细表现"

    path_headers = ['路径编号', '分组类型'] + [f'第{i}次' for i in range(1, 21)] + \
                   ['平均相似度', '最高相似度', '最低相似度', '标准差']

    for col, header in enumerate(path_headers, 1):
        cell = ws_paths.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = PatternFill(start_color=header_color, end_color=header_color, fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    ws_paths.row_dimensions[1].height = 30

    for path_id in range(1, NUM_PATHS + 1):
        row = path_id + 1

        if path_id in similar_group_paths:
            group_type = "高关联度路径组"
            row_color = similar_group_color
        elif path_id in isolated_group_paths:
            group_type = "低关联度路径组"
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

    group_headers = ['分组名称', '包含路径'] + [f'第{i}次' for i in range(1, 21)] + \
                    ['平均相似度', '标准差']

    for col, header in enumerate(group_headers, 1):
        cell = ws_groups.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = PatternFill(start_color=header_color, end_color=header_color, fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    ws_groups.row_dimensions[1].height = 30

    row = 2

    # 高关联度路径组
    cell = ws_groups.cell(row=row, column=1, value="高关联度路径组")
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
        group_sim = np.mean([
            run_data['path_similarities'].get(p, {}).get('avg_similarity', 0.0)
            for p in similar_group_paths
        ])
        group_similarities.append(group_sim)

        cell = ws_groups.cell(row=row, column=3 + run_idx, value=round(group_sim, 4))
        cell.number_format = '0.0000'
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

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

    # 低关联度路径组
    if isolated_group_paths:
        cell = ws_groups.cell(row=row, column=1, value="低关联度路径组")
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
            iso_sim = np.mean([
                run_data['path_similarities'].get(p, {}).get('avg_similarity', 0.0)
                for p in isolated_group_paths
            ])
            isolated_similarities.append(iso_sim)

            cell = ws_groups.cell(row=row, column=3 + run_idx, value=round(iso_sim, 4))
            cell.number_format = '0.0000'
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border

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

    ws_groups.column_dimensions['A'].width = 16
    ws_groups.column_dimensions['B'].width = 22
    for col in range(3, 23):
        ws_groups.column_dimensions[get_column_letter(col)].width = 10
    ws_groups.column_dimensions[get_column_letter(23)].width = 14
    ws_groups.column_dimensions[get_column_letter(24)].width = 12

    # === 工作表3: 详细样本数据 ===
    ws_samples = wb.create_sheet("详细样本数据")

    sample_headers = ['运行次数', '路径编号', '样本序号', '天气', '时间段',
                      '行人数量', '相似度', '触发规则集合']

    for col, header in enumerate(sample_headers, 1):
        cell = ws_samples.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = PatternFill(start_color=header_color, end_color=header_color, fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    ws_samples.row_dimensions[1].height = 30

    sample_row = 2
    for run_idx, run_data in enumerate(all_runs_data, 1):
        for path_id in range(1, NUM_PATHS + 1):
            samples = run_data['path_samples'].get(path_id, [])

            if path_id in similar_group_paths:
                path_color = similar_group_color
            elif path_id in isolated_group_paths:
                path_color = isolated_group_color
            else:
                path_color = "FFFFFF"

            for sample_idx, (state_tuple, reward, sim, triggered) in enumerate(samples, 1):
                weather, time_period, z = state_tuple
                triggered_str = ','.join(map(str, sorted(triggered)))

                cell = ws_samples.cell(row=sample_row, column=1, value=f"第{run_idx}次")
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.fill = PatternFill(start_color=path_color, end_color=path_color, fill_type="solid")
                cell.border = thin_border

                cell = ws_samples.cell(row=sample_row, column=2, value=f"路径{path_id}")
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.fill = PatternFill(start_color=path_color, end_color=path_color, fill_type="solid")
                cell.border = thin_border

                cell = ws_samples.cell(row=sample_row, column=3, value=sample_idx)
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = thin_border

                for col_offset, value in enumerate([weather, time_period, z]):
                    cell = ws_samples.cell(row=sample_row, column=4 + col_offset, value=value)
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = thin_border

                cell = ws_samples.cell(row=sample_row, column=7, value=round(sim, 4))
                cell.number_format = '0.0000'
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = thin_border

                cell = ws_samples.cell(row=sample_row, column=8, value=f"{{{triggered_str}}}")
                cell.alignment = Alignment(horizontal="left", vertical="center")
                cell.border = thin_border

                sample_row += 1

    sample_widths = [13, 13, 11, 10, 10, 12, 12, 45]
    for i, width in enumerate(sample_widths, 1):
        ws_samples.column_dimensions[get_column_letter(i)].width = width

    # === 工作表4: 运行统计汇总 ===
    ws_summary = wb.create_sheet("运行统计汇总")

    summary_headers = ['运行次数', '训练时间(秒)', '总体平均相似度', '最高相似度',
                       '最低相似度', '高关联组平均', '低关联组平均', '共享池大小']

    for col, header in enumerate(summary_headers, 1):
        cell = ws_summary.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = PatternFill(start_color=header_color, end_color=header_color, fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    ws_summary.row_dimensions[1].height = 30

    for run_idx, run_data in enumerate(all_runs_data, 1):
        row = run_idx + 1

        # 计算高关联组和低关联组的平均相似度
        high_group_avg = np.mean([
            run_data['path_similarities'].get(p, {}).get('avg_similarity', 0.0)
            for p in similar_group_paths
        ])

        low_group_avg = 0.0
        if isolated_group_paths:
            low_group_avg = np.mean([
                run_data['path_similarities'].get(p, {}).get('avg_similarity', 0.0)
                for p in isolated_group_paths
            ])

        values = [
            f"第{run_idx}次",
            round(run_data['training_time'], 2),
            round(run_data['overall_avg_similarity'], 4),
            round(run_data['max_similarity'], 4),
            round(run_data['min_similarity'], 4),
            round(high_group_avg, 4),
            round(low_group_avg, 4),
            20000  # 共享池容量
        ]

        for col, value in enumerate(values, 1):
            cell = ws_summary.cell(row=row, column=col, value=value)
            if col == 1:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif col == 2:
                cell.number_format = '0.00'
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif col >= 3 and col <= 7:
                cell.number_format = '0.0000'
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border

    # 添加统计行
    stat_row = len(all_runs_data) + 2
    stat_labels = ['', '总计/平均', '平均值', '最大值', '最小值', '平均值', '平均值', '固定值']

    for col, label in enumerate(stat_labels, 1):
        cell = ws_summary.cell(row=stat_row, column=col, value=label)
        cell.font = Font(bold=True, size=11)
        cell.fill = PatternFill(start_color=stats_color, end_color=stats_color, fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    stat_row += 1

    # 计算统计值
    training_times = [r['training_time'] for r in all_runs_data]
    overall_avgs = [r['overall_avg_similarity'] for r in all_runs_data]
    max_sims = [r['max_similarity'] for r in all_runs_data]
    min_sims = [r['min_similarity'] for r in all_runs_data]

    high_group_avgs = []
    low_group_avgs = []
    for run_data in all_runs_data:
        high_avg = np.mean([
            run_data['path_similarities'].get(p, {}).get('avg_similarity', 0.0)
            for p in similar_group_paths
        ])
        high_group_avgs.append(high_avg)

        if isolated_group_paths:
            low_avg = np.mean([
                run_data['path_similarities'].get(p, {}).get('avg_similarity', 0.0)
                for p in isolated_group_paths
            ])
            low_group_avgs.append(low_avg)

    stat_values = [
        '统计结果',
        round(np.sum(training_times), 2),
        round(np.mean(overall_avgs), 4),
        round(np.max(max_sims), 4),
        round(np.min(min_sims), 4),
        round(np.mean(high_group_avgs), 4),
        round(np.mean(low_group_avgs), 4) if low_group_avgs else 0.0,
        20000
    ]

    for col, value in enumerate(stat_values, 1):
        cell = ws_summary.cell(row=stat_row, column=col, value=value)
        cell.font = Font(bold=True, size=11)
        cell.fill = PatternFill(start_color=stats_color, end_color=stats_color, fill_type="solid")
        if col == 1:
            cell.alignment = Alignment(horizontal="center", vertical="center")
        elif col == 2:
            cell.number_format = '0.00'
            cell.alignment = Alignment(horizontal="center", vertical="center")
        elif col >= 3 and col <= 7:
            cell.number_format = '0.0000'
            cell.alignment = Alignment(horizontal="center", vertical="center")
        else:
            cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    summary_widths = [13, 16, 18, 14, 14, 16, 16, 14]
    for i, width in enumerate(summary_widths, 1):
        ws_summary.column_dimensions[get_column_letter(i)].width = width

    output_path = os.path.join(output_dir, "20次运行综合报告_新变量范围.xlsx")
    wb.save(output_path)
    print(f"\n✅ 综合Excel报告已生成: {output_path}")
    print(f"   包含4张工作表: 各路径详细表现、分组统计、详细样本数据、运行统计汇总")

def run_20_times_training():
    """运行20次完整的训练流程"""
    model_path_base = r"D:\实验\CNN\DQNNEW\saved_models_new_vars"
    path_documents = r"D:\实验\CNN\DQNNEW\path_samples_individual"
    output_dir = r"D:\实验\CNN\对比实验二\excel_reports_new_vars"

    os.makedirs(model_path_base, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    similar_group, isolated_group = group_paths_by_similarity(targetPaths)

    similar_group_display = [idx + 1 for idx in similar_group]
    isolated_group_display = [idx + 1 for idx in isolated_group]

    print("=" * 60)
    print("开始20次连续训练实验 - 新变量范围版本")
    print("=" * 60)
    print(f"\n路径总数: {NUM_PATHS} 条")
    print(f"\n自动分组结果:")
    print(f"  相似路径组: {similar_group_display}")
    print(f"  孤岛路径组: {isolated_group_display}")
    print(f"\n新变量取值范围:")
    print(f"  天气 (weather): {WEATHER_MIN}-{WEATHER_MAX}")
    print(f"  时间段 (time_period): {TIME_MIN}-{TIME_MAX}")
    print(f"  行人数量 (z): {Z_MIN}-{Z_MAX}")
    print(f"\n变量映射逻辑:")
    print(f"  x = z (行人数量)")
    print(f"  y = (weather * time_period * 10 + z) % 100 + 1")
    print(f"\n归一化配置:")
    print(f"  训练时: 所有状态归一化到[0, 1]")
    print(f"  抽取时: 自动反归一化到原始范围")
    print(f"  采样策略: 不重复抽取（使用seen_states去重）")
    print(f"\n动作空间配置 (共30个动作):")
    print(f"  weather维度: ±1, 0(x2)")
    print(f"  time_period维度: ±1, 0(x2)")
    print(f"  z维度: ±12, ±6, ±3, 0(x2)")
    print("\n" + "=" * 60)

    all_runs_data = []
    total_start_time = time.time()

    for run_id in range(1, 21):
        print(f"\n{'=' * 60}")
        print(f"开始第 {run_id}/20 次运行")
        print(f"{'=' * 60}")

        print(f"[第{run_id}次] 正在生成样本...")
        generate_samples_for_all_paths(num_candidates=2000, top_k=200, run_id=run_id)

        print(f"[第{run_id}次] 开始训练...")
        agent, shared_buffer, total_cumulative_reward, path_rewards, training_time = \
            generate_and_train_for_individual_paths(path_documents, repeats=5,
                                                    batch_size=32, run_id=run_id)

        model_path = os.path.join(model_path_base, f"trained_model_run_{run_id}.pth")
        torch.save({
            'model_state_dict': agent.model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'run_id': run_id,
            'normalized': True,
            'value_ranges': {
                'weather': [WEATHER_MIN, WEATHER_MAX],
                'time_period': [TIME_MIN, TIME_MAX],
                'z': [Z_MIN, Z_MAX]
            }
        }, model_path)
        print(f"[第{run_id}次] 模型已保存: {model_path}")

        run_data = {
            'run_id': run_id,
            'training_time': training_time,
            'total_reward': total_cumulative_reward,
            'path_rewards': path_rewards,
            'path_similarities': {},
            'path_samples': {}
        }

        all_similarities = []
        for path_idx in range(len(targetPaths)):
            target_path = targetPaths[path_idx]
            high_reward_samples = shared_buffer.get_high_reward_samples(target_path, num_samples=20)

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
    print(f"\n所有结果已保存到: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    run_20_times_training()