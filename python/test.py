import numpy as np
import ot_estimators  # 导入你的 C++ 模块

# 创建 OTEstimators 对象
ote = ot_estimators.OTEstimators()

def generate_test_data(samples_A_count, samples_B_count, features):
    samples_A = np.random.rand(samples_A_count, features)
    samples_B = np.random.rand(samples_B_count, features)
    return samples_A, samples_B

# 加载词汇表（vocab）
samples_A_count = 10
samples_B_count = 10
features = 2

samples_A, samples_B= generate_test_data(samples_A_count, samples_B_count, features)

vocab = np.vstack([samples_A, samples_B])

vocab = vocab.astype(np.float32)
ote.load_vocabulary(vocab)

# 加载数据集（dataset），假设数据集只包含两个数据点
dataset = [
    [(i, 1/samples_A_count) for i in range(samples_A_count)],  # 第一个数据点
    [(i+samples_A_count, 1/samples_B_count) for i in range(samples_B_count)]   # 第二个数据点
]
ote.load_dataset(dataset)

# 计算数据集中两个数据点之间的 FlowTree EMD 和流矩阵
# emd, flow_matrix = ote.compute_flowtree_emd_between_dataset_points()

# # 输出结果
# print("EMD between two dataset points:", emd)
# print("Flow matrix:", flow_matrix)

