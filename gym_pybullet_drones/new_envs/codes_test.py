import torch
from torch.distributions import Categorical
import time

# 定义概率分布
probs = torch.tensor([
    [0.13243707, 0.12140542, 0.12257885, 0.118792, 0.1231009, 0.12414426, 0.1308618, 0.1266796],
    [0.13333048, 0.11851338, 0.12312149, 0.12077631, 0.12801889, 0.12291928, 0.12531583, 0.12800433],
    [0.13425545, 0.12110135, 0.12147482, 0.11911386, 0.12387094, 0.12306637, 0.12980565, 0.12731153]
])

# 创建Categorical分布
dist = Categorical(probs=probs)

# 测量抽样时间
start_time = time.time()
samples = dist.sample()
end_time = time.time()

# 打印抽样结果和时间
print("Sampled indices:", samples)
print("Time taken:", end_time - start_time, "seconds")
