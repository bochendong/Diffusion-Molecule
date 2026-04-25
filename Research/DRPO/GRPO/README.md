# 极简版GRPO

这是一个极简版本的GRPO（Group Relative Policy Optimization）实现，代码简洁易懂，包含GRPO的核心算法。

## 文件结构

```
GRPO/
├── grpo.py      # 极简版GRPO实现（单文件包含所有功能）
└── README.md    # 说明文档
```

## 核心功能

- **模型加载**：训练模型和参考模型
- **文本生成**：根据prompt生成多个答案
- **奖励计算**：基于规则计算奖励（正确性+格式）
- **GRPO损失**：包含PPO clipping和KL散度惩罚
- **训练循环**：完整的训练流程

## 使用方法

### 1. 修改配置

在 `grpo.py` 文件顶部修改配置：

```python
MODEL_PATH = "your_model_path"  # 模型路径
DEVICE = "cuda"  # 或 "cpu"
LEARNING_RATE = 1e-6
NUM_STEPS = 1000
# ... 其他配置
```

### 2. 准备数据

修改 `DATA` 列表，添加你的训练数据：

```python
DATA = [
    {"Q": "问题1", "A": "答案1"},
    {"Q": "问题2", "A": "答案2"},
    # ...
]
```

### 3. 运行训练

```bash
python grpo.py
```

## 代码结构

### SimpleGRPO类

- `__init__()`: 初始化模型、tokenizer、优化器
- `generate()`: 生成答案
- `get_logps()`: 计算token的log概率
- `grpo_step()`: GRPO损失计算
- `train_step()`: 训练一步

### 辅助函数

- `compute_reward()`: 计算奖励值
- `prepare_batch()`: 准备训练批次
- `train()`: 主训练函数

## GRPO算法核心

1. **生成阶段**：对每个问题生成多个答案
2. **奖励计算**：计算每个答案的奖励
3. **归一化**：归一化奖励值
4. **GRPO损失**：
   - PPO clipping：限制策略更新幅度
   - KL散度惩罚：防止偏离参考模型太远

## 依赖

```bash
pip install torch transformers tqdm
```

## 注意事项

1. 确保模型路径正确
2. 数据格式：每个item包含'Q'（问题）和'A'（答案）
3. 奖励函数是简化版，可根据任务自定义
4. 代码极简，便于理解和修改

## 扩展

可以根据需要扩展：
- 更复杂的奖励函数
- 数据加载器
- 模型保存/加载
- 验证集评估
- 等等

