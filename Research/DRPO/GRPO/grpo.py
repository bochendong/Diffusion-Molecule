"""
极简版GRPO实现
包含GRPO的核心算法，代码简洁易懂
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import random
import re
from tqdm import tqdm

# ==================== 配置 ====================
MODEL_PATH = "Qwen/Qwen2.5-7B"  # 模型路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-6
BETA = 0.04  # KL散度惩罚系数
CLIP_PARAM = 0.2  # PPO clipping参数
NUM_STEPS = 1000
BATCH_SIZE = 1
NUM_SAMPLES = 8  # 每个问题生成的答案数量
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.9

# ==================== 数据 ====================
# 示例数据（实际使用时替换为真实数据）
DATA = [
    {"Q": "What is 2+2?", "A": "4"},
    {"Q": "What is 3*3?", "A": "9"},
    {"Q": "What is 10-5?", "A": "5"},
]

# ==================== 奖励函数 ====================
def compute_reward(item, answer):
    """计算奖励：正确性 + 格式"""
    # 正确性奖励
    pattern = r'\d+'
    nums = re.findall(pattern, answer)
    correct = 1.0 if nums and float(nums[-1]) == float(item["A"]) else -1.0
    
    # 格式奖励（简化版）
    format_ok = 1.0 if "<think>" in answer and "<answer>" in answer else -1.0
    
    return correct + format_ok

# ==================== GRPO模型 ====================
class SimpleGRPO:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE, lr=LEARNING_RATE):
        self.device = torch.device(device)
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载训练模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32
        ).to(self.device)
        self.model.train()
        
        # 加载参考模型（冻结）
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32
        ).to(self.device)
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # 生成配置
        self.gen_config = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            num_return_sequences=NUM_SAMPLES,
            pad_token_id=self.tokenizer.pad_token_id,
        )
    
    def generate(self, prompts):
        """生成答案"""
        # 格式化prompt
        texts = []
        for p in prompts:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)
        
        # Tokenize
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        
        # 生成
        with torch.inference_mode():
            # self.model is AutoModelForCausalLM
            # outputs is a tensor with shape
            # (batch_size * num_return_sequences, sequence_length)
            # 每个 prompt 会生成 NUM_SAMPLES（默认8）个不同的序列
            outputs = self.model.generate(**inputs, generation_config=self.gen_config)
        
        # 解码
        prompt_len = inputs["input_ids"].shape[1]
        answers = [
            self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            for out in outputs
        ]
        # Example: "What is 2+2?"
        # answers = ["The answer is 4",           # 第1个句子
        #            "It's 4",                    # 第2个句子
        #            "2+2 equals 4",              # 第3个句子
        #            ...
        #             "4"]

        return answers
    
    def get_logps(self, model, input_ids):
        """计算每个token的log概率"""
        logits = model(input_ids).logits[:, :-1, :]  # (B, L-1, V)
        target_ids = input_ids[:, 1:]  # (B, L-1)
        
        logps = []
        for logit_row, target_row in zip(logits, target_ids):
            log_probs = logit_row.log_softmax(dim=-1)
            token_logp = torch.gather(log_probs, dim=1, index=target_row.unsqueeze(1)).squeeze(1)
            logps.append(token_logp)
        return torch.stack(logps)
    
    def grpo_step(self, batch):
        """GRPO训练步

        训练前, 对于batch里的每一个prompt, 先生成一组(8个) answer, 放到batch里, 同时计算每个answer的reward
        """

        # inputs包含merged id （同时有prompt和answer）
        inputs = batch["inputs"].to(self.device)
        rewards = batch["rewards"].to(self.device).unsqueeze(1) 
        prompt_len = batch["prompt_len"]
        
        # logps = 当前模型对answer中每个 token 的 log 概率
        logps = self.get_logps(self.model, inputs)
        logps = logps[:, prompt_len-1:]
        
        # 参考模型logps
        with torch.inference_mode():
            ref_logps = self.get_logps(self.ref_model, inputs)
            ref_logps = ref_logps[:, prompt_len-1:]
        
        # KL散度
        kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
        
        # PPO clipping
        gen_logps = batch["gen_logps"].to(self.device)

        # logps当前模型对answer中每个 token 的 log 概率
        # gen_logps是历史模型对answer中每个 token 的 log 概率
        ratio = torch.exp(logps - gen_logps)
        clipped_ratio = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM)
        loss_term = torch.min(ratio * rewards, clipped_ratio * rewards)
        
        # 最终损失
        loss = -(loss_term - BETA * kl).mean()
        return loss
    
    def train_step(self, batch):
        """训练一步"""
        loss = self.grpo_step(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# ==================== 训练 ====================
def prepare_batch(grpo, items):
    """准备训练批次"""
    prompts = [item["Q"] for item in items]
    
    # 生成答案
    answers = grpo.generate(prompts)
    if not answers:
        return None
    
    # 计算奖励
    rewards = []
    for i, item in enumerate(items):
        for a in answers[i*NUM_SAMPLES:(i+1)*NUM_SAMPLES]:
            rewards.append(compute_reward(item, a))
    
    rewards = torch.tensor(rewards, dtype=torch.float32)
    
    # 归一化奖励
    if rewards.max() - rewards.min() < 1e-4:
        return None
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    
    # Tokenize
    prompt_texts = [
        grpo.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True
        )
        for p in prompts
    ]
    prompt_ids = grpo.tokenizer(prompt_texts, return_tensors="pt", padding=True)["input_ids"]
    
    answer_texts = answers
    answer_ids = grpo.tokenizer(answer_texts, return_tensors="pt", padding=True)["input_ids"]
    
    # 合并prompt和answer
    prompt_len = prompt_ids.shape[1]
    prompt_rep = prompt_ids.repeat(1, NUM_SAMPLES).view(-1, prompt_len)
    merged_ids = torch.cat([prompt_rep, answer_ids], dim=1)
    
    # 计算生成时的logps（用于PPO clipping）
    with torch.inference_mode():
        gen_logps = grpo.get_logps(grpo.model, merged_ids.to(grpo.device))
        gen_logps = gen_logps[:, prompt_len-1:].cpu()
        # gen_logps = 在生成答案时，模型对答案里每个 token 的 log 概率，因此需要[:, prompt_len-1:]

    
    batch = {
        "inputs": merged_ids,
        "rewards": rewards,
        "prompt_len": prompt_len,
        "gen_logps": gen_logps,
    }
    return batch

def train():
    """训练函数"""
    # 初始化
    grpo = SimpleGRPO()
    
    print("开始训练...")
    for step in tqdm(range(1, NUM_STEPS + 1)):
        # 采样数据
        items = random.sample(DATA, min(BATCH_SIZE, len(DATA)))
        
        # 准备批次
        batch = prepare_batch(grpo, items)
        if batch is None:
            continue
        
        # 训练
        loss = grpo.train_step(batch)
        
        # 打印
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")
        
        # 保存
        if step % 200 == 0:
            save_path = f"./checkpoint_step_{step}"
            grpo.model.save_pretrained(save_path)
            grpo.tokenizer.save_pretrained(save_path)
            print(f"模型已保存到: {save_path}")

if __name__ == "__main__":
    train()

