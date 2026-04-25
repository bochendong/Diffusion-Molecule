"""
简化版GRPO模型框架
基于simple_grpo，去除了GPU配置和deepspeed，搭建基础的模型架子
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import re
import random
import time
from tqdm import tqdm
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# ==================== 配置参数 ====================
class GRPOConfig:
    model_path = "Qwen/Qwen2.5-7B"  # 模型路径，可以改为本地路径
    beta = 0.04  # KL散度惩罚系数
    num_pre_Q = 8  # 每个问题生成的答案数量
    all_steps = 1000  # 总训练步数
    max_prompt_length = 400  # 最大prompt长度
    save_steps = 200  # 保存模型步数间隔
    compute_gen_logps = True  # 是否计算生成时的logps
    clip_param = 0.2  # PPO clipping参数
    learning_rate = 1e-6  # 学习率
    batch_size = 1  # 批次大小
    device = None  # 设备，None表示自动检测，后续可以设置为'cpu'或'cuda'
    
    # 生成配置
    max_new_tokens = 512
    temperature = 0.9
    
    # 系统提示词
    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""


# ==================== 工具函数 ====================
def get_per_token_logps(logits, input_ids):
    """
    计算每个token的log概率
    Args:
        logits: (B, L, V) 模型的logits输出
        input_ids: (B, L) 输入token ids
    Returns:
        per_token_logps: (B, L) 每个token的log概率
    """
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)  # (L, V)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)  # (L,)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


# ==================== Reward函数 ====================
def reward_correct(item, answer):
    """
    计算答案正确性奖励
    TODO: 需要根据实际情况实现math_verify或替换为其他验证方法
    """
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)
    if len(nums) == 0:
        return -1.0
    lastnum = nums[-1]
    # 简化版：直接比较字符串，实际应该使用math_verify
    try:
        # 这里先用简单的数值比较，后续可以替换为math_verify
        ans_num = float(lastnum)
        ground_truth_num = float(item["A"])
        return 1.0 if abs(ans_num - ground_truth_num) < 1e-6 else -1.0
    except:
        return -1.0


def reward_format(item, answer):
    """
    计算格式正确性奖励
    """
    # 注意：这里使用 <think> 标签，与配置中的 system_prompt 保持一致
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1.0


# ==================== GRPO模型类 ====================
class GRPOModel:
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = config.device if config.device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer
        print("加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        
        # 加载训练模型
        print("加载训练模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16 if self.device != 'cpu' else torch.float32,
            _attn_implementation="sdpa"
        )
        self.model.to(self.device)
        self.model.train()
        
        # 加载参考模型（用于计算参考logps，不更新参数）
        print("加载参考模型...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16 if self.device != 'cpu' else torch.float32,
            _attn_implementation="sdpa"
        )
        self.ref_model.to(self.device)
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        
        # 优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # 生成配置
        self.generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=config.temperature,
            num_return_sequences=config.num_pre_Q,
            pad_token_id=self.tokenizer.pad_token_id,
        )
    
    def gen_answers(self, prompts):
        """
        生成答案
        """
        tip_text = []
        for x in prompts:
            tip_text.append(self.tokenizer.apply_chat_template([
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": x}
            ], tokenize=False, add_generation_prompt=True))
        
        tip_inputs = self.tokenizer(
            tip_text, 
            return_tensors="pt", 
            padding=True, 
            padding_side="left", 
            add_special_tokens=False
        )
        prompt_length = tip_inputs["input_ids"].shape[-1]
        
        if prompt_length > self.config.max_prompt_length:
            return []
        
        tip_inputs = {k: v.to(self.device) for k, v in tip_inputs.items()}
        
        with torch.inference_mode():
            tip_completion_ids = self.model.generate(**tip_inputs, generation_config=self.generation_config)
        
        completion_ids = tip_completion_ids[:, prompt_length:]
        answers = [self.tokenizer.decode(x, skip_special_tokens=True) for x in completion_ids]
        return answers
    
    def gen_samples(self, inputs):
        """
        生成样本并计算奖励
        Args:
            inputs: list of dict, 每个dict包含'Q'和'A'
        Returns:
            prompt_inputs: tokenized prompts
            output_ids: tokenized outputs
            rewards: 奖励值
            answers: 原始答案文本
        """
        prompts = [x["Q"] for x in inputs]
        answers = self.gen_answers(prompts)
        
        if len(answers) == 0:
            return None, None, None, None
        
        rewards = []
        for i, inp in enumerate(inputs):
            for a in answers[i*self.config.num_pre_Q:(i+1)*self.config.num_pre_Q]:
                rewards.append(reward_correct(inp, a) + reward_format(inp, a))
        
        prompts_text = [
            self.tokenizer.apply_chat_template([
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": x}
            ], tokenize=False, add_generation_prompt=True) 
            for x in prompts
        ]
        
        prompt_inputs = self.tokenizer(
            prompts_text, 
            return_tensors="pt", 
            padding=True, 
            padding_side="left", 
            add_special_tokens=False
        )["input_ids"]
        
        output_ids = self.tokenizer(
            answers, 
            return_tensors="pt", 
            padding=True, 
            padding_side="right", 
            add_special_tokens=False
        )["input_ids"]
        
        return prompt_inputs, output_ids, torch.tensor(rewards, dtype=torch.float32), answers
    
    def compute_ref_logps(self, input_ids, prompt_length):
        """
        使用参考模型计算logps
        """
        with torch.inference_mode():
            logits = self.ref_model(input_ids.to(self.device)).logits
            logits = logits[:, :-1, :]
            input_ids_shifted = input_ids[:, 1:].to(self.device)
            per_token_logps = get_per_token_logps(logits, input_ids_shifted)
            per_token_logps = per_token_logps[:, prompt_length-1:]
        return per_token_logps.cpu()
    
    def compute_gen_logps(self, input_ids, prompt_length):
        """
        使用当前模型计算生成时的logps（用于PPO clipping）
        """
        with torch.inference_mode():
            logits = self.model(input_ids.to(self.device)).logits
            logits = logits[:, :-1, :]
            input_ids_shifted = input_ids[:, 1:].to(self.device)
            per_token_logps = get_per_token_logps(logits, input_ids_shifted)
            per_token_logps = per_token_logps[:, prompt_length-1:]
        return per_token_logps.cpu()
    
    def GRPO_step(self, batch):
        """
        执行一个GRPO训练步
        Args:
            batch: dict containing:
                - 'inputs': (B, L) tokenized input
                - 'prompt_length': int, prompt的长度
                - 'rewards': (B,) 奖励值（已经归一化）
                - 'ref_logps': (B, L_completion) 参考模型的logps
                - 'gen_logps': (B, L_completion) 生成时的logps（可选）
        Returns:
            loss: 标量损失值
        """
        prompt_length = batch['prompt_length']
        inputs = batch['inputs'].to(self.device)
        advantages = batch['rewards'].to(self.device).unsqueeze(1)  # (B, 1)
        
        # 前向传播
        logits = self.model(inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V)
        input_ids = inputs[:, 1:]  # (B, L-1)
        
        # 计算当前模型的logps
        per_token_logps = get_per_token_logps(logits, input_ids)
        per_token_logps = per_token_logps[:, prompt_length-1:]  # 只保留completion部分
        
        # 获取参考logps
        ref_per_token_logps = batch['ref_logps'].to(per_token_logps.device)
        
        # 计算KL散度
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # 计算completion mask（排除padding）
        completion_mask = (inputs[:, prompt_length:] != self.tokenizer.pad_token_id).int()
        
        # 计算损失
        if 'gen_logps' in batch and self.config.compute_gen_logps:
            # 使用PPO clipping
            gen_logps = batch['gen_logps'].to(per_token_logps.device)
            ratio = torch.exp(per_token_logps - gen_logps)
            clipped_ratio = torch.clamp(ratio, 1-self.config.clip_param, 1+self.config.clip_param)
            per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
        else:
            # 不使用clipping（类似REINFORCE）
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        
        # 添加KL惩罚并取负（因为是最大化问题）
        per_token_loss = -(per_token_loss - self.config.beta * per_token_kl)
        
        # 平均损失（只在completion部分计算）
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        
        return loss
    
    def train_step(self, batch):
        """
        执行一个训练步（包括前向、反向、更新）
        """
        loss = self.GRPO_step(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def save_model(self, save_path):
        """
        保存模型
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")


# ==================== 训练循环 ====================
def prepare_batch(grpo_model: GRPOModel, inputs):
    """
    准备一个训练批次
    """
    prompt_inputs, output_ids, rewards, answers = grpo_model.gen_samples(inputs)
    
    if prompt_inputs is None:
        return None
    
    # 归一化奖励
    if rewards.max() - rewards.min() < 1e-4:
        return None
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    
    # 合并prompt和output
    rep = output_ids.shape[0] // prompt_inputs.shape[0]
    prompt_length = prompt_inputs.shape[1]
    Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
    merged_ids = torch.cat([Qrep, output_ids], dim=1)
    
    # 计算参考logps
    ref_logps = grpo_model.compute_ref_logps(merged_ids, prompt_length)
    
    batch = {
        'inputs': merged_ids,
        'prompt_length': prompt_length,
        'rewards': rewards,
        'ref_logps': ref_logps,
    }
    
    # 可选：计算生成时的logps
    if grpo_model.config.compute_gen_logps:
        gen_logps = grpo_model.compute_gen_logps(merged_ids, prompt_length)
        batch['gen_logps'] = gen_logps
    
    return batch


def train(grpo_model: GRPOModel, dataset, config: GRPOConfig):
    """
    训练函数
    Args:
        grpo_model: GRPO模型实例
        dataset: 数据集，list of dict with keys 'Q' and 'A'
        config: 配置对象
    """
    print("开始训练...")
    progress = tqdm(range(1, config.all_steps + 1))
    
    for step in progress:
        # 生成batch
        # 这里简化处理，实际可以从数据集中采样
        inputs = random.sample(dataset, config.batch_size)
        batch = prepare_batch(grpo_model, inputs)
        
        # 如果batch生成失败，跳过
        if batch is None:
            continue
        
        # 训练一步
        loss = grpo_model.train_step(batch)
        
        # 更新进度条
        progress.set_description(f"Loss: {loss:.6f}")
        
        # 保存模型
        if step % config.save_steps == 0:
            save_path = f"./checkpoints/step_{step}"
            os.makedirs(save_path, exist_ok=True)
            grpo_model.save_model(save_path)


# ==================== 主函数 ====================
if __name__ == '__main__':
    # 初始化配置
    config = GRPOConfig()
    
    # 初始化模型
    grpo_model = GRPOModel(config)
    
    # 准备数据（这里需要根据实际情况加载数据）
    # 示例：使用GSM8K数据集（需要安装datasets库）
    try:
        from datasets import load_dataset
        dataset = load_dataset("openai/gsm8k", "main", split="train[:100]")  # 只用前100条作为示例
        QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(dataset['question'], dataset['answer'])]
        print(f"加载了 {len(QAs)} 条数据")
    except ImportError:
        print("警告: 未安装datasets库，使用示例数据")
        QAs = [
            {'Q': 'What is 2+2?', 'A': '4'},
            {'Q': 'What is 3*3?', 'A': '9'},
        ]
    
    # 开始训练
    train(grpo_model, QAs, config)

