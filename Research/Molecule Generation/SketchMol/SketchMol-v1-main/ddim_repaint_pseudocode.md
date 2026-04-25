# DDIM + RePaint 伪代码（z 记号）

```python
z_t = q_sample(z_0, t)                                   # 原图在 t 步噪声
z_t_with_mask = z_t * mask + z_t_gen * (1 - mask)       # 先做 mask 融合（保留区锁住）
pred_noise = model(z_t_with_mask, cond, t)              # 全图预测噪声
z_t_minus_1_gen = ddim_update(z_t_with_mask, pred_noise)# 全图更新
# 下一轮/下一步又会再次 mask 融合，所以保留区持续被约束
```
