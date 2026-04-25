# SketchMol 项目摘录：随机种子与图像预处理

本文档从本仓库代码中整理**与随机种子固定**以及**灰度 / 归一化预处理**相关的实现位置与行为，便于单独学习或复用。

**扩展**：

- MolScribe 把**归一化后的 2D 坐标**量化成离散 ID，并与 **SMILES 字符词表**拼成同一条序列，见 [`learn/molscribe_pixel_to_vocab.md`](./molscribe_pixel_to_vocab.md)。
- **行均值 / 行标准差 / 对比度 / 峰频 → C、N、S 与分子规模** 这类显式公式**不在本仓库实现**；若你关心该理论对照与和 SketchMol 数据流的关系，见 [`learn/image_stats_to_composition.md`](./image_stats_to_composition.md)。

---

## 1. 随机种子（确定性）

### 1.1 主训练入口 `main.py`

- **实际生效逻辑**：每次启动会用 `torch.randint` 采一个 32 位整数，再调用 PyTorch Lightning 的 `seed_everything`。
- **命令行参数**：解析器里定义了 `-s` / `--seed`（默认 `23`，说明文字为 “seed for seed_everything”），但**当前源码中未将 `opt.seed` 传给 `seed_everything`**，因此 CLI 种子在默认代码路径下**不会**固定为你传入的值。

相关代码位置：

```514:515:main.py
    random_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    seed_everything(random_seed)
```

若需要**完全可复现**的训练，应改为使用固定种子（例如直接使用 `opt.seed` 或常量），并注意多 worker、CUDA、cudnn 等仍可能带来残余非确定性。

### 1.2 DataLoader `worker_init_fn`

在 `main.py` 中，当配置使用 `worker_init_fn` 时，会为每个子进程基于当前 NumPy 随机状态与 `worker_id` 再设 `np.random.seed`，用于多进程加载时的随机性拆分。

```147:160:main.py
def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)
```

### 1.3 MolScribe 工具函数 `seed_torch`（`evaluate/molscribe/utils.py`）

仓库内提供了一套较完整的 PyTorch 侧种子设置（含 `PYTHONHASHSEED`、`cudnn.deterministic`）。默认种子为 `42`。

```58:64:evaluate/molscribe/utils.py
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```

说明：在本仓库的 `evaluate/` 下未检索到对 `seed_torch` 的调用；若你在自己的训练或推理脚本里需要确定性，可直接复用该函数或在入口显式调用。

### 1.4 采样脚本

`scripts/` 下的扩散采样脚本（如 `sample_diffusion_condition_continuousV2.py`）未统一封装种子设置；若要对采样过程固定随机性，需在脚本入口自行设置 `torch.manual_seed` / `numpy.random.seed` 等，并关注 DDIM 等步骤中的随机源。

---

## 2. 灰度与图像预处理

本项目中**不同子任务**的预处理不一致，需按用途区分。

### 2.1 扩散模型训练数据：`ldm/data/pubchemdata.py`

`pubchemBase` / `pubchemBase_RL` / `pubchemBase_various_continuousV2` 等数据集的典型流程为：

- 读图后若非 RGB 则 `convert("RGB")`；
- 取**中心正方形**裁剪；
- 按配置 `resize` 到目标边长；
- （部分类）随机水平翻转；
- 像素归一化：`(image / 127.5 - 1.0)`，即映射到约 **[-1, 1]** 的 float32。

这里**不做显式灰度化**，输入仍为三通道（与 LDM 常见设定一致）。代码中注释也提到与 score-SDE 类预处理相关。

### 2.2 MolScribe（结构图识别）：`evaluate/molscribe/dataset.py`

`get_transforms` 在**非 debug** 模式下，在 Albumentations 流水线末尾固定执行：

1. **`A.ToGray(p=1)`**：将图像转为单通道灰度（再经后续步骤以张量形式进入网络）；
2. **`A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`**：使用 **ImageNet** 的均值与标准差（与常见 RGB 预训练一致；灰度后三通道统计量仍被用作归一化参数，属于该项目/MolScribe 沿用设定）；
3. **`ToTensorV2()`**。

```36:60:evaluate/molscribe/dataset.py
def get_transforms(input_size, augment=True, rotate=True, debug=False):
    trans_list = []
    if augment and rotate:
        trans_list.append(SafeRotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    trans_list.append(CropWhite(pad=5))
    if augment:
        trans_list += [
            # NormalizedGridDistortion(num_steps=10, distort_limit=0.3),
            A.CropAndPad(percent=[-0.01, 0.00], keep_size=False, p=0.5),
            PadWhite(pad_ratio=0.4, p=0.2),
            A.Downscale(scale_min=0.2, scale_max=0.5, interpolation=3),
            A.Blur(),
            A.GaussNoise(),
            SaltAndPepperNoise(num_dots=20, p=0.5)
        ]
    trans_list.append(A.Resize(input_size, input_size))
    if not debug:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans_list += [
            A.ToGray(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(trans_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
```

推理从文件读图时，在 `evaluate/molscribe/interface.py` 中通常用 OpenCV 读入 BGR 再转为 RGB，再交给上述 `transform`。

### 2.3 分子图 inpaint / 掩膜：`scripts/inpaint_continuousV2.py`

为根据线条生成掩膜，使用 **OpenCV 灰度化 + 固定阈值二值化**（例如阈值 250），再做膨胀、轮廓等形态学处理。这是**分割/掩膜构造**用的灰度流程，与 MolScribe 的 Albumentations 训练预处理不同。

### 2.4 数据管线其它灰度用法

- **`data_process/mp_convert_valid_to_invalid.py`**：将 PIL 图像转为灰度 `L` 后送入 LSD 线段检测，用于无效样本相关处理。
- **`ldm/modules/image_degradation/`**：BSRGAN 等退化模型内部含 BGR→灰度、灰度噪声等，属于**数据增强/退化仿真**，与上文「读入用户分子图」的前处理是不同层级。

---

## 3. 快速对照表

| 模块 | 是否转灰度 | 数值范围 / 归一化 |
|------|------------|-------------------|
| `pubchemdata`（扩散训练） | 否（保持 RGB） | `pixel/127.5 - 1` → 约 [-1, 1] |
| `evaluate/molscribe/dataset.py` | 是（`ToGray`） | ImageNet mean/std + Tensor |
| `inpaint_continuousV2`（掩膜） | 是（`cvtColor` GRAY + threshold） | OpenCV uint8 二值逻辑 |
| 主程序种子 | `seed_everything` | 当前为**随机**整数，非 CLI `--seed` |

---

## 4. 复用建议

- **只关心扩散训练**：以 `pubchemdata.py` 的裁剪、resize、`/127.5-1` 为准；种子需自行改为固定值若要对齐论文/实验。
- **只关心 MolScribe 式输入**：复制 `get_transforms` 中非 debug 分支的 `ToGray` + `Normalize` + `ToTensorV2`，并与 `input_size`、增广开关一致。
- **需要严格可复现**：除设置全局种子外，建议查阅 PyTorch 文档中关于 `cudnn.benchmark`、`deterministic` 与多 GPU 的说明，并评估 `worker_init_fn` 与数据顺序的影响。

以上均直接对应本仓库现有实现，未包含对上游 LDM / MolScribe 论文的额外假设。
