# Deterministic Seed Derivation（确定性随机种子）

对应实现：`ImagiChem code/imagichem_core.py` 中的 `image_to_seed` 与 `set_seed`。

**公式预览**：若使用 Cursor / VS Code 预览，请开启 **`Markdown › Math: Enabled`**。

---

## 作用（一句话）

用整张图解码后的**灰度像素字节**做一次 **SHA-256**，再折成 32 位整数作为种子；同一张图每次运行得到**相同**种子，从而后续随机选 core、挂片段等结果**可复现**。

---

## 伪代码（中文）

### 从图像得到种子

```
算法：从整张图确定性地得到一个随机数种子

输入：图像文件路径 path
输出：32 位整数 seed

1. 以二进制只读方式打开 path，读出全部字节
2. 用图像解码器（OpenCV imdecode）将字节解码为灰度矩阵 img
   若解码失败 → 报错（文件不存在或无法读取）
3. 将 img 在内存中的像素按存储布局展开为字节流 B
   （与 decode 后的灰度矩阵内容一一对应，而非「文件原始字节」）
4. H ← SHA256(B)
5. 将 H 的十六进制表示解析为大整数 N
6. seed ← N mod 2^32
7. 返回 seed
```

### 用种子固定所有随机源

`run_imagichem_processing` 在得到 `image_seed` 后会调用 `set_seed`：

```
算法：用 seed 固定随机源

输入：整数 seed

1. random.seed(seed)          // Python 标准库
2. np.random.seed(seed)       // NumPy
3. RNG.seed(seed)             // graph_utils 中用于组装的随机数发生器
```

---

## Pseudocode (English)

### Derive seed from image

```python
Algorithm: DeriveDeterministicSeedFromImage

Input:  image file path
Output: 32-bit integer seed

1. Open path in binary mode and read all bytes from the file.
2. Decode those bytes into a grayscale image matrix img.
3. Flatten img to a byte sequence B in memory layout order.
4. H ← SHA256(B)
5. Parse the hexadecimal digest of H as a big integer N.
6. seed ← N mod 2^32
7. Return seed
```

### Fix all random sources with the seed

After `image_seed` is obtained, `run_imagichem_processing` calls `set_seed`:

```
Algorithm: SetAllRandomSeeds

Input:  integer seed

1. random.seed(seed)     // Python standard library
2. np.random.seed(seed)  // NumPy
3. RNG.seed(seed)        // assembly RNG in graph_utils
```

---

## 与代码的对应关系

| 步骤 | 代码 |
|------|------|
| 读文件 + 解码灰度 | `open` + `np.fromfile` / `cv2.imdecode(..., IMREAD_GRAYSCALE)` |
| 字节流 $B$ | `img.tobytes()` |
| $H$ | `hashlib.sha256(...).hexdigest()` |
| $N \bmod 2^{32}$ | `int(h, 16) % (2**32)` |

---

## 性质说明

- **确定性**：解码得到的矩阵相同 → $B$ 相同 → SHA-256 相同 → `seed` 相同。
- **哈希对象**：实现里是对**解码后的灰度像素**做哈希，不是直接对压缩文件字节做哈希；若两种文件格式解压后像素完全一致，会得到相同种子。
- **非机器学习**：纯哈希与取余，无训练、无可学习参数。

---

## 大白话版（便于向非程序员解释）

同一张照片，程序会先给像素算一个固定长度的「指纹」（SHA-256），再从这个指纹推出一个数字当随机种子。这样**同一张图多次运行，随机过程也一致**；换图则指纹变、种子变，后面随机组装也会变。

---

## 相关文件

- `ImagiChem code/imagichem_core.py`：`image_to_seed`、`set_seed`、`run_imagichem_processing`
