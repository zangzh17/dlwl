# 变更日志

## 2024-12-30: 上采样支持与1D可视化改进

### 新增功能

#### 1. 仿真上采样因子 (simulation_upsample)

添加两个上采样参数：

```python
optimization = OptimizationParams(
    simulation_upsample=2,  # 优化时的上采样因子
    eval_upsample=2,        # 评估时的上采样因子
)
```

**作用**:
- 优化时将相位上采样后再进行传播仿真
- 更高的采样率可以获得更准确的仿真结果
- 2x上采样通常足够大多数应用

#### 2. 1D可视化改进

修复`plot_1d_splitter_result()`函数:
- 强度柱状图不再使用colormap（避免重复表达，已用bar高度表示强度）
- 器件相位显示为2D图像（平铺显示周期性）
- 保留单周期相位的折线图

### 测试更新

新增测试案例:
- **Case 11**: simulation_upsample效果对比 (1x, 2x)

所有测试案例均保存图片到`splitter_results/`目录。

### 相关文件

| 文件 | 修改内容 |
|------|----------|
| `doe_optimizer/core/config.py` | 添加upsample参数 |
| `doe_optimizer/core/optimizer.py` | SGDOptimizer支持上采样 |
| `doe_optimizer/utils/visualization.py` | 1D可视化修复 |
| `doe_optimizer/pipeline/two_step.py` | 传递upsample参数给优化器 |
| `test_splitter.py` | 添加Case 11 |

### 注意事项

关于效率权重参数的考虑：由于仿真区域的`total_intensity`可能不代表真实总能量（尤其是ASM等方法，落在仿真区域外的能量未计入），因此未实现基于绝对效率的优化。当前loss函数仅优化相对效率（均匀度）。

---

## 2024-12-30: Splitter仿真与优化流程修复

### 当前方案

#### 1. 优化分辨率设计

**核心原则**: 优化分辨率 = period_pixels（物理周期的像素数）

```python
# config.py: get_splitter_resolution()
period_pixels = int(round(period / pixel_size))
# 2D splitter
return (period_pixels, period_pixels)
# 1D splitter
return (period_pixels, 1)
```

**设计理由**：
- FFT有N个输入像素，就有N个输出频率点（衍射级次）
- 优化在period_pixels分辨率上进行，无需非整数重采样
- 评估和加工使用同一相位，结果一致

#### 2. 器件相位生成

```
优化相位 (period_pixels × period_pixels 或 period_pixels × 1)
    ↓ 直接平铺（无重采样）
器件相位 (device_resolution × device_resolution)
    ↓ + Fresnel相位（Strategy 2）
最终器件相位
```

#### 3. ZoomFFT2修复

**问题**: 原实现对奇数尺寸数组有数值问题

**原因**: 起始频率公式 `a = -0.5` 仅对偶数N正确

**修复**:
```python
# 正确的起始频率（适用于奇偶N）
ax = 2 * np.pi * (-(num_x // 2) / num_x) * sx
ay = 2 * np.pi * (-(num_y // 2) / num_y) * sy
```

**归一化修复**: 使用标准FFT归一化代替ortho，在cfft2中添加ortho归一化因子

#### 4. 1D Splitter支持

**修复内容**:
- `get_splitter_resolution()`: 1D返回 `(period_pixels, 1)`
- `get_splitter_params()`: 1D的order_positions中px固定为0
- `evaluation.py`: 处理squeeze后变成1D数组的情况
- `two_step.py`: 使用`.squeeze(0).squeeze(0)`保持2D形状

**1D可视化**:
- 新增`plot_1d_splitter_result()`函数
- 相位显示为折线图（非2D图像）
- 衍射效率按实际角度位置显示

### 测试结果

| 案例 | k-space均匀度 | SFR均匀度 |
|------|--------------|-----------|
| Case 3 (6x6, 100mm) | 0.9943 | 0.9943 |
| Case 5 (1D, 7 spots) | 0.9633 | - |
| Case 8 (1D, 50mm, 5 spots) | ~0.99 | - |

### 关键文件

| 文件 | 功能 |
|------|------|
| `doe_optimizer/core/config.py` | 分辨率和周期计算 |
| `doe_optimizer/pipeline/two_step.py` | 相位生成和平铺 |
| `doe_optimizer/pipeline/evaluation.py` | 评估函数（含1D支持）|
| `doe_optimizer/utils/fft_utils.py` | ZoomFFT2实现 |
| `doe_optimizer/utils/visualization.py` | 可视化（含1D支持）|
| `doe_optimizer/core/propagation.py` | 传播函数 |

### 注意事项

1. **ZoomFFT vs 标准FFT**: 当zoom=1.0时，propagation_SFR会自动使用标准FFT（更稳定）

2. **非整数平铺**: 器件尺寸不必是周期的整数倍，边界截断是真实物理行为

3. **num_orders vs period_pixels**: 这是两个不同概念
   - `num_orders`: 级次数（包括工作和非工作级次）
   - `period_pixels`: 物理周期的像素数（决定优化分辨率）

---

## 衍射效率分析

### 现象

大部分splitter测试案例的总衍射效率低于理论最大值（~0.7-0.9 vs 理论1.0）

### 原因分析

#### 1. Loss函数设计

当前loss函数使用L2归一化：
```python
# math_utils.py: compute_loss()
recon_norm = (recon ** 2).sum().sqrt()
target_norm = (target ** 2).sum().sqrt()
recon = recon / recon_norm * sqrt(sz)
target = target / target_norm * sqrt(sz)
```

**影响**:
- 优化的是**相对效率**（各级次的分布形状），而非**绝对效率**
- 非工作级次的能量损失不直接参与loss计算
- 这是传统DOE优化的常见做法，优先保证均匀度

#### 2. 传播算法能量守恒

| 传播方法 | 归一化 | 能量守恒 |
|---------|-------|---------|
| propagation_FFT | ortho | ✓ 满足Parseval定理 |
| propagation_ASM | ortho | ✓ |
| propagation_SFR | 自定义 | ✓ |

#### 3. 采样问题

当前实现中，每个DOE像素对应1个采样点（无上采样）。在实际加工仿真中：
- 通常需要2x或更高上采样因子
- `phase_pixel_multiplier`参数存在但主要用于工艺优化

**建议**：对于高精度仿真，可使用`phase_pixel_multiplier=2`

### 测试案例

参见`test_splitter.py`中的新测试：
- Case 9: tolerance参数影响测试
- Case 10: pixel_multiplier参数影响测试
- `analyze_efficiency()`: 能量分布详细分析

### 提高效率的方法

1. **添加效率惩罚项**: 在loss中加入非工作级次能量的惩罚
2. **使用绝对效率优化**: 不归一化，直接比较目标和重建的绝对值
3. **增加迭代次数**: 更长的优化可能找到更优解
4. **优化初始值**: 使用解析初始相位（如Dammann光栅）

---

## 参数影响

### tolerance参数

| tolerance | 周期(um) | 网格大小 | 均匀度 |
|-----------|---------|---------|--------|
| 2.0% | ~小 | 较小 | 较高 |
| 1.0% | 中等 | 中等 | 高 |
| 0.5% | ~大 | 较大 | 最高 |

**说明**: 较小的tolerance要求更密集的k空间采样，导致更大的周期

### pixel_multiplier参数

| multiplier | 相位形状 | 均匀度 | 计算时间 |
|------------|---------|--------|---------|
| 1x | period_pixels | 基准 | 快 |
| 2x | 2*period_pixels | 略高 | ~4x |

**说明**: 更高的multiplier提供更精细的相位采样，但增加计算开销
