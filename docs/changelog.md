# 变更日志

## v3.9 (2025-01-14) - 修复2D Splitter ASM模式target_size问题

### 问题
2D Splitter wizard在ASM模式下，target pattern尺寸和图案不正确。
- 输入target span [0.5,0.5]mm，但生成的target pattern只有256×256像素
- 应该生成551×551像素（含10% margin：0.55mm / 1um ≈ 550）

### 原因
`_create_asm_params()`传递`target_pixels`给ASMParams构造函数，但ASMParams在`__post_init__`中会覆盖该值（从`target_size`计算）。由于`target_size`未传递，默认为DOE尺寸，导致输出分辨率错误。

### 修复
`splitter.py`: `_create_asm_params()`改为传递`target_size`（物理尺寸，米），而非`target_pixels`。ASMParams自动计算正确的`target_pixels`和`target_resolution`。

---

## v3.8 (2025-01-13) - 优化页面显示GPU状态

### 功能
优化进度页面现在显示当前使用的计算设备：
- GPU模式：显示GPU型号和显存大小（绿色背景）
- CPU模式：提示CUDA不可用（橙色背景）

### 修改文件
- `app.py`: 添加`/api/system-info`端点
- `index.html`: 添加device-info-bar元素
- `optimizer.js`: 添加`fetchDeviceInfo()`函数
- `styles.css`: 添加`.device-info-bar`样式

---

## v3.7 (2025-01-13) - 修复ASM Analysis Upsample

### 问题
ASM模式下Analysis Upsample 2x后，Simulated图案与Target不匹配（类似SFR之前的问题）。

### 原因
`_reevaluate_asm()`未传递`output_size`和`output_resolution`给`propagation_ASM()`，导致重评估时输出尺寸与原优化不一致。

### 修复
- `reevaluate.py`: `_reevaluate_asm()`添加`target_size`参数，传递给`propagation_ASM()`
- `task_manager.py`: ASM也使用与SFR相同的`target_size`计算逻辑（含margin）

---

## v3.6 (2025-01-13) - ASM传播支持可配置输出尺寸

### 问题
ASM优化点击后失败，报错"tensor size mismatch (1000 vs 256)"。

### 原因
ASM传播器输出尺寸由`target_span/pixel_size`决定（如1mm/1um=1000像素），但target_pattern为256×256，导致loss计算时尺寸不匹配。

### 修复
为ASM添加`output_size`和`output_resolution`参数（与SFR接口一致）：

| Case | 条件 | 处理方式 |
|------|------|---------|
| A | output_size > DOE_size | 零填充后FFT，裁剪到output_size，重采样到output_resolution |
| B | output_size ≤ DOE_size | 直接FFT，裁剪到output_size，重采样到output_resolution |

**修改文件**：
- `doe_optimizer/core/propagation.py`: `propagation_ASM()`添加output_size/output_resolution参数
- `doe_optimizer/params/asm_params.py`: ASMParams添加target_size/target_resolution属性
- `doe_optimizer/core/propagator_factory.py`: PropagatorBuilder传递output_size给ASM
- `doe_optimizer/pipeline/runner.py`: 创建ASMParams时使用target_size

**测试**：
- `tests/test_asm_propagation.py`: 覆盖Case A/B、奇偶像素、接口一致性
- Web界面ASM优化流程验证通过

---

## v3.5 (2025-01-13) - 修复SFR Analysis Upsample参数不一致

### 问题
SFR模式下Analysis Upsample 2x/4x后，Simulated图案与Target不匹配（1x正常）。

### 原因
原优化使用 `target_size = target_span × 1.1` (含margin)，但re-evaluation只用了 `target_span`。

### 修复
`task_manager.py`: 添加margin_factor计算，使target_size与原优化一致。

---

## v3.4 (2025-01-13) - 修复SFR/ASM Analysis Upsample实现

### 问题
Analysis Upsample功能在SFR模式下不工作（1x正常，2x/4x后仿真结果错误）。

### 原因
**理解错误**：原实现增加了output_resolution（更多输出采样），正确做法是对输入相位上采样。

| 方面 | 错误 | 正确 |
|------|-----|------|
| 输入相位 | 原始 | nearest-neighbor上采样 |
| pixel_size | 不变 | `pixel_size / k` |
| 输出分辨率 | 增加 | **不变** |

### 修复
`reevaluate.py`: `_reevaluate_sfr()` 对输入相位上采样，使用更小的feature_size，保持原始output_resolution。

---

## v3.3 (2025-01-13) - 统一re-evaluation模块

新增 `doe_optimizer/evaluation/reevaluate.py`，为所有传播类型提供统一的高分辨率重评估接口。

---

## v3.2 - 解耦Wizards与Core Pipeline

效率计算现在从`target_pattern`自动提取target_indices，不再依赖wizard元数据。

---

## v3.1 - Strategy 2 (Periodic+Fresnel) 支持

添加有限距离周期DOE传播方案。

---

## v3.0 - Web Frontend

添加FastAPI后端 + Plotly前端的交互式测试界面。

---

## 历史版本 (v2.x)

详见代码注释和git历史。主要包括：
- Splitter仿真与优化流程
- ZoomFFT2修复
- 1D Splitter支持
- 上采样参数支持
