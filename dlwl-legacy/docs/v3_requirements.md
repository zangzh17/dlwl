# DOE Optimizer v3.0 需求文档

## 1. 目标

为 DOE Optimizer 后端添加一个简单的 Web 测试前端，用于：
- 方便调整各种参数组合进行测试
- 验证后端功能完整性
- 模拟生产环境的显示和流程
- 全部使用 Plotly 可视化（替代现有 matplotlib）

核心原则：**功能完整、流程清晰、实现从简**

---

## 2. 用户流程

### 2.0 全局公共参数

- 用户设置
  - wavelength：中心工作波长（米），默认 532e-9
  - device_shape: 'square' 或 'circular'
  - device_diameter：DOE 直径（米）
- 内部参数（不需要提供用户GUI）
  - 最大允许仿真像素数（默认2000x2000）
  - pixel_size：DOE像素尺寸（米，默认1e-6）
  - ASM/FSR传播模式优化时的仿真面积增加比例，防止图案出现在仿真区域边缘，默认10%
  - 迭代更新频率，默认为50次更新一次进度


### 2.1 Wizard 输入

用户首先选择 DOE 类型，然后填写对应的参数：

**Splitter (1D/2D)**
- num_spots: 光斑数量（1D为整数，2D为 [ny, nx]）
- target_type: 'angle' 或 'size'
- target_span: 目标范围（deg或米）
- 如选择'size'，则需要给定工作距离 working_distance
- 如选择'size'，则出现一个选框，选定则用Strategy 2的周期化方法
- grid_mode: 'natural' 或 'uniform'
- tolerance: 'uniform'模式的容差（百分比）

**Diffuser**
- shape: 'square' 或 'circular'
- target_type: 'angle' 或 'size'
- 如选择'size'，则需要给定工作距离 working_distance
- target_span: 扩散范围

**Lens / Lens Array**
- focal_length: 焦距（mm）
- lens_type: 'normal', 'cylindrical_x', 'cylindrical_y'
- array_size: 透镜阵列尺寸 [ny, nx]（仅 lens array）

**Custom Pattern**
- image_data: 图片路径或 base64
- target_resolution: 目标分辨率 [H, W]
- target_type / target_span

### 2.2 生成结构化参数

用户点击「确认」后，Wizard 将用户输入转换为结构化参数，显示给用户确认。用户也可以跳过Wizard，直接设置此处的结构化参数。

**传播模式**（自动选择）
- FFT: 无穷远或 Strategy 2（注意这里的“Strategy 2”要对应结构化参数中，以与wizard解耦）
- ASM: 有限距离，目标 ≤ 2×DOE
- SFR: 有限距离，目标 > 2×DOE

**计算值（可调）**
- period / period_pixels: 周期
- num_periods: 设备内周期数
- working_orders: 工作级次列表
- order_angles: 对应角度
- strategy: 选用的传播策略
- tolerance_limit: 可达到的最小容差（splitter）
- airy_radius: 艾里斑半径（lens）

**优化参数（可调）**
- phase_method: SGD / GS / BS
- phase_lr: 学习率，默认 1e-8
- phase_iters: 迭代次数，默认 10000
- loss_type: L1 / L2 / focal_efficiency
- simulation_upsample: 仿真上采样倍数
- phase_pixel_multiplier: 优化分辨率倍增

### 2.3 参数验证

执行 validation，结果分为：

**错误（阻止优化）**
- wavelength/pixel_size/device_diameter ≤ 0
- pixel_size > device_diameter
- target_type='size' 但 working_distance=null
- working_distance < 0
- num_spots ≤ 0 或过大
- target_span ≤ 0
- 分辨率超出计算限制

**警告（允许优化，提示用户）**
- pixel_size > λ/2（可能产生混叠）
- 周期数过少（效率可能降低）
- ASM 模式下 Fresnel 数 < 1（应考虑 SFR）
- tolerance 比可达极限更严格

验证通过后自动更新预览，由于预览对应的是结构化参数，不应该和wizard关联。具体包括两个预览图：

**几何示意图**
- 用类似SVG的矢量图，画出来侧视图
- DOE 轮廓和尺寸标注
- 目标图案位置/角度示意
- 工作距离标注

**目标图**
- 通过一个toggle进行两种模式的选择
  - 各级次位置的散点，对应展示其角度或实际位置
  - 直接显示目标2D图案，坐标轴采用角度或实际位置

### 2.5 执行优化

点击「开始优化」后：

**实时显示**
- 进度条：当前迭代 / 总迭代
- 预计剩余时间
- 实时 Loss 曲线（每 50 次迭代更新，对数坐标；优化结束后依旧显示）
- 中断按钮

**中断处理**
- 点击中断后等待当前迭代完成
- 返回当前最佳结果

### 2.6 结果展示

优化完成后，基于优化结果进行一次仿真（基于结构化参数而非wizard信息来重新仿真，同样为了解耦）进行分析，显示分析图表。

**基础图表（始终显示）**
- 相位分布图（单周期 / 完整Device；注意如发现是1D情况，则需要改可视化方式）
- 目标强度 vs 仿真强度（可切换线性/对数强度显示）
- Loss 曲线

**可选图表（可以由用户勾选显示/隐藏特定图表）**
- 各级次效率柱状图（带均值参考线；仅当目标分布中非零级次小于200时可用;有限远的情况用目标分布中的所有峰值附近的airy圈入效率）
- 角度/位置分布散点图（颜色表示效率）
- 焦点剖面图（目标分布中的所有峰值附近）

**汇总指标**
- total_efficiency: 总效率
- uniformity: 均匀性
- mean_efficiency / std_efficiency

### 2.7 Fabrication Optimization（可选下一步）

在结果分析完成后，可以选择进入 2-step fabrication optimization：
- 加载当前优化的目标相位
- 配置 fabrication 参数（laser power curve，材料折射率 等）
- 执行 fabrication 优化
- 显示 fabrication 结果

（此功能为扩展，v3.0 可先预留入口）

---

## 3. 技术实现

### 3.1 目录结构

```
web_frontend/
├── backend/
│   ├── app.py              # FastAPI 入口
│   ├── routes/
│   │   ├── optimize.py     # 优化相关接口
│   │   ├── validate.py     # 验证接口
│   │   └── preview.py      # 预览接口
│   └── services/
│       └── optimizer.py    # 封装 doe_optimizer 调用
├── frontend/
│   ├── index.html
│   ├── css/styles.css
│   └── js/
│       ├── app.js          # 主逻辑
│       ├── wizard.js       # Wizard 面板
│       ├── preview.js      # 预览渲染
│       ├── optimizer.js    # 优化控制
│       └── results.js      # 结果显示
└── requirements.txt
```

### 3.2 API 接口

```
POST /api/validate     # 验证用户输入，返回 ValidationResult
POST /api/preview      # 生成预览数据（WizardOutput 的子集）
POST /api/optimize     # 启动优化，返回 task_id
GET  /api/status/{id}  # 查询进度
POST /api/cancel/{id}  # 取消优化
GET  /api/result/{id}  # 获取结果
POST /api/export/{id}  # 导出文件
```

### 3.3 实时通信

使用 WebSocket 推送优化进度：

```
ws://localhost:8000/ws/optimize/{task_id}

# 推送消息格式
{
  "iteration": 5000,
  "total": 10000,
  "loss": 0.00123,
  "best_loss": 0.00098,
  "eta_seconds": 45.2
}
```

### 3.4 技术栈

- 后端: FastAPI + WebSocket
- 前端: 原生 HTML/CSS/JS + Plotly.js
- 任务管理: 内存队列（简单实现）

### 3.5 与核心库的隔离

- 前端仅通过 JSON API 与后端通信
- 后端 services 层封装 doe_optimizer 调用
- 复用 doe_optimizer.visualization 中已有的 Plotly 数据结构

---

## 4. 附注

现有代码有的地方还不完整，相关部分可尝试实现完成，保持简单，便于快速迭代和测试。我准备通过构建此网页前端，一切都为了加速测试过程，边测试边添加功能或补全，逐步完善。另外目前代码冗余的部分（不限于matplotlib可视化的部分等等），可在重构后直接删除。

