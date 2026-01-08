# Part A: Python 服务开发

## 1. 推荐架构

```
FastAPI (HTTP API)  ←→  Redis (任务队列)  ←→  Worker (GPU计算)
```

**为什么选择这个架构？**
- 优化任务耗时 2-10 分钟，需要异步处理
- Redis 队列支持任务持久化、进度更新、水平扩展

## 2. 目录结构

```
python_service/
├── main.py                 # FastAPI 入口
├── worker.py               # 任务消费者
├── config.py               # 配置
├── algorithms/
│   ├── base.py             # 优化器基类（必须实现）
│   ├── diffuser.py         # 各模式优化器
│   ├── spot_projector.py
│   └── utils.py            # FFT、传播等工具函数
├── schemas/
│   ├── request.py          # 请求模型
│   └── response.py         # 响应模型
├── requirements.txt
└── Dockerfile
```

## 3. 依赖

```
fastapi>=0.104.0
uvicorn>=0.24.0
redis>=5.0.0
pydantic>=2.5.0
pydantic-settings>=2.0.0
torch>=2.1.0
numpy>=1.24.0
scipy>=1.11.0
```

## 4. API 端点设计

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/optimize` | 提交优化任务，返回 task_id |
| GET | `/api/v1/optimize/{task_id}/status` | 查询任务状态和进度 |
| POST | `/api/v1/optimize/{task_id}/cancel` | 取消任务 |
| GET | `/api/v1/health` | 健康检查（含GPU状态） |

## 5. 数据模型

### 5.1 请求模型 (schemas/request.py)

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

class DOEMode(str, Enum):
    DIFFUSER = "diffuser"
    SPLITTER_1D = "1d_splitter"
    SPOT_PROJECTOR_2D = "2d_spot_projector"
    LENS = "lens"
    # ... 其他模式

class BasicParams(BaseModel):
    wavelength_nm: float
    device_diameter_mm: float
    device_shape: Literal["circular", "square"] = "circular"
    working_distance_mm: Optional[float] = None  # None = 无穷远
    mode: DOEMode

class OptimizationRequest(BaseModel):
    task_id: str
    design_id: int
    user_id: str
    basic: BasicParams
    mode_params: dict  # 模式特定参数
    max_iterations: int = Field(100, ge=10, le=10000)
    convergence_threshold: float = 1e-6
```

### 5.2 响应模型 (schemas/response.py)

```python
from pydantic import BaseModel
from typing import List, Optional, Literal

class OrderEnergy(BaseModel):
    order: str      # 必须是字符串: "-2", "0", "2"
    energy: float

class EfficiencyMetrics(BaseModel):
    totalEfficiency: float       # 0-1
    uniformityError: float       # 0-1
    zerothOrderLeakage: float    # 0-1

class OptimizationResult(BaseModel):
    phaseMap: List[List[float]]         # 0-255
    targetIntensity: List[List[float]]
    actualIntensity: List[List[float]]
    orderEnergies: List[OrderEnergy]
    efficiency: EfficiencyMetrics

class TaskStatus(BaseModel):
    task_id: str
    status: Literal["pending", "processing", "completed", "failed", "cancelled"]
    progress: float  # 0-100
    current_iteration: Optional[int] = None
    total_iterations: Optional[int] = None
    error_message: Optional[str] = None
    result: Optional[OptimizationResult] = None
```

## 6. 优化器基类模板 (algorithms/base.py)

```python
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional

class BaseOptimizer(ABC):
    """
    优化器基类 - 所有模式的优化器必须继承此类

    必须实现的方法:
    - initialize(): 初始化相位图和目标
    - step(): 执行一次迭代，返回误差
    - compute_efficiency(): 计算效率指标
    - compute_far_field(): 计算远场强度
    - compute_order_energies(): 计算衍射阶能量
    """

    def __init__(
        self,
        wavelength_nm: float,
        device_diameter_mm: float,
        device_shape: str = "circular",
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
    ):
        self.wavelength = wavelength_nm * 1e-9
        self.device_diameter = device_diameter_mm * 1e-3
        self.device_shape = device_shape
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.phase_map: Optional[torch.Tensor] = None
        self.target_intensity: Optional[torch.Tensor] = None

    @abstractmethod
    def initialize(self) -> None:
        """初始化相位图和目标强度分布"""
        pass

    @abstractmethod
    def step(self) -> float:
        """执行一次迭代，返回当前误差"""
        pass

    @abstractmethod
    def compute_efficiency(self) -> dict:
        """返回 {totalEfficiency, uniformityError, zerothOrderLeakage}"""
        pass

    @abstractmethod
    def compute_far_field(self) -> torch.Tensor:
        """计算远场强度分布"""
        pass

    @abstractmethod
    def compute_order_energies(self) -> list:
        """返回 [{order: str, energy: float}, ...]"""
        pass

    def run(self, progress_callback: Optional[Callable] = None) -> dict:
        """运行优化，返回结果字典"""
        self.initialize()

        for i in range(self.max_iterations):
            error = self.step()
            if progress_callback:
                progress_callback(i + 1, self.max_iterations, error)
            if error < self.convergence_threshold:
                break

        return self._format_result()

    def _format_result(self) -> dict:
        """格式化结果 - 确保与前端接口匹配"""
        phase = self.phase_map.cpu().numpy()
        phase_8bit = ((phase / (2 * np.pi)) * 255).astype(np.uint8)

        return {
            "phaseMap": phase_8bit.tolist(),
            "targetIntensity": self._normalize(self.target_intensity).tolist(),
            "actualIntensity": self._normalize(self.compute_far_field()).tolist(),
            "orderEnergies": self.compute_order_energies(),
            "efficiency": self.compute_efficiency(),
        }

    def _normalize(self, arr: torch.Tensor) -> np.ndarray:
        arr = arr.cpu().numpy()
        return arr / arr.max() if arr.max() > 0 else arr
```

## 7. Worker 模板 (worker.py)

```python
import redis
import json
from config import settings

class OptimizationWorker:
    def __init__(self):
        self.redis = redis.from_url(settings.redis_url)

    def run(self):
        """主循环 - 阻塞等待任务"""
        while True:
            result = self.redis.brpop(settings.queue_name, timeout=5)
            if result:
                _, task_data = result
                self.process_task(json.loads(task_data))

    def process_task(self, task: dict):
        task_id = task["task_id"]
        try:
            self._update_status(task_id, "processing", 0)
            optimizer = self._create_optimizer(task)
            result = optimizer.run(
                progress_callback=lambda cur, total, err:
                    self._update_status(task_id, "processing", cur/total*100, cur, total)
            )
            self._save_result(task_id, result)
            self._update_status(task_id, "completed", 100)
        except Exception as e:
            self._update_status(task_id, "failed", 0, error_message=str(e))

    def _create_optimizer(self, task: dict):
        """根据 mode 创建对应的优化器实例"""
        # 实现工厂逻辑
        pass

    def _update_status(self, task_id, status, progress, cur=None, total=None, error_message=None):
        data = {"task_id": task_id, "status": status, "progress": progress,
                "current_iteration": cur, "total_iterations": total, "error_message": error_message}
        self.redis.setex(f"{settings.status_prefix}{task_id}", settings.task_ttl, json.dumps(data))
        self.redis.publish(f"doe:optimization:progress:{task_id}", json.dumps(data))

    def _save_result(self, task_id, result):
        self.redis.setex(f"{settings.result_prefix}{task_id}", settings.task_ttl, json.dumps(result))
```

## 8. 本地开发启动

```bash
# 启动 Redis
docker run -d -p 6379:6379 redis:7-alpine

# 启动 API
uvicorn main:app --reload --port 8000

# 启动 Worker (另一个终端)
python worker.py

# 测试
curl http://localhost:8000/api/v1/health
```

## 9. 注意事项

### 数据格式要求

| 字段 | 格式 | 说明 |
|------|------|------|
| `phaseMap` | `List[List[int]]` | 值 0-255，表示 0 到 2π 相位 |
| `efficiency.*` | `float` | 必须在 0-1 范围内 |
| `orderEnergies[].order` | `str` | 字符串如 "-2", "0", "2" |

### GPU 内存管理

```python
# 每个任务结束后清理
torch.cuda.empty_cache()
```

### 大数据处理

- phaseMap 尺寸 > 512x512 时，建议上传到 S3，只返回 URL

### 错误分类

| 错误类型 | 可重试 | 说明 |
|----------|--------|------|
| `INVALID_PARAMETERS` | 否 | 参数校验失败 |
| `GPU_ERROR` | 是 | GPU 计算错误 |
| `CONVERGENCE_FAILED` | 是 | 未收敛 |
| `TIMEOUT` | 是 | 超时 |
