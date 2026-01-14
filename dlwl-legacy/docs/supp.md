# Tolerance & Sampling Theory 具体理论关系

## 1. 变量定义 (Variable Definitions)
* $\lambda$: 工作波长 (Wavelength)
* $P$: 仿真/优化周期尺寸 (Simulation Period), $P \le D$
* $D$: 器件全孔径尺寸 (Full Aperture Size)
* $z$: 工作距离 (Working Distance)
* $T_{\%}$: 允许的误差公差百分比 (Tolerance Percentage)

## 2. 基础原理
根据傅里叶光学，空间域的周期化 ($P$) 导致频率域（$k$空间/角度域）的离散采样。

* **频率采样间隔**: $\Delta f = \frac{1}{P}$
* **最大量化误差 (Worst-case Error)**: 发生在采样网格中心，即间隔的一半。
    $$\epsilon_{max} = \frac{1}{2} \Delta f = \frac{1}{2P}$$

## 3. 无穷远/角度空间

假设目标图案的角度跨度范围（视场）为 $\Delta(\sin\theta)_{FOV}$。

* **绝对角度误差**: $\delta(\sin\theta) = \frac{\lambda}{2P}$
* **公差定义**: $T_{\%} = \frac{\delta(\sin\theta)}{\Delta(\sin\theta)_{FOV}}$
* **仿真周期 $P$ 约束**:
    $$P \ge \frac{\lambda}{2 \cdot T_{\%} \cdot \Delta(\sin\theta)_{FOV}}$$

### 理论下限
当仿真周期取最大物理尺寸时 ($P=D$)，获得物理衍射极限下的最小公差：

$$T_{\%}^{min} = \frac{\lambda}{2 D \cdot \Delta(\sin\theta)_{FOV}}$$

---

## 4. 有限工作距离

对于工作距离 $z$，通过叠加 Fresnel 透镜相位抵消二次项，衍射斑位置 $x$ 与空间频率 $f$ 的关系为 $x = \lambda z f$。
假设目标图案在像面的物理尺寸跨度为 $S_{FOV}$。

* **绝对位置误差**: $\delta x = \lambda z \cdot \epsilon_{max} = \frac{\lambda z}{2P}$
* **公差定义**: $T_{\%} = \frac{\delta x}{S_{FOV}}$
* **仿真周期 $P$ 约束**:
    $$P \ge \frac{\lambda z}{2 \cdot T_{\%} \cdot S_{FOV}}$$

### 理论下限
当仿真周期取最大物理尺寸时 ($P=D$)，获得物理衍射极限下的最小公差：

$$T_{\%}^{min} = \frac{\lambda z}{2 D \cdot S_{FOV}}$$

## 5. 总结

若用户设定的 $T_{\%}$ 小于上述 $T_{\%}^{min}$，则物理上不可行（受限于孔径衍射极限），可以接着优化，不过需要提示用户

$$
T_{\%}^{limit} = \begin{cases} 
\frac{\lambda}{2 D \cdot \Omega} & \text{Infinite (Range } \Omega \text{ in } \sin\theta \text{)} \\
\frac{\lambda z}{2 D \cdot S} & \text{Finite (Size } S \text{ at distance } z \text{)}
\end{cases}
$$

# Simulation Pixel Size & Down-sampling Logic

注意pixel size 倍数策略仅适用于无穷远或者较远距离（SFR）模式，不适合ASM模式

## 1. 变量定义 (Definitions)
* $\lambda$: 工作波长
* $p_{global}$: 全局基础像素大小 (Global "Unit" Pixel Size, fixed)
* $p_{opt}$: 实际仿真/优化使用的像素大小 (Optimization Pixel Size)
* $N$: 下采样倍数 (Integer Multiplier), $N \in \mathbb{Z}^+$
* $\theta_{max}$: 目标图案的最大衍射半角 (Max Half-Angle)

## 2. 最大衍射角推导 (Derivation of $\theta_{max}$)
根据工作模式确定 $\theta_{max}$，这是决定 k 空间 (角度谱) 覆盖范围的关键。

* **情况 A: 无穷远 / 角度空间 (Infinite Conjugate)**
    * 设目标图案的总视场角跨度为 $\Theta_{FOV}$。
    * $\theta_{max} = \frac{\Theta_{FOV}}{2}$

* **情况 B: 有限工作距离 (Finite Conjugate)**
    * 设目标图案在工作距离 $z$ 处的物理尺寸（直径或最大对角线跨度）为 $S_{target}$。
    * $\theta_{max} = \arctan\left(\frac{S_{target}/2}{z}\right)$

## 3. 像素尺寸限制 (Pixel Size Constraint)
为了使 DOE 能够将光衍射到 $\theta_{max}$，必须满足采样定理以避免混叠。

* **k空间带宽需求**: $B_k = \frac{2 \sin(\theta_{max})}{\lambda}$ (覆盖 $\pm \theta_{max}$)
* **奈奎斯特极限 (Nyquist Limit)**: 实空间采样频率必须大于 k 空间带宽。
    $$\frac{1}{p_{opt}} \ge B_k \implies p_{opt} \le \frac{\lambda}{2 \sin(\theta_{max})}$$

因此，允许的**最大仿真像素** $p_{limit}$ 为：
$$p_{limit} = \frac{\lambda}{2 \sin(\theta_{max})}$$


## 4. 倍数选择逻辑 (Integer Multiplier Selection)
我们在 $p_{global}$ 基础上寻找整数倍 $N$，使得 $p_{opt} = N \cdot p_{global} \le p_{limit}$。

### 计算最大倍数 $N_{max}$
$$N_{max} = \left\lfloor \frac{p_{limit}}{p_{global}} \right\rfloor$$

### 用户选项 (User Options)
* **若 $N_{max} \ge 1$**:
    用户可选择的优化倍数为 $n \in \{1, 2, ..., N_{max}\}$。
    对应的优化像素为 $p_{opt} = n \cdot p_{global}$。
    *(倍数越大，计算越快，但接近极限时可能会在大角度处产生效率下降)*

* **若 $N_{max} < 1$ (即 $p_{limit} < p_{global}$)**:
    **Error**: 全局像素 $p_{global}$ 过大，无法支持目标所需的衍射角度。

## 5. 实例演算 (Example Calculation)

**场景**:
* **波长**: $\lambda = 532 \text{ nm} = 0.532 \text{ }\mu\text{m}$
* **目标**: 无穷远投射，全视场角 $\Theta_{FOV} \approx 15.2^\circ$ ($\sin \theta_{max} \approx 0.133$)
* **全局像素**: $p_{global} = 0.5 \text{ }\mu\text{m}$

**步骤**:
1.  **计算理论极限**:
    $$p_{limit} = \frac{0.532}{2 \times 0.133} = 2.0 \text{ }\mu\text{m}$$

2.  **计算最大倍数**:
    $$N_{max} = \lfloor \frac{2.0}{0.5} \rfloor = 4$$

3.  **提供选项**:
    用户可选择以下方案进行第一步 DOE 优化：
    * **1x** ($0.5 \text{ }\mu\text{m}$): 最高精度，最慢。
    * **2x** ($1.0 \text{ }\mu\text{m}$): 均衡。
    * **3x** ($1.5 \text{ }\mu\text{m}$): 较快。
    * **4x** ($2.0 \text{ }\mu\text{m}$): 极限速度，刚好满足采样定理。

4.  **后续流程**:
    若用户选了 4x ($2.0 \text{ }\mu\text{m}$) 优化，完成后需最近邻插值(Up-sampling)回 $0.5 \text{ }\mu\text{m}$ 进行最终的工艺约束优化。