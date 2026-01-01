# 原始项目（重构前的）代码功能

原始项目代码（根目录）面向DOE（衍射光学元件）的设计与基于激光直写（DLWL）工艺的加工优化。

通过光学部分和工艺部分的前向模型，并用pytorch实现，并且定义图案误差的MES Loss，从而可以直接调用默认优化器（Adam）对加工参数或相位进行优化，实现目标光学强度图案与仿真的图案之间差别最小化。

重要代码文件或配置等包括：

- config 文件夹：记录激光直写（DLWL）工艺的校准测量参数，用于工艺模型。主要包括GT曲线（非线性映射，描述光刻时剂量和高度的非线性）以及低通曲线（这里称LP；原理类似光刻镜头对应的MTF，描述光刻镜头形成聚焦光斑的邻近效应）
- utils 文件夹：包含很多重要的工具函数，具体包括：
    - gen_pattern.py：用于生成不同种类的DOE的目标pattern（为1D或2D的光强或幅度分布），包括自定义图像，分束器（1D或2D强度点阵），偏转器（或blazed grating）等
    - load_config.py：更新、读取等config 文件夹相关的操作
    - modules.py：优化模块，会调用对应的优化算法执行优化
    - show_resutls.py：可视化
    - utils_tensorboard: tensorboard以及记录优化过程相关
    - utils.py：工具函数，loss定义，通用算法，IO函数
- algorithms.py：真正执行优化的函数，包含几种优化算法，会被modules.py调用
    - 注意这里我们可能会用到两种优化策略：SGD和BS。其中SGD中主要是利用默认优化器Adam，利用前向模型通过自动反向传播来做最常见的梯度下降，适合大部分情况；BS主要针对1D情况以及规模极小的2D情况的优化，会不断遍历每个pixel做一次binary search，性能有时候会好于SGD，但计算量很大。其中还实现了GS算法，但我们目前没有使用（因为工艺模型不适合做GS；GS可能只适合做DOE形貌设计，即后面会提到的两步法的第一步）。
- fit_model.py：主要负责GT曲线的分段多项式拟合，形成拟合后的模型
- optimize.py：为了易于使用构建的执行优化recipe的入口函数。会调度优化，包括IO、暂存或多轮优化等。会调用utils.modules（内部会转向algorithms.py）构建并执行优化。包含多个可以直接使用的功能：
    - init_model： 初始化模型
    - fab_eval：把某个dose结果送入工艺仿真器，查看仿真结果
    - e2e_opt：执行端到端（e2e）优化，即通过目标光强图案的Loss（基于光学+工艺模型）直接优化加工Dose
    - fab_opt：仅执行工艺优化，即通过给定DOE高度形貌分布，优化加工参数（Dose分布）
    - height_opt：仅执行形貌优化（即传统DOE优化），给定目标光强图案，优化DOE高度形貌分布
- physical_model.py 包含光学部分（衍射光学传播）和工艺部分的前向模型：
    - 光学部分的模型：
        - propagation_ASM：主要用于较近距离的光学传播（特点：输出和输入面大小相近）。propagation_ASM算法会根据传播距离来选择两种算法，一种在频域构建传播kernel，称为角谱法（ASM），适合于近距离传播；另一种在空域构建传播kernel，称为瑞利-索莫非卷积（RSC）。本方法其实更应该称为卷积法，不过为了简单，命名为propagation_ASM，来体现更常见的ASM方法。
        - propagation_SFR：主要用于稍远距离的光学传播（特点：输出面大小会自然随着传播距离增长）。SFR的含义是单次single-Fourier transform based Fresnel transform（参考./SFR_theory.md），即仅使用一次FT的方法（和基于卷积的propagation_ASM形成对比）。原始的SFR方法的输出面大小是固定的。这里为了让输出面大小可以调整，在原始SFR方法的基础上引入了zoom-FFT（或成为非均匀采样FFT），从而比起原始的SFR，输出面的大小可以做调整。这里的目标面尺寸，由config（见config/gauss.yml)中的output_size决定；另外值得一提的是，目前的测试主要基于propagation_ASM和后面的propagation_FFT，propagation_SFR没有过多测试，可能有些bug.
        - propagation_FFT：前面的两种方法均用于有限远的传播。而这里的propagation_FFT会直接通过FT计算角谱，映射到角度空间（对应无限远），因此专用于无限远的传播
    - 工艺部分的模型：
        - 通过fabrication_model类完整描述，其中的forward函数即为工艺模型的构成。输入模型为加工参数（Dose分布），通过模型的非线性映射和低通后，转换为预测或仿真的加工后的DOE高度形貌分布。这里还提供了backward功能，作为adjoint运算，不过实际中应该没有利用这个功能。

代码运行可以通过下面的例子展示：

- main_train.py：展示了e2e（端到端模式）的优化的工作流程，即给定目标图案，优化加工dose
- main_train_2step.py: 展示了两步法（先做传统DOE优化，给定目标图案，优化得到加工目标形貌；再将此目标形貌作为加工目标，优化加工Dose）的工作流程


因为我们最终可能以两步法为主，因此可以重点看main_train_2step，了解如何做传统DOE的高度优化，以及如何再做工艺优化的流程。

部分config参数解释：

- slm_res：DOE或Dose的仿真像素数目
- output_res：光学仿真器仿真后，会尝试插值上采样到output_res
- target_res：用户输入的目标图案会resize到这个分辨率
- roi_res：最后evaluate loss的时候，会对仿真结果和目标图案均crop到roi_res再计算loss
- save_wrap_res：仅临时使用，保存的时候有时候会扩展像素大小，不再需要。

# 本项目目标

## 整体目标

本项目的目标是为交互式的用户App（主项目）提供后端计算（本项目仅需要包括基于python部分的所有DOE优化相关的函数）。因此本项目相当于是一个子项目，重构目标是代码的可扩展性、简洁以及结构清晰。目前的代码因为经历了很多次较随意的修改，一开始也没有仔细设计，因此结构略微有些混乱，应该需要仔细重构。

在整个app项目（主项目）的架构下，本项目会接收来自前端或后端服务器的信息，作为用户参数。用户参数会包含DOE类型，需要优化的特定目标的参数（或目标图案），以及是否进行工艺优化（以及相关工艺信息）。本项目专门执行相关的优化，并返回优化结果（可能是DOE优化高度形貌结果以及光学仿真效果；或者工艺优化后结果进行工艺+光学仿真后的光学效果）到前端或后端服务器。

这里和本项目目前的一个重大区别是，我们将仅关注“2step”或两步法，因此要么运行传统DOE优化，得到目标DOE形貌和相位（供用户下载）以及对应的evaluation（再跑一次光学正向模型）和分析结果提供给用户。；要么基于第一步的基础，再运行工艺优化，得到加工Dose参数（不提供给用户），在把这个优化得到的dose再跑一次正向全模型（工艺+光学）后得到的evaluation和分析结果提供给用户。

另外，我们需要根据下面的用户参数详情来重构代码，以满足各种情况下的设计。

python_service.md 文件是主项目部分对本python项目的初步集成想法，包括主项目和此项目之前的接口，但因为没考虑目前此子项目的结构，因此可以参考后做出调整。注意本项目应该仅基于python，和主项目易于分离，相对独立。

## 按DOE类别的目标用户参数详情

目前从用户角度的定义还不清晰。目前我们主要对下面这些DOE类型感兴趣：beam splitter，spot projector，diffuser，pattern projector， diffractive lens，diffractive lens array，deflector (blazed grating)；但目前的代码尝试过beam splitter，spot projector，pattern projector， diffractive lens, deflector；且目前都是通过config文件来定义参数，有的参数定义不太用户友好，有的不太合理，都需要改进。后面是详情。

在讨论细节之前，需要指出一个设计逻辑上的考虑。因为用户只关心DOE的最终光学效果，因此目前的参数绝大部分都是最后的光学指标，但有的地方涉及一些DOE重要参数的估计，需要特别关注。这里我暂时想到的例子是器件尺寸、周期化对于用户设定的目标尺寸（角度）之间的影响（仅适用与工作距离无穷远的DOE）。这里我选用了一个tolerance参数来权衡这两项。

#### （附）tolerance参数解释：器件尺寸和周期化方法

作为用户参数和内部DOE参数之间转换的考虑，这里引入了一个以百分比为单位的tolerance参数，衡量用户可以接受的设计或加工后的DOE在目标面上产生的光学分布与用户指定的图案之间的角度或尺寸差异。不过需要注意的是这个参数应该只适用于无穷远/角度空间类型的DOE，或者较远工作距离的DOE（对于有限工作距离，需要在设计完成后叠加一个全局Fresnel透镜相位；这是由于Fresnel传播公式中需要抵消此相位后，才可考虑其周期化版本对应的k空间；或者理解为无穷远工作的周期化DOE叠加了聚焦相位，实现有限远距离工作；但由于Fresnel相位的特点，为了避免采样不足问题，仅适用于较远的工作距离）
设置这个参数是因为，设计DOE时，我们的仿真区域第一受到器件限制；第二是受到可选的周期化参数限制。如不引入周期化，则整个器件都可以自由优化；如引入周期化，则整个器件看作一个更小的区域的周期延拓，从而仿真区域得到缩小，优化速度加快。但周期化会导致k空间（角度空间）的采样，进而导致实际可行的角度分布落在一个k空间的均匀网格上（或通过k和角度之间的三角函数关系转换为角度空间的非均匀网格），这些角度空间的网格与用户定义的角度（例如点阵情况，用户定义的角度在一个自定义均匀网格上）一般不会重合，就造成了角度误差，他们之间存在的最大差异便导致了tolerance参数的考虑。如果允许更大的tolerance，则可以选择更小的周期（仿真/优化区域），且获得更加稀疏的级次分布（有可能降低杂散衍射级次）；不过即使选择整个器件，也需要注意到，我们的角度采样依旧是离散的，因此对于给定尺寸的器件，tolerance有一个理论下限。

#### （附）pixel size的考虑

我们会设置一个由加工设备所提供的固定的全局pixel size，作为工艺优化阶段采用的（参考main_train_2step.py）。但在运行DOE优化（第一步）时，我们可以选择是全局pixel size若干倍数的等效“大pixel”用于优化，也可以有效减少各种情况的仿真/优化像素数的计算量。不过这里要考虑用户的目标尺寸（角度）的范围，对于有限远情况，要计算等效最大衍射角；对于无限远情况，用目标角度的发散半角作为最大衍射角。再用此最大衍射角对应到k空间span后，再对应到实空间采样间距，并按照其1/2或更小（满足采样定理）的尺寸来作为最大的“大pixel”可选值。因此，可以以这里计算的实空间采样间距的一半作为基准；当其比全局pixel size大时，在可以考虑给用户所有选择（包括所有可行的“大pixel”可选值）。这里可以举个例子。如全局pixel size为0.5um，但刚才按照角度span计算出来的实空间采样间距为4um，其一半为2um，因此用户可以选择全局pixel的1倍（原始的0.5um），2倍（1um），3倍（1.5um）或4倍（2um）来进行第一步DOE优化。之后如用户选择工艺优化，则再重新按照固定倍数上采样到全局pixel size后进行工艺优化。如遇到实空间采样间距的一半小于全局pixel size的情况，需要抛出错误，表示衍射角过大。

### 公共参数（所有 DOE 类型共用）

- 工作距离（可以为无限远）
- 工作波长
- DOE 类型
- 器件直径
- 器件形状：（"circular"或"square"；对于circular情况可以按照默认square设计，但最后加一个mask输出即可）


### 加工模拟参数（可选，所有类型共用）

- 是否启用加工优化
- 工艺配方选择

### 特定DOE的额外参数

#### 2D Spot Projector（二维点阵投影器）

- 投影点阵的行数和列数
- targetType 目标规格类型：根据工作距离是否无穷远，指定目标尺寸的单位是角度还是大小。
- 对于有限远传播的情况，对于给用户一个选项，可以选择性开启目标尺寸设定（对应采用SFR来仿真输出光强）；否则则基于ASM函数，输出尺寸无法更改，强制和输入尺寸相同。
- 目标尺寸：对于无限远的情况（targetType="size"），对应是投影一维点阵的角度span（水平和垂直一致）；对于有限远的情况（targetType="angle"），对应是投影一维点阵的空间span（长或者宽，应该一致）
- 对应有两种模式（可切换）：
    - 第一种是强制均匀网格，用户输入目标尺寸和容差，目标图案会设置成与均匀分布点阵最接近的点阵图案（容差会决定精度，会snap到最接近的k空间采样点上）
    - 第二种是自然网格（对应自然的衍射级次，即k空间均匀网格，但大角度会出现畸变），用户输入目标尺寸和容差，会根据目标尺寸决定最高级次的衍射角，得到k空间的采样间隔，对应器件周期）

#### Diffuser（匀光片）

- 扩散形状：可以选方形或圆形（均匀）
- targetType 目标规格类型：根据工作距离是否无穷远，指定目标尺寸的单位是角度还是大小。
- 对于有限远传播的情况，对于给用户一个选项，可以选择性开启目标尺寸设定（对应采用SFR来仿真输出光强）；否则则基于ASM函数，输出尺寸无法更改，强制和输入尺寸相同。
- 目标尺寸：对于无限远的情况（targetType="size"），对应是投影一维点阵的角度span（水平和垂直一致）；对于有限远的情况（targetType="angle"），对应是投影一维点阵的空间span（长或者宽，应该一致）


#### 1D Splitter（一维分束器）

- 分束数目
- targetType 目标规格类型：根据工作距离是否无穷远，指定目标尺寸的单位是角度还是大小。
- 对于有限远传播的情况，对于给用户一个选项，可以选择性开启目标尺寸设定（对应采用SFR来仿真输出光强）；否则则基于ASM函数，输出尺寸无法更改，强制和输入尺寸相同。
- 目标尺寸：对于无限远的情况（targetType="size"），对应是投影一维点阵的角度span（水平和垂直一致）；对于有限远的情况（targetType="angle"），对应是投影一维点阵的空间span（长或者宽，应该一致）
- 对应有两种模式（可切换）：
    - 第一种是强制均匀网格，用户输入目标尺寸和容差，目标图案会设置成与均匀分布点阵最接近的点阵图案（容差会决定精度，会snap到最接近的k空间采样点上）
    - 第二种是自然网格（对应自然的衍射级次，即k空间均匀网格，但大角度会出现畸变），用户输入目标尺寸和容差，会根据目标尺寸决定最高级次的衍射角，得到k空间的采样间隔，对应器件周期）

---

#### Lens（衍射透镜）

- 焦距
- 透镜类型：普通或柱面（X或Y方向）
- 特殊功能: 可以选多个给定目标平面计算loss，实现扩展焦深；也可以选多个给定工作波长值，实现扩展带宽优化

注：对于透镜情况，直接基于ASM函数仿真传播，因为不需要很大的目标尺寸

#### Lens Array（透镜阵列）

- 阵列规模（如 5 对应 5×5）
- 焦距
- 透镜类型（同 Lens）
- 特殊功能（同 Lens）

注：对于透镜情况，直接基于ASM函数仿真传播，因为不需要很大的目标尺寸

#### Prism（棱镜）或 deflector (偏转器) 或 blazed grating （含义相同）

- 偏转角度：包含二维角度


#### Custom Pattern（自定义图案或hologram）

- 用户上传的图片数据
- resize后的分辨率：类似目前 config/gauss.yml 中的target_res
- targetType 目标规格类型：根据工作距离是否无穷远，指定目标尺寸的单位是角度还是大小。
- 对于有限远传播的情况，对于给用户一个选项，可以选择性开启目标尺寸设定（对应采用SFR来仿真输出光强）；否则则基于ASM函数，输出尺寸无法更改，强制和输入尺寸相同。
- 目标尺寸：对于无限远的情况（targetType="size"），对应是投影一维点阵的角度span（水平和垂直一致）；对于有限远的情况（targetType="angle"），对应是投影一维点阵的空间span（长或者宽，应该一致）
- 按照之前对 tolerance 的解释，当讨论此处的tolerance参数时，应该以尽力满足给定的分辨率（输出目标尺寸按照像素数目分割）为主，如不能满足，可以给出等效像素数目（即不利用周期化方法，发现输入面最多实现的像素低于用户要求，应当正常优化后告知）

# 目前初步重构后的代码情况

  原始代码结构（根目录）

  dlwl/
  ├── physical_model.py      # 光学+工艺模型（混合）
  ├── optimize.py            # 优化入口和调度
  ├── algorithms.py          # 优化算法实现
  ├── fit_model.py           # GT曲线拟合
  ├── utils/
  │   ├── gen_pattern.py     # 目标图案生成
  │   ├── modules.py         # 优化模块
  │   ├── utils.py           # 通用工具
  │   └── ...
  └── config/                # YAML配置文件

  重构后代码结构（doe_optimizer包）

  doe_optimizer/
  ├── core/
  │   ├── config.py          # 统一配置类（dataclass）
  │   ├── propagation.py     # 光学传播（ASM/FFT/SFR）
  │   ├── fabrication.py     # 工艺模型
  │   └── optimizer.py       # 核心优化器
  ├── patterns/
  │   ├── base.py            # 图案基类
  │   ├── splitter.py        # 分束器
  │   ├── diffuser.py        # 匀光片
  │   ├── lens.py            # 透镜
  │   ├── deflector.py       # 偏转器
  │   └── factory.py         # 工厂模式
  ├── pipeline/
  │   ├── two_step.py        # 两步法流程
  │   └── evaluation.py      # 评估函数
  └── utils/
      ├── fft_utils.py       # FFT工具（ZoomFFT2）
      ├── math_utils.py      # 数学工具
      ├── image_utils.py     # 图像处理
      └── visualization.py   # 可视化

重构代码的测试代码位于根目录：
- test_custom_pattern.py
- test_splitter.py

  主要区别

  | 方面     | 原始代码                    | 重构代码                                             |
  |----------|-----------------------------|------------------------------------------------------|
  | 配置方式 | YAML文件 + 散落的参数       | 统一的dataclass配置类（DOEConfig, PhysicalParams等） |
  | 代码组织 | 功能混杂在少数大文件        | 模块化分层（core/patterns/pipeline/utils）           |
  | DOE类型  | if-else分支在gen_pattern.py | 工厂模式 + 策略模式（patterns/）                     |
  | 传播函数 | physical_model.py混合实现   | 独立的propagation.py，接口统一                       |
  | 优化流程 | optimize.py + algorithms.py | optimizer.py + pipeline/                             |
  | 可扩展性 | 添加新DOE需修改多处         | 继承PatternGenerator基类即可                         |
  | 类型提示 | 基本无                      | 完整的类型注解                                       |
  | 文档     | 散落注释                    | 模块级docstring + 函数文档                           |

  关键改进点

  1. 配置系统

  原始：散落的YAML和硬编码
  config = load_config('config/splitter.yml')
  slm_res = config['slm_res']

  重构：类型安全的dataclass
  config = DOEConfig(
      doe_type=DOEType.SPLITTER_2D,
      physical=PhysicalParams(wavelength=532e-9, ...),
      target=TargetParams(num_spots=(5,5), ...),
  )

  2. 有限距离策略

  原始：隐式处理，逻辑分散
  重构：显式策略枚举
  class FiniteDistanceStrategy(Enum):
      ASM = "asm"                      # 策略1：直接ASM
      PERIODIC_FRESNEL = "periodic"    # 策略2：周期+Fresnel叠加

  3. 传播函数接口

  原始：physical_model.py中混合
  def prop(u_in, config, ...):  # 参数混乱

  重构：统一接口
  def propagation_ASM(u_in, feature_size, wavelength, z, ...) -> Tensor
  def propagation_FFT(u_in, output_resolution, z) -> Tensor
  def propagation_SFR(u_in, feature_size, wavelength, z, output_size, ...) -> Tensor

  4. 评估系统

  原始：eval.py简单实现
  重构：完整的评估流程
  @dataclass
  class FiniteDistanceEvaluation:
      intensity: np.ndarray
      spot_efficiencies: List[float]
      uniformity: float
      total_efficiency: float
      airy_radius_meters: float
      ...

# 额外说明和待解决问题

由于本项目以2step方法为目标，但config/base.yml和config/gauss.yml都是e2e方法，也需要参考对应的splitter_2step.yml, dot_projector_2step.yml, fresnel_lens_2step.yml 和 gauss_2step.yml

本项目需要给定一个全局变量，方便从外部设置一个限制，给定允许运行优化或仿真的最大像素规模，如2000*2000

运行原始项目或重构项目的python环境（位于根目录下的.venv环境）：.venv/Scripts/python.exe 

项目做过的改动或升级，大部分会反映在 changelog.md中，作为最新版本的参考。

## splitter相关

另外对于1D splitter的优化，之前的config的解释举例如下：
```yaml
roi_res: [6,1] # ROI whose inside considers loss
slm_res: [6,1]
pattern_params:
  dim: 1
  orders: [0,1,2,3,4]
  order_shift: [0,0] # for 1D, use [shift_value,0]
  roi_order_num: 5 # ROI which encircles operating orders
  tot_order_num: 6 # total pixels in width/height
  save: True
  load: False
```

这里的逻辑与我们的目标有些区别，需要额外说明（可以参考 utils/gen_pattern.py）。这里面并没有给定目标尺寸，而是用户直接给定slm_res，代表DOE的周期化和周期尺寸直接给定，相当于直接给定了k空间的范围和网格，因此后面的tot_order_num大小与其相同。roi_res和之前自定义图案的逻辑类似，只考虑中心附近这样的大小（计算结果crop到这个大小）后的loss。这里的orders代表需要的工作级次，注意这里的序号是指的全部roi_order_num个级次中，取出中心的roi_order_num个级次，并重新定义级次索引（对应orders），order_shift对应roi_order_num个级次偏离中心的情况。这里的逻辑对应的是自然网格（对应自然的衍射级次，即k空间均匀网格），最大衍射角空间范围（tot_order_num的边缘级次）由pixel size直接指定，并按照k空间均匀切分后，优化中心的roi_order_num个级次中的特定级次（索引为orders）。和我们目标实现的逻辑不太一样，因此需要注意。而在 dot_projector_2step.yml 中，方式有所不同，这里的distort_corr 即对应是用自然网格还是强制均匀网格。这里的尺寸等定义与自定义图案的方式类似。最后指的一提的是，由于2step的第一步优化目标是height（单位是m，但一般为um量级），因此之前这一步的学习率均为 1e-8，1e-9量级；第二步工艺优化输出是0-255的dose，但输入还是上一步产生的height（um量级）因此学习率也比较不标准，之前尝试过1.0e+6或1.0e+7  (参考2step相关的yml配置)。考虑沿用或者按照单位预缩放一下学习率。

【本项目目标】对于本项目，我们需要实现两种方法：k空间均匀采样，和角度/目标面（对应无限远或有限工作距离）空间（近似）均匀采样两种方法。这里面都涉及周期化。无限远直接对应周期器件；有限远的情况需要特殊处理（见后面说明）

- 对于k空间均匀采样的情况，需要通过搜索决定最小仿真周期$\Lambda_min$，使对应的k空间网格$2\pi/\Lambda_min\times [m,n]$换算成角度或尺寸后，在目标尺寸的范围内恰好包含对应【分束数目】个衍射级次，然后采用此$\Lambda_min$作为周期化的仿真区域大小
- 对于空间（近似）均匀采样两种方法，则需要根据给定的tolerance，参考 supp.md中的方法估计所需的最小仿真周期，从而对应的k空间网格足够密集，以至于在转换为角度或尺寸后，所有snap到网格上的角度和目标均匀角度网格相比依旧可以满足tolerance

【注意】对于偶数且k空间均匀采样的情况，会造成非对称的图案，因此可以把搜索确定的周期尺寸加倍，从而k空间加密一倍后，可以间隔着选择级次，跳过其余的级次，因此实现对称的图案。举个例子，对于2*2自然分束的情况，如果原本计算的周期包含4*4个像素，对应级次为(-1,-1),(-1,0),(0,0),(0,-1)；则加倍周期变成8*8像素，再仅选择(-1,-1),(-1,1),(1,-1),(1,1)作为目标图案

【注意】有限远的情况有两种策略：
1. 目标尺寸和整个DOE大小相近（目标尺寸按照目前选择的“大pixel”来计算优化/仿真时需要的像素规模，当满足之前提到的允许运行优化或仿真的最大像素规模限制时，可以认为“相近”），则利用ASM传播函数来直接优化，类似图案优化的情况，不需要考虑衍射级次。此时不需要tolerance参数，只能用空间均匀采样一种模式（非周期器件，不考虑k空间）。
2. 不满足第一点（目标尺寸过大），则用周期化的策略，将其视为无限远的情况。因为可以把有限远工作的DOE视为聚焦Fresnel相位叠加在无限远的周期化DOE上，因此设计方法按照无限远，只是把有限远情况的目标尺寸转换为角度，进行【无限远】的情况的优化。优化后需要对【全局】（指整个器件，而不是单周期）叠加聚焦Fresnel相位。

【可视化】包括单周期的phase，整个器件的phase，所有工作级次衍射效率的stem图（包括参考理论值作为红色虚线标出，和统计量），以及用散点图（颜色代表衍射效率）绘制各工作级次的角度分布或位置分布（横纵轴对应角度，对应无限远情况；有限远情况需要特别讨论，如下）。

【有限远的情况可视化】最终评估衍射效率等时，如果用的策略2，则可以沿用无限远的情况得到类似的结果之外（不过把角度换算为位置），不过还需要用SFR传播函数仿真一下目标面的光强情况，并依次处理每个光斑（具体是计算对应的每个分束聚焦光斑的位置，再在此位置取一个圆形区域，半径由DOE全孔径的Airy Spot Radius决定，计算此区域内的能量占入射能量的比例作为此光斑的衍射效率），作为另一个衍射效率的估计。
如果用的策略1，则不存在前面的角度换算为位置，而是直接依次处理每个光斑（具体是计算对应的每个分束聚焦光斑的位置，再在此位置取一个圆形区域，半径由DOE全孔径的Airy Spot Radius决定，计算此区域内的能量占入射能量的比例作为此光斑的衍射效率
