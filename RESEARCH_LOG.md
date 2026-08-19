# Research log

本文件只追加，不删除死路。日期按本地会话日期记录。

## 2026-08-18 — 问题核定与首轮文献侦探

### 问题与榜单

- 核对 Friedman 原页：`N=1..36`，定义为单位正方形内多个单位点源的倒平方强度和之连续最小值。
- 核对提交说明：页面需要构型图、数值精度、方法和归属；是否扩展类别由维护者决定。
- 发现条目高度集中在 2026 年 7 月，说明该具体榜项很新；这只是页面时间证据，不代表数学问题新。

### 直接同题命中（改变项目定位）

GitHub repository search 查询：

- `"Riesz polarization" in:name,description,readme`
- `"minimum light intensity" Erich Friedman in:readme,description`
- `"square Riesz" polarization in:readme,description`

直接命中：
[Driedsandwich/square-riesz-polarization-certificates](https://github.com/Driedsandwich/square-riesz-polarization-certificates)，创建于 2026-07-28，最后检查到的 push 为 2026-08-08。

关键事实：

- 覆盖 44 个 `N`，到 `N=52`；
- 公开坐标和两套严格固定构型证书；
- 不公开候选搜索代码、种子和完整轨迹；
- `N=9,13` 未改进；
- 多个 `N=15..36` 的仓库内部严格下界远高于官网当前显示值；
- 官网延迟不能用来制造“纪录”。

这纠正了早先“没有别人先做有限 `N` 系统计算”的判断。原因是项目刚公开、搜索引擎尚未充分收录，精确问题关键词的普通网页检索没有返回它；GitHub repository search 找到了它。

### 数学文献查询

主要查询：

- `Riesz polarization square numerical computation`
- `Riesz d-polarization critical s=d next order`
- `lattice polarization Riesz potential`
- `polarization Epstein zeta`
- `Chebyshev points covering separation`
- 核心论文的参考文献与作者作品列表。
- 德语/英文组合查询：`Riesz-Polarisationsproblem Quadrat numerisch`、`Riesz polarization square rectangle thesis`、`Chebyshev constant square Riesz potential numerical`。

确认：

- 标准领域是 Riesz polarization / Chebyshev constants；
- `s=d=2` 首项为 `pi*N*log N`，渐近均匀；
- 周期二维 Gaussian polarization 中六角晶格有严格最优性；
- `1/r^2` 无限二维晶格临界发散，迁移需要重整化；
- 未找到单位正方形临界问题的明确次阶展开或有限 `N` 全局理论。
- 组合语言与 thesis 查询仍只返回一般最小值原理、球/球体、渐近或无关的经典 Chebyshev 容量，没有出现同题有限正方形数值论文。

负面结果边界：最后一条只是当前查询阴性，不能当成不存在证明。

### 优化文献查询

主要查询：

- `polarization mixed integer programming compact sets`
- `nonlinear semi-infinite programming exchange adaptive discretization`
- `Remez exchange algorithm maximin`
- `continuous protected area worst point facility location`

确认：

- polarization 是标准半无限规划；
- exchange/adaptive discretization 是成熟轮子；
- Rolfes 等给出结构条件和 MIP 上下界；
- 其 MIP 假设连续有限核，不能无修改套到奇异 `1/r^2`；
- 固定构型严格验证已有同题精确有理实现，应优先复用。

### 跨领域查询

1. **照明/光度学**：IES 与 LED 阵列论文。结论：真实照明有高度、余弦、方向图和反射，本题是抽象同平面 Riesz 模型。
2. **放疗**：brachytherapy seed placement、TG-43。结论：共享多源剂量叠加与布点优化，但真实剂量核更复杂。
3. **设施选址**：multiple obnoxious facilities、continuous protected areas。结论：共享逆平方累积与最坏点；目标符号相反，Voronoi/不利点迭代可借。
4. **无线能量**：power beacon placement。结论：共享路径损耗与保障覆盖，但工程问题常含随机 outage 和不同路径损耗指数。
5. **统计物理/晶格**：Coulomb/Riesz gases、renormalized energy、Gaussian lattice sums。结论：最有希望解释 bulk 三角晶格，但 energy 与 polarization 不同。
6. **半色调/blue-noise**：gray pattern、BNOT、CVT、GPU N-body。结论：最有希望提供非重复的初始化和结构统计轮子。

### 代码检索

- GitHub 中精确同题只发现上述公开证书库。
- 未发现 Rolfes–Schüler–Zimmermann 论文的公开实现仓库。
- BNOT 作者页提供源码，可作为初值生成器。
- 同题证书库的验证代码可复用；搜索器需要独立实现。

### 硬件检查

- `nvidia-smi`：RTX 5090，32607 MiB，driver 610.88，compute capability 12.0。
- `E:\math\.venv` 中可见 PyTorch CUDA 包，但 Python executable 当前返回 access denied。实现前先修复环境并实际运行 CUDA smoke test。

### 当前决策

**有条件 GO**：不做“官网延迟套利”，做公开可复现 GPU 搜索 + 严格证书 + 有预注册分析的有限尺寸数据项目。

## 待继续核查

- [ ] 沿 Borodachov–Bosuwan、Rolfes 等论文做完整 backward/forward citation chase。
- [ ] 检查专著 *Discrete Energy on Rectifiable Sets* 第 14 章的 square/critical 细节。
- [ ] 检索德语、俄语及学位论文中的 polarization/Chebyshev constant 数值工作。
- [ ] 查明 `N=3..6` 是否已有非网页形式的精确/高精度结果。
- [ ] 检查 Gaussian lattice result 对临界 Riesz 的正则化陈述能推进到什么程度。
- [ ] 外部联系领域作者前先取得用户批准。

## 2026-08-18：N=3 搜索、严格证书与证明 ROI

### 数值发现

- RTX 5090，`seed=2026081803`，`batch=256` 的通用 soft-min 搜索在约 11.4 秒
  总时间内得到连续评估 `7.5537698477` 的非对称候选。
- 将其投影到单轴反射族后提升到 `7.55909`；以四角和下边中点为活跃集做
  epigraph/KKT 精炼，得到 100 位稳定解 `7.5683896400296902150...`。
- 冻结的 19 位有效数字坐标见 `data/candidates/n03_symmetric.json`。257 网格加
  内部/四边多起点连续搜索只得到五个等高最暗点：四角与下边中点。

### 严格固定构型证书

- 泛化上游 BSD-3-Clause spectral 精确分数验证器，并保留第三方声明。
- 另泛化 componentwise Hessian 区间验证器；两者分别精确复现上游 `N=7`
  的 499 和 489 次分割。
- 对冻结 `N=3` 坐标，spectral 用 141 次、componentwise 用 123 次分割均证明
  `I(X) >= 7.56838963`；`(1/2,0)` 的精确有理求值给出
  `I(X) <= 7.56838964002969021461149434458...`。
- 全部 8 个单元测试通过。证书证明固定构型，不证明源位置空间中的全局最优。

### 局部与全局最优诊断

- 五个活跃点存在严格正 KKT 权重；完整六维源梯度残差约 `4.5e-16`。
- 把下边中点错误地固定时，临界子空间 Hessian 有 `+3.38` 方向；这只说明
  固定五点上界不够紧。加入边界最暗点随源移动的 Schur-complement 包络修正后，
  两个临界特征值为约 `-20.90`、`-1.63`。四角及下边中点的可行法向导数也严格。
  这支持完整连续问题的严格局部最优猜想，尚需区间化才能成为定理。
- 全局证明要求覆盖六维源位置空间并处理奇异核与 `forall-exists` 结构；GPU 只适合
  候选/反例搜索，精确上界主要受 CPU branch-and-bound 维数墙限制。
- 依据仓库新增的证明 ROI 原则，暂不直接追求精确全局等式。先做多盆地反例搜索；
  只有放宽常数的粗全局上界原型显示良好剪枝率，才继续收紧。

## 2026-08-18：3D 游戏照明应用复核

- Unreal Engine 当前 Point Light 文档明确提供 physically based inverse-squared
  falloff，Attenuation Radius 仅截断贡献。因此在无阴影、各向同性、线性叠加、
  平面活动域的特例中，游戏点光源最坏点布局与本题同构。
- 真实表面光照还包含高度、表面法线/BRDF、可见性、阴影、有限作用半径、不同
  亮度和间接光。本题应定位为严格可验证的基核，而不是完整渲染模型。
- Unity 6 Light Probe 文档要求探针形成 3D 体积，在光照变化/遮挡多处加密，并
  明确探针数量与内存/运行计算的权衡；Unreal Volumetric Lightmap 在几何附近
  自适应提高采样密度。这是空间采样/插值问题，与光源布局相邻但不等价。
- ROI 更新：整体“数学核 + GPU 求解器 + 游戏表面推广”重要性为中高；`N=3`
  精确全局定理仍只是中等重要的基准，不应阻塞应用路线。

## 2026-08-19：并行纪录搜索与 N=3 计算机辅助证明

### 可恢复 CPU/GPU 搜索器

- 新增 `scripts/record_hunt.py`：CUDA soft-min 生产者不停等 CPU，CPU 进程池并行做
  连续域最暗点验证；GPU job、CPU evaluation、best 与 summary 均原子增量落盘，
  可从中断处恢复。
- 首轮扫描 `N=4,5,6,9,13`，每个 N 八个种子。`N=4,6,9,13` 未过当前公开基线；
  `N=5` 得到连续值 `22.057482928667138`。
- 第二轮对 `N=4,5,6` 各跑 12 个更深种子并完成 257 网格连续验证；`N=4`
  最好 `17.62597<17.729`，`N=6` 最好 `29.50152<29.791`，均未破基线。
- `N=5` 提高到 `22.063083016036774`。冻结有限小数后，spectral/componentwise
  两套精确验证器分别用 160/177 次分割证明 `I(X)>=22.06>21.342`。

### N=3 全局上界

- 实现有限见证源盒树：浮点只选择见证，所有剪枝用 `Fraction` 重算；独立标准库
  验证器自行重建 dyadic 盒、逐叶检查见证界，并验证每个根的 prefix-free 完整覆盖。
- `P_3<=7.7`：816 roots、41,078 leaves，独立审计及删叶/重复叶/非法见证等对抗
  测试通过。
- 审计发现并修复证书参数的隐式类型转换、float target 与非二次幂生成网格问题；
  当前验证器强制十进制字符串 target 和精确 JSON 类型，生成器限制初始划分为二次幂。
- `P_3<=7.58`：1,304,240 leaves；独立重放 103.3 秒通过，最大精确叶界
  `7370316709888/972337298465 = 7.579999987168342...`。

### 对称 KKT 根

- 新增纯有理区间与一阶自动微分；对称七方程系统以精确高斯消元构造 Krawczyk
  算子。
- 半径 `1e-40` 的七维盒严格包含唯一 KKT 根，三个活跃轨道权重均正；这是驻点
  存在/唯一性，不是完整局部或全局最优性。

## 2026-08-19：榜单高端正式扫描

### 搜索设计

- 对 `N=29,31,32,33,34,35` 各跑 4 个确定性种子、`batch=512`、500 步、
  `129x129` GPU witness 网格；CPU 以 `257x257` 网格和每个 `N` 至少 290 个
  连续局部搜索起点复核每个 GPU job 的前 12 名。
- 初值以公开证书库的有限小数 incumbent 为中心加入标准差 `0.001` 的扰动；GPU
  witness 集额外包含 incumbent 的连续域局部极小点。优化器保留每个 population
  member 的历史最好 hard-min 构型，避免 soft-min 后期毁掉强初值。
- 正式命令、24 个 GPU job、288 个 CPU 连续复核与最终摘要保存在
  `runs/record_hunt_large_20260819/`；总墙钟时间约 505 秒。
- 为避免跳过 `N=30` 造成选择偏差，随后以完全相同超参数补跑 4 个种子和 48 个
  CPU 连续复核，保存在 `runs/record_hunt_n30_20260819/`，耗时约 78 秒。

### 严格结果

| `N` | 公开既有构型的严格上界 | 新构型数值最小值 | 已证明下界 | spectral / componentwise 分割数 |
|---:|---:|---:|---:|---:|
| 29 | 272.49597364647275 | 282.8569252861270 | 282.8 | 2298 / 2601 |
| 30 | 285.32674238354997 | 285.3456853298848 | 285.34 | 3369 / 3400 |
| 31 | 294.20889327042522 | 305.2983669115241 | 305.2 | 2391 / 2602 |
| 32 | 304.28079912987812 | 317.2038189278292 | 317.1 | 2473 / 2732 |
| 33 | 311.66564132966665 | 330.5954794801075 | 330.5 | 2788 / 3080 |
| 34 | 323.40992809730865 | 337.9062358230380 | 337.8 | 2455 / 2751 |
| 35 | 329.70896680863521 | 347.1957223312929 | 347.1 | 2432 / 2773 |

- 两种验证均使用 `Fraction` 对整个单位正方形做穷尽分支覆盖，不依赖浮点剪枝；
  六个目标全部通过。spectral 运行约 15--23 秒/例，componentwise 约
  45--65 秒/例。
- 每个 spectral 记录还在数值找到的最暗点作精确有理求值，得到新构型上界
  `282.85692528612702, 285.34568532988481, 305.29836691152415, 317.20381892782922,
  330.59547948010763, 337.90623582303798, 347.19572233129303`。
- `N=30` 的严格提升只有 `285.34-285.32674238354997≈0.01326`，而相邻六项的
  严格提升为约 `10.30..18.83`。因此目前数据反对“只要 N 大就自动容易大幅破纪录”；
  更合理的工作假说是具体 incumbent 的搜索质量/盆地差异。
- 因而这些是对当时可查公开固定构型的严格改进；它们不是 `P_N` 的全局最优性
  证明，也尚未对外提交。
