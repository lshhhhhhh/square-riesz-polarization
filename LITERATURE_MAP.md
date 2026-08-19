# 文献地图与可迁移知识

检索日期：2026-08-18。这里记录的是目前完成的针对性检索，不把“没有搜到”表述成“世界上不存在”。技术结论尽量链接原论文、作者页面、官方标准或原始代码库。

## 1. 同题直接先行工作

### Friedman 纪录页

[Maximizing Minimum Light Intensity](https://erich-friedman.github.io/packing/light/) 定义单位正方形内的倒平方叠加 max-min 问题。当前页面列出 `N=1..36`；大部分 `N=7..36` 条目标注为 Satoshi Kishimoto 与 GPT-5.6 Sol Pro 于 2026 年 7 月得到。

页面是策展的官方纪录展示，但显示精度低且更新有处理延迟。

### 公开严格证书库

[square-riesz-polarization-certificates](https://github.com/Driedsandwich/square-riesz-polarization-certificates) 是目前最直接的公开先行工作：

- 公开 `44` 个 `N` 的固定构型与严格连续下界；
- 两套证书都用精确有理算术和正方形 branch-and-bound；
- 证明的是“给定固定坐标的连续最小值至少为某数”，不是全局最优；
- 候选搜索代码和完整轨迹未公开。

可直接复用：BSD-3-Clause 的证书代码与 CC BY 4.0 的数据/文档（必须保留归属并固定 commit/release）。

不可假装复用：其未公开的候选生成器。我们的搜索器必须独立、可复现，并把其坐标作为 incumbent 而非“待发现答案”。

## 2. 离散势论 / Riesz polarization

### 基本定义与一般理论

- Erdélyi–Saff, [Riesz Polarization Inequalities in Higher Dimensions](https://arxiv.org/abs/1206.4729)：系统讨论 max-min Riesz 势、球与球体上的界和渐近。
- Borodachov–Bosuwan, [Asymptotics of Discrete Riesz d-Polarization](https://arxiv.org/abs/1307.1160)：临界 `s=d` 首项及渐近均匀分布。
- Borodachov–Hardin–Reznikov–Saff, [Optimal discrete measures for Riesz potentials](https://arxiv.org/abs/1606.04128)：带权 `N` 点 polarization 及 `s>=d` 的首项渐近。
- Reznikov–Saff–Vlasiuk, [A Minimum Principle for Potentials](https://arxiv.org/abs/1607.07283)：Riesz-like 核的最小值原理，以及离散最优测度弱星极限与连续问题的联系。
- Reznikov–Saff–Volberg, [Covering and separation of Chebyshev points](https://arxiv.org/abs/1703.00106)：`s>d` 时与覆盖问题及弱分离的关系。我们的 `s=d` 是临界情形，不能把它的分离结论直接搬来。
- Hardin–Petrache–Saff, [Unconstrained polarization problems](https://arxiv.org/abs/1902.08497)：约束/无约束 polarization、Riesz 渐近和开放问题。

可严格采用：问题的标准名称、允许多重集、`P_N/(N log N)->pi`、渐近均匀分布。

尚未得到：单位正方形临界情形的次阶项、有限 `N` 全局最优构型、边界层定理。

### 有限规模结构与严格计算

Rolfes–Schüler–Zimmermann, [Bounds on polarization problems on compact sets via mixed integer programming](https://arxiv.org/abs/2303.10101)（[期刊版](https://link.springer.com/article/10.1007/s00454-024-00635-z)）：

- 把 polarization 写成半无限规划；
- 证明局部最优构型包含在其最暗点凸包内；
- 最暗点位于构型凸包内部或区域边界，局部最优时至少有一个边界最暗点；
- 构造收敛的 MIP 上下界序列。

重要限制：该 MIP 构造假设核函数在 `r=0` 连续有限，并以 Gaussian 为算例；`1/r^2` 在源点奇异，所以它不是本题的开箱即用全局证书。结构定理对严格递减距离核仍是重要诊断工具，固定构型证书则可直接采用公开证书库的方法。

进一步核对期刊正文与引用链后，仍未找到“单位正方形、`s=d=2`、`N=3`”的
显式全局精确构型。Rolfes–Schüler–Zimmermann 也明确没有给一般简单多边形/球的
局部或全局显式构型；这支持项目的新颖性判断，但依然只是截至本次检索的阴性结果。

对证明路线有三个重要边界：

- Reznikov–Saff–Vlasiuk 的连续测度最小值原理针对 `0<s<d` 的 d-Riesz-like 核，
  不能跨过临界 `s=d`；固定连续对偶测度在奇点处还会产生无限势。
- Marendet 等关于量化约束的 interval branch-and-bound
  ([EJOR 2020](https://doi.org/10.1016/j.ejor.2019.10.025)) 提供通用方法论，
  但本项目的有限 witness 源盒上界可直接避开 witness 落入源盒时的奇点。
- Markót 的正方形 packing 区间证明
  ([2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8550790/)) 展示了“浮点搜索找结构、
  区间树给全局证书”的可迁移工程范式；具体目标函数与本题不同。

## 3. 晶格、热核与数学物理

这是最可能产生理论突破的跨领域接口。

### 二维周期 Gaussian polarization

Bétermin–Faulhuber–Steinerberger, [A variational principle for Gaussian lattice sums](https://arxiv.org/abs/2110.06008)：在固定密度的二维晶格中，六角/三角点阵对每个 Gaussian 核最大化整个平面上的最小晶格和。

Riesz 核满足 Gaussian 的 Laplace 叠加表示

\[
r^{-s}=\frac{1}{\Gamma(s/2)}\int_0^\infty e^{-\alpha r^2}\alpha^{s/2-1}\,d\alpha.
\]

所以该结果强烈支持“内部局部结构趋向三角点阵”。但在二维 `s=2` 时，无限晶格和在远处对数发散，需要重整化；有限正方形又有边界与角点。不能把周期 Gaussian 定理直接宣称为本题定理。

相关工作：

- Faulhuber–Steinerberger, [An extremal property of the hexagonal lattice](https://doi.org/10.1007/s10955-019-02368-3)：六角晶格的局部极值性质。
- Faulhuber, [The polarization problem for the honeycomb structure](https://arxiv.org/abs/2509.24687)：比较蜂窝结构与最大 polarization 晶格，显示六角晶格在所有密度下更高。
- Faulhuber–Steinerberger, [Maximal Polarization for Periodic Configurations](https://academic.oup.com/imrn/article/2024/9/7914/7597823)：周期构型、Gaussian/完全单调核、热方程采样之间的联系。

### 能量最小化与重整化能量

- Sandier–Serfaty, [2D Coulomb Gases and the Renormalized Energy](https://arxiv.org/abs/1201.3503)。
- Petrache–Serfaty, [Next Order Asymptotics and Renormalized Energy for Riesz Interactions](https://arxiv.org/abs/1409.7534)。
- Hardin–Saff–Simanek–Su, [Next order energy asymptotics on flat tori](https://arxiv.org/abs/1511.01552)。

这些研究解释了能量最小化中宏观均匀、微观晶化和 Epstein zeta 常数的出现。它们研究的是点对点总能量，不是“最暗点势最大化”。可借鉴重整化、尺度分离和晶格常数的思想，不能直接转移次阶系数。

## 4. 半无限优化、鲁棒优化与 Remez/exchange 思想

本题可写成

\[
\max_{A,t}\;t\quad\text{s.t.}\quad
t\leq U_A(x)\quad\forall x\in[0,1]^2.
\]

这正是有限变量、无限约束的非凸半无限规划。

- Blankenship–Falk, [Infinitely constrained optimization problems](https://doi.org/10.1007/BF00934096)：经典自适应离散/约束生成框架。
- Seidel–Küfer, [Adaptive discretization with quadratic convergence](https://arxiv.org/abs/1910.13798)：把经典离散与双层局部信息结合。
- Jungen–Djelassi–Mitsos, [Adaptive discretization-based algorithms](https://link.springer.com/article/10.1007/s00186-022-00792-y)：现代收敛框架与应用讨论。

最适合本题的工程化版本是 exchange loop：

1. 在有限测试点集上优化光源；
2. 对所得构型全局/多起点寻找新的最暗点；
3. 加入违反最严重的最暗点；
4. 保留活跃约束并重复；
5. 最后用独立连续域证书收尾。

它与 Remez 等振荡思想相似，但本题外层非凸、内层多极小点，不能期待经典一维 Remez 的全局保证。

## 5. 运筹学：设施选址与连续保护区

### 多个逆平方源的累积影响

Drezner–Drezner–Kalczynski, [Multiple obnoxious facilities location: A cooperative model](https://doi.org/10.1080/24725854.2020.1753898) 研究多个设施对社区的累积负面影响，目标是最小化最受影响社区的总 nuisance。论文报告通用 SNOPT/IPOPT 多起点表现差，专门的 Voronoi 启发式明显更好。

Miklas-Kalczynska–Kalczynski, [the case of protected areas](https://doi.org/10.1007/s10287-024-00503-4) 把离散社区扩展到二维连续保护区，用迭代加入最不利点的方式处理连续区域。

它们与本题共享：多源叠加、逆平方、最坏点和迭代不利点。符号与几何角色相反：它们让有害源远离需求区，本题让有益源提高全域最低势。Voronoi/不利点生成方法可借，解与结论不可直接翻转。

Coletti–Kalczynski–Drezner, [On the combined inverse-square effect of multiple point sources](https://arxiv.org/abs/2305.02912) 给出外部点源包围闭区域时极值位于边界的维数结论。本题的源在区域内部且有奇点，不满足其设置。

## 6. 放射治疗 / 剂量布点

近距离放疗把多个放射性 seed 的剂量贡献相加并优化位置：

- Lee–Zaider, [Treatment planning for brachytherapy](https://pubmed.ncbi.nlm.nih.gov/10071881/)：整数规划、branch-and-bound 与遗传算法优化 seed 布点和离散剂量约束。
- AAPM TG-43U1, [brachytherapy dose-calculation protocol](https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1118/1.1646040)：真实剂量还包含点/线源几何函数、径向剂量函数和各向异性函数。

可借：离散候选位置、稠密剂量矩阵、上下剂量约束、整数/连续混合搜索、严格区分候选生成与剂量验证。

不可借：把临床剂量简化成同一平面上的纯 `1/r^2`；真实核不是本题核，也不能把数学玩具包装成临床设计结论。

## 7. 真实照明与可见光通信

[IES inverse-square law](https://ies.org/definitions/inverse-square-law/) 指出，点源对法向表面的照度在适用条件下按 `I/d^2` 变化；有限光源只有在距离相对源尺寸足够大时才近似点源。

Moreno–Muñoz–López, [Designing LED arrays for uniform near-field irradiance](https://opg.optica.org/abstract.cfm?uri=ao-45-10-2265) 研究不完美 Lambertian LED 阵列的均匀近场照明和最优间距。

真实“天花板到工作面”模型通常有高度、入射余弦、光强方向图、遮挡和反射。若源与接收面相隔高度 `h`，距离是 `sqrt(h^2+r^2)`；即使各向同性点源，水平面照度还带入射余弦。因此 Friedman 的同平面 `1/r^2` 是抽象 Riesz 模型，不是字面房间照明。

正确用法：把真实照明作为后续核函数族与稳健布局实验，而不是宣称当前记录可以直接用于灯具布置。

## 8. 无线能量、传感器与覆盖

- Huang–Lau 等方向的 [optimal deployment of power beacons](https://arxiv.org/abs/2012.04467) 研究网络级能量 outage 约束下的 power beacon 布置。
- Jia–Zhou, [Power beacon placement](https://arxiv.org/abs/2101.12415) 研究 backscatter 网络中的保障覆盖。

自由空间接收功率在理想视距下有线性尺度 `1/r^2`，但工程模型常用路径损耗指数、阴影、随机用户和 outage 概率。可借 max-min/robust deployment 的实验设计，不是同一确定性连续势问题。

## 9. 计算机图形学、半色调与 blue noise

这是第二条最有潜力的算法接口。

### Gray Pattern Problem

Brimberg–Kalczynski–Drezner, [Different formulations of the gray pattern problem](https://doi.org/10.1016/j.ejor.2024.01.048) 研究周期重复的黑点图案，其中经典代价直接使用最近周期像之间的倒平方距离。目标是视觉上均匀的灰度分布。

它不是 polarization：它主要是离散、周期、点对点色散/能量目标，不是连续区域上的最小总势。但其周期构型、逆平方代价、对称性与多种等价表述都适合产生初始点集。

### Blue-noise / optimal transport

- de Goes et al., [Blue Noise through Optimal Transport](https://www.geometry.caltech.edu/BlueNoise/bluenoise.html)：容量约束 power diagram 与最优传输，作者页提供源码。
- Chen et al., [Variational blue noise sampling](https://doi.org/10.1109/TVCG.2012.94)：连续变分与准 Newton 优化。
- Ahmed et al., [A Simple Push-Pull Algorithm](https://graphics.uni-konstanz.de/publikationen/Ahmed2016SimplePushPull/)：同时控制最近点间距、覆盖半径和 Voronoi 容量。
- Ahmed–Ren–Wonka, [Gaussian Blue Noise](https://arxiv.org/abs/2206.07798)：Gaussian 核优化生成高质量点集。
- Wong–Wong, [N-body simulation-based blue noise](https://research.monash.edu/en/publications/blue-noise-sampling-using-an-n-body-simulation-based-method/)：电荷粒子模型并有 GPU 实现。

可借：三角/blue-noise 初值、等面积 Voronoi 初始化、GPU 粒子批处理、频谱和 Voronoi 统计。

不可借：blue-noise 能量高就推出 polarization 高。所有初值仍须经过本题的 adversarial minimum 优化与严格验证。

## 10. 现有代码与不重复造轮子原则

### 直接复用

1. 固定构型证书：`square-riesz-polarization-certificates`，固定公开 commit/release 并保留许可。
2. 初值：BNOT 源码、SciPy Voronoi/Delaunay，或自行实现很薄的三角晶格裁剪；不重写完整最优传输库。
3. 数值求导/批处理：PyTorch CUDA；`N<=52` 时直接张量广播足够，不必先引入 FMM/KeOps。
4. 局部精化：SciPy `L-BFGS-B`/`SLSQP` 或 PyTorch LBFGS；只在性能证据出现后更换。

### 暂不引入

- FMM：当前源数小，主要瓶颈是候选数、测试点数和非凸盆地，不是单次 `N` 体求和。
- 通用 MIP 全局求解：奇异核使现成 polarization MIP 不能直接应用；小 `N` 全局证明可作为独立后续课题。
- 神经网络生成器：在有稳定 benchmark 与传统基线前没有必要。

## 11. 目前最可信的研究缺口

按证据强弱排序：

1. **公开工程缺口（强）**：已有严格坐标证书，但没有公开候选生成器、随机种子和完整搜索日志。
2. **有限尺寸数据缺口（较强）**：没有找到单位正方形 `s=2`、跨大量 `N`、同时包含所有最暗点与结构统计的系统数据集。
3. **临界次阶项（候选）**：针对性检索未找到单位正方形 `P_N=pi*N*log N+...` 的次阶展开；这必须继续做引用链和作者文献核查后才能称为开放问题。
4. **边界层/聚簇转变（候选）**：公开数值显示边缘近重合点与对称性变化，但尚未找到理论解释。也可能只是有限 `N` 或参数化效应。

任何论文级 novelty 声明前，仍须：追踪核心论文的全部引用与被引、检查专著第 14 章、检索非英文文献和学位论文，并最好向该领域作者询问；外部联系需用户批准。
