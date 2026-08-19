# Unit-square Riesz 2-polarization

状态：**阶段 A 已严格超过 7 个公开固定构型；N=3/N=5 另超过 Friedman
页面显示值；N=3 已有 0.153% 全局夹逼**（2026-08-19）。

本项目研究 Erich Friedman 的 [Maximizing Minimum Light Intensity](https://erich-friedman.github.io/packing/light/)：在单位正方形中放置 `N` 个允许重合的单位点源，最大化

\[
P_N=\max_{a_1,\ldots,a_N\in[0,1]^2}\;\min_{x\in[0,1]^2}
\sum_{i=1}^N \frac{1}{\lVert x-a_i\rVert^2}.
\]

数学上，这是二维临界指数 `s=d=2` 的离散 Riesz polarization（Chebyshev 型 max-min 势）问题。

## 与 3D 游戏照明的关系

这不是牵强类比。在无阴影、线性叠加、各向同性点光源和平方反比衰减的简化
游戏场景中，若玩家活动域是一块平面，自动摆放固定数量点光源以最大化活动域的
最低光照，目标函数与本题同构。Unreal Engine 的 Point Light 当前直接支持
[Inverse Squared Falloff](https://dev.epicgames.com/documentation/unreal-engine/point-lights-in-unreal-engine?lang=en-US)，
其中 Attenuation Radius 对贡献做截断。

实际游戏核可逐层扩展为

\[
L(x)=\sum_i V_i(x)\,w_i\,A(\lVert x-a_i\rVert)\,
B(n_x,a_i-x,\text{material}),
\]

其中 `V` 是遮挡/阴影，`A` 是带半径截断或软化的衰减，`B` 包含表面法线、
聚光方向与材质响应。若灯固定在地面上方高度 `h`，第一步可研究正则核
`1/(r^2+h^2)`；水平 Lambert 面的物理直射照度还会多一个入射余弦因子。

因此本项目有两条应用支线：

1. **自动点光源布局**：在 navmesh、可行走表面或设计师标注的重要区域上做
   最坏点照明保障；这是本问题最直接的推广。
2. **Light Probe/Volumetric Lightmap 布局**：同样涉及有限采样预算下的空间覆盖，
   但目标是插值误差、光照变化与内存，不是倒平方光强和。Unity 6 手册明确要求
   在光照变化大的区域放更多探针，并权衡探针数量与内存；Unreal 的 Volumetric
   Lightmap 也在几何附近自适应加密。它们可借用 blue-noise/CVT/exchange 工具，
   但不能声称与本题数学等价。

应用上的成功指标应是：相同灯数/渲染预算下提高 navmesh 的最低照度、减少黑区，
并能处理遮挡和截断半径；`N=3` 纪录则作为小型可验证回归基准。

## 为什么只是“有条件 GO”

2026-07-28 已有同题公开项目：
[Driedsandwich/square-riesz-polarization-certificates](https://github.com/Driedsandwich/square-riesz-polarization-certificates)。它给出：

- `N=7,8,10,11,12` 及 `N=14..52` 的有限小数坐标；
- 每个固定构型的两套精确有理 branch-and-bound 连续域下界证书；
- 大量明显高于 Friedman 当前页面显示值、但尚未全部处理或提交的构型；
- `N=9` 与 `N=13` 的未改进搜索结果；
- **不包含**候选生成/优化代码、随机种子和完整搜索轨迹。

因此：

- 如果目标只是利用官网更新滞后“刷新页面数字”，**不值得做**；
- 如果目标是公开可复现的 GPU 候选搜索器，并用数据研究有限尺寸、边界层、晶格与聚簇转变，**值得做**；
- 任何纪录比较必须同时对照 Friedman 页面与上述公开仓库。官网未处理的公开结果也是先行结果。

## 两阶段目标

### 阶段 A：真实刷新公开前沿

1. 固定并复现官网与公开证书库的基线。
2. 优先扫描证书库空缺的 `N=3,4,5,6,9,13`。
3. 对其他 `N`，必须超过证书库的当前值，而不只是超过官网显示值。
4. 用 GPU 批量生成候选，用连续域对抗搜索找真实最暗点。
5. 对最终有限小数坐标运行独立严格证书。

阶段 A 的成功判据不是“浮点打分更高”，而是：

- 严格超过当时可查到的公开 incumbent；
- 发布文本本身通过连续域下界验证；
- 候选搜索、种子、日志和失败实验可复现；
- 对外提交前获得人类逐次批准。

## 当前结果：N=3

RTX 5090 批量 soft-min 搜索先找到非对称候选；反射对称精炼与 100 位 KKT
求解随后得到有限小数构型

```text
(0.1198267882563919621, 0.4023192320163492298)
(0.8801732117436080379, 0.4023192320163492298)
(0.5,                   0.9801974656079521770)
```

两种精确有理 full-square branch-and-bound 均证明

\[
7.56838963\le I(X)\le
7.568389640029690214292220397565988\ldots,
\]

其中上界是在角点 `(0,0)` 对冻结的字面小数坐标精确求值得到。Friedman 当前
显示值为 `7.507+`，所以新下界严格超过网页显示数；由于网页没有给出旧构型的
完整坐标和严格上界，这里不把它表述成对旧未舍入构型的严格胜出。

- spectral Hessian 证书：141 次分割，最大深度 32；
- componentwise Hessian 区间证书：123 次分割，最大深度 31；
- 两种通用验证器分别精确复现上游 `N=7` 的 499/489 次分割；
- 高精度 KKT 根有四角和下边中点五个等高活跃最暗点；冻结到 19 位小数后，
  角点 `(0,0)` 比下边中点低约 `3.19e-19`，这不影响已证明的下界；
- 完整六维 KKT 梯度残差约 `4.5e-16`；考虑下边最暗点移动后的临界
  二阶特征值约为 `-20.90`、`-1.63`，支持严格局部最优猜想。

此外，有限见证点的六维源盒精确覆盖已经独立验证

\[
P_3\le 7.58.
\]

证书含 816 个初始根盒、1,304,240 个叶盒；最紧叶的精确界约
`7.579999987168342`。因此当前严格夹逼为

\[
7.56838963\le P_3\le7.58,
\]

相对缺口约 `0.153%`。纯有理 Krawczyk 检验还证明：半径 `1e-40` 的对称
七维 KKT 盒内存在唯一真根，且三个活跃轨道权重为正。这仍**不证明 N=3
全局最优或完整局部最优**；剩余局部活跃集隔离与二阶充分条件见
[PROOF_SKELETON.md](./PROOF_SKELETON.md)。未经用户批准，不进行任何对外提交。

## 当前结果：N=5

通用 GPU 搜索还给出一个此前公开库未覆盖的 `N=5` 构型。第二轮连续域求值为
`22.063083016036774`，两套独立精确验证器分别用 160/177 次分割证明

\[
I(X)\ge22.06>21.342,
\]

所以新下界严格超过 Friedman 当前显示数。旧网页同样没有提供足以证明未舍入旧
构型上界的资料；该构型尚未做针对性精炼，不能据此推断 `P_5` 接近 `22.06308`。

## 当前结果：N=29..35

在证明 `N=3` 的同时，RTX 5090 对榜单高端做了公开 incumbent 邻域搜索。比较
对象不是 Friedman 页面上较旧的显示值，而是
[公开严格证书库](https://github.com/Driedsandwich/square-riesz-polarization-certificates)
中每个既有有限小数构型的**严格上界见证**。这样，新构型的严格下界一旦超过它，
就必然胜过既有构型，而不是利用不同证书精度或网页更新延迟。

| `N` | 既有构型严格上界（向上舍入至 `1e-12`） | 新的严格下界 | 新构型点值上界 |
|---:|---:|---:|---:|
| 29 | 272.495973646473 | **282.8** | 282.8569252862 |
| 30 | 285.326742383550 | **285.34** | 285.3456853299 |
| 31 | 294.208893270426 | **305.2** | 305.2983669116 |
| 32 | 304.280799129879 | **317.1** | 317.2038189279 |
| 33 | 311.665641329667 | **330.5** | 330.5954794802 |
| 34 | 323.409928097309 | **337.8** | 337.9062358231 |
| 35 | 329.708966808636 | **347.1** | 347.1957223313 |

每个下界都对冻结的字面小数坐标分别通过 spectral 和 componentwise 两套精确
有理 full-square branch-and-bound。表中最后一列是在一个字面小数最暗点精确求值，
所以每个新构型本身也被夹在约 `0.0057..0.107` 宽的区间内。这仍只证明固定构型的
分数，不证明对应 `P_N` 的全局最优性。

这轮不是七次独立的随机奇迹：高 `N` incumbent 已呈清晰边界层/近晶格结构，围绕
它做小扰动 soft-min 优化很容易同时修正大量暗区；旧公开项目发布了严格构型和
验证器，却没有发布候选搜索代码、种子或完整轨迹。`N=30` 只严格提高约
`0.01326`，也说明大幅跃升不是随 `N` 自动发生；更像是若干旧构型落入较差盆地，
而我们补上了搜索层。旧数据中的近重合点与较差分数相关，但上游文档明确说明
`minimum_source_separation` 只是测得的描述字段，不是优化约束；因此当前证据支持
“盆地/搜索质量”假说，不支持“移除了最小间距约束”的说法。这些 `N` 因而也是
研究有限尺寸和边界层的更好数据，而不只是榜单分数。

### 阶段 B：从数据提取数学

重点不预设结论，先检验以下问题：

1. `P_N - pi*N*log(N)` 的有限尺寸修正是什么量级？
2. 内部是否趋向三角点阵？正方形边界、角点怎样形成边界层？
3. 对称构型何时失稳为非对称构型？
4. 近重合点/多重光源何时出现，是真正最优结构还是数值参数化产物？
5. 最暗点的数目、位置与 KKT 权重能否给出可检验的“等势/力平衡”规律？
6. 改变 Riesz 指数 `s` 或采用有高度的物理照明核时，结构是否发生相变？

## 已知理论基线

Borodachov–Bosuwan 证明，对面积为 1 的二维集合，在 `s=d=2` 时

\[
\frac{P_N}{N\log N}\longrightarrow \pi,
\]

并且渐近最优点集按面积测度均匀分布。见
[Asymptotics of discrete Riesz d-polarization](https://arxiv.org/abs/1307.1160)。

这只给出首项，不回答有限 `N` 的正方形边界结构、次阶项或具体最优构型。

## 第一阶段目标优先级

| 优先级 | `N` | 原因 |
|---|---:|---|
| A | 9, 13 | 官网仍为 Erich Friedman 构型；公开证书项目明确报告已复现盆地但未改进，是最干净的真实挑战 |
| A | 3, 4, 5, 6 | 公开证书库未覆盖；小规模也更可能发现可证明结构，但 GPU 优势较小 |
| B | 7, 8, 10..12, 14..36 | 官网显示值可能已被公开库内部值大幅超过；只有超过公开库才算新结果 |
| B | 37..52 | 有公开严格下界，但不是 Friedman 当前榜项；适合扩展数据研究，不宜包装成官网纪录 |
| C | >52 | 可生成渐近数据与新实例，但首先是研究扩展，不是现有榜单刷新 |

当前已知的空缺基线：

- `N=9`: 已复现数值盆地 `57.38839577734211`，公开搜索未跨过 `57.389`；
- `N=13`: 已复现数值盆地 `96.34710139024597`，无对称精化未得到稳健改进。

来源：[公开库的 non-improvement-results.csv](https://github.com/Driedsandwich/square-riesz-polarization-certificates/blob/main/data/non-improvement-results.csv)。

## 文件

- [INDEPENDENT_AUDIT_REPORT.md](./INDEPENDENT_AUDIT_REPORT.md)：面向独立 AI/人工复核者的声明边界、干净克隆复现命令、证书哈希与对抗测试清单。
- [INDEPENDENT_AUDIT_FINDINGS.md](./INDEPENDENT_AUDIT_FINDINGS.md)：独立 AI 的原始审查结果；提交 `2ef5eb9` 保留了审查时的逐字快照。
- [LITERATURE_MAP.md](./LITERATURE_MAP.md)：数学、优化、物理和工程的文献地图及迁移边界。
- [MODEL_AND_ROADMAP.md](./MODEL_AND_ROADMAP.md)：GPU 求解器、严格验证与研究数据路线。
- [RESEARCH_LOG.md](./RESEARCH_LOG.md)：检索过程、直接命中、负面结果和后续待查项。
- [PROOF_ROI.md](./PROOF_ROI.md)：`N=3` 局部/全局最优证明的价值、风险、工具匹配与早停闸门。
- [PROOF_SKELETON.md](./PROOF_SKELETON.md)：局部 KKT/包络 Hessian 与全局有限见证证明骨架。
- [scripts/record_hunt.py](./scripts/record_hunt.py)：GPU 生产者与 CPU 连续域验证并行、可恢复的纪录搜索器。
- [data/candidates/n03_symmetric.json](./data/candidates/n03_symmetric.json)：冻结坐标、高精度 KKT 与连续域诊断。
- [data/candidates/n05_hunt_best.json](./data/candidates/n05_hunt_best.json)：首轮严格超过 Friedman 显示值的 `N=5` 构型。
- [runs/record_hunt_large_20260819/summary.json](./runs/record_hunt_large_20260819/summary.json)：高 `N` 首批正式扫描的种子、连续域复核与基线比较。
- [runs/record_hunt_n30_20260819/summary.json](./runs/record_hunt_n30_20260819/summary.json)：同参数 `N=30` 补充扫描，防止只汇报大幅成功项。
- [data/candidates](./data/candidates)：`N=29..35` 的冻结字面小数构型。
- [data/certificates](./data/certificates)：上述七个构型的 spectral/componentwise 精确下界运行记录。
- [data/certificates/n03_target_7_56838963.json](./data/certificates/n03_target_7_56838963.json)：spectral 精确下界证书运行结果。
- [data/certificates/n03_componentwise_target_7_56838963.json](./data/certificates/n03_componentwise_target_7_56838963.json)：componentwise 精确下界证书运行结果。
- [data/certificates/n03_global_upper_7_58.zip](./data/certificates/n03_global_upper_7_58.zip)：全六维源空间的精确有限见证上界树分发包；解压为 `n03_global_upper_7_58.json` 后 SHA-256 为 `6F6F936767A73A548FCAFDD18A4F739AED4AE201B093271009BEC4DEE2E0ACEF`。原始 JSON 约 95.9 MiB，不重复纳入 Git。
- [data/certificates/SHA256SUMS](./data/certificates/SHA256SUMS)：压缩包与解压后原始证书的校验值。
- [data/certificates/n03_symmetric_kkt_krawczyk.json](./data/certificates/n03_symmetric_kkt_krawczyk.json)：对称 KKT 根的纯有理 Krawczyk 包含证书。
- [scripts/verify_global_upper_certificate_cleanroom.py](./scripts/verify_global_upper_certificate_cleanroom.py)：审查后加固的第二套纯标准库全局上界验证器。
- [data/certificates/n05_spectral_target_22_06.json](./data/certificates/n05_spectral_target_22_06.json)：`N=5` spectral 精确下界证书。
- [data/certificates/n05_componentwise_target_22_06.json](./data/certificates/n05_componentwise_target_22_06.json)：`N=5` componentwise 精确下界证书。
- [requirements-audit.txt](./requirements-audit.txt)：CPU 精确复核与 CI 所需依赖，不含 PyTorch。
- [requirements-search.txt](./requirements-search.txt)：GPU 搜索依赖，包含 PyTorch。
- [requirements.txt](./requirements.txt)：兼容入口，等价于搜索环境。

## 当前硬件/环境

- GPU：NVIDIA GeForce RTX 5090，32607 MiB，compute capability 12.0。
- Python/PyTorch：`E:\math\.venv`，PyTorch 2.9.0+cu128，CUDA smoke test 已通过。
- 实测吞吐：`batch=1024, N=13, 129x129` 网格约 0.05485 秒/轮，约
  202 亿 source-point pairs/s，峰值分配约 528 MiB。
