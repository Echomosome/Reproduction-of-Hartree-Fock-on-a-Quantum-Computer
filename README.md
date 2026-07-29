# 二氮烯（HN=NH）Hartree–Fock 量子复现方案

版本：2026-07-29  
对象：二氮烯 cis–trans 异构化的两条路径  
主比较量：`RHF/STO-3G` 总能量  
量子表示：10 qubits、每个自旋分量 6 particles  
交付范围：18 个论文印刷几何、180 条线路、平台 JSON 处理、1-RDM、
HF 能量、Gaussian 输入、误差分解、bootstrap 与自动测试

---

## 0. 一句话结论

本包把每个二氮烯构象的 Cartesian 坐标逐点变成

```text
AO 积分
→ 两次预备 SCF
→ 冻结最低两个双占据轨道
→ 10 模/6 粒子 RHF
→ Givens 线路
→ 10 组测量
→ 1-RDM
→ HF 总能量
→ 与同几何 Gaussian RHF 比较
```

而且不复用前一构象的积分、轨道或角度。所有固定参数的来源或推导列在
[`PARAMETER_PROVENANCE.csv`](PARAMETER_PROVENANCE.csv)，所有随构象变化的
数值列在 [`results/parameter_evolution.csv`](results/parameter_evolution.csv)
及每个构象自己的 `reference/`、`circuits/` 文件中。

包内精确门级模拟在 18 个构象上都重构出目标 1-RDM，最大线路闭合误差为
`2.3313e-9`；HF 能量与 10 模经典参考的最大差为
`2.84e-11 mHa`。固定随机种子的每 setting 1000 shots 演示中，各点
rank-6 能量误差为 `5.453–17.493 mHa`，所以它可辨别约 40 mHa 的路径差异，
但**没有达到逐点 1 kcal/mol 化学精度**。

Gaussian 需要用户本地许可证，本包没有伪造 Gaussian 运行结果；已经给出
18 个可直接提交的 `.gjf`、运行命令和 `SCF Done` 解析器。

---

## 1. 先明确“复现了什么”

### 1.1 本包直接复现的对象

1. 论文 Supplementary Information Appendix J、Tables III/IV 中印刷的
   9 个面外路径几何和 9 个面内路径几何；
2. 每个固定几何的闭壳层 `RHF/STO-3G` 单点能量；
3. 论文所述“先做两次 SCF 更新，再冻结最低两个轨道”的 10 模活性空间；
4. 该活性空间 RHF Slater 行列式的量子制备、1-RDM 测量和 HF 能量；
5. 量子结果、10 模冻结核参考、完整 12 轨道 PySCF RHF 与 Gaussian RHF
   之间的逐层误差。

### 1.2 本包没有偷换成下列命题

- **不是相关能量复现。** 当前 ansatz 是单个 Slater 行列式，正确基准是
  RHF，不是 FCI、CCSD 或实验热化学能。
- **不是完整论文硬件原始数据复刻。** 本包生成可提交线路和标准 JSON
  管线，但没有制造量子硬件后端、校准或 counts。
- **不是论文完整 20 点路径的 Cartesian 复刻。** OpenFermion manifest
  列出每条路径 20 个坐标标签，Appendix J 只印刷其中 9 个硬件点的
  Cartesian 坐标。本包只对可逐坐标核查的 9+9 点声称直接复现。
- **不是论文线路的逐门二进制副本。** 论文报告 `N+1=11` 个测量 setting
  和典型 `50 sqrt(iSWAP)+80 RZ`。本包利用 RHF 轨道为实数这一条件，
  把 45 个实非对角元按 `K10` 完美匹配分成 9 组，加 1 组对角测量，
  因而是 10 个 setting；重新编译后通常为
  `48 SQISWAP+72 RZ`。两者测量同一个实 1-RDM，但电路不是逐门相同。
- **不是 Gaussian 松弛扫描替代论文路径。** `gaussian/optional_relaxed_scan/`
  是独立扩展，便于重新生成一条 Gaussian 路径；不能冒充论文印刷几何。

### 1.3 四个能量层次必须分开

| 记号 | 定义 | 用途 |
|---|---|---|
| `E_full^PySCF` | 同几何、完整 12 个 STO-3G 空间轨道的 PySCF RHF | 独立经典参考 |
| `E_active^RHF` | 两次预备 SCF 后冻结 2 个轨道的 10 模 RHF | 量子线路的严格目标 |
| `E_Q` | 测得 1-RDM 经 rank-6 投影后代入同一活性空间积分的能量 | 量子/模拟结果 |
| `E_full^Gaussian` | 同几何、完整 12 轨道 Gaussian RHF 的 `SCF Done` | 外部程序交叉核对 |

不能把 `E_Q-E_full^Gaussian` 全部叫作“量子噪声”，因为其中还含冻结核近似
和经典程序差异。

---

## 2. 目录和审计入口

```text
Diazene_HF_Reproduction/
├── README.md
├── PARAMETER_PROVENANCE.csv
├── CHECKSUMS.sha256
├── requirements.txt
├── requirements-chemistry.txt
├── code/
│   ├── diazene_core.py
│   ├── build_references.py
│   ├── generate_circuits.py
│   ├── simulate_data.py
│   ├── analyze_geometry.py
│   ├── analyze_scan.py
│   ├── bootstrap_mechanism_gap.py
│   ├── summarize_parameters.py
│   ├── generate_gaussian_inputs.py
│   ├── parse_gaussian.py
│   └── run_demo.py
├── data/
│   ├── geometries_paper.json
│   ├── paper_scan_grid.json
│   ├── platform_json_template.json
│   ├── ideal/<geometry>/<setting>.json
│   └── sample_1000_shots/<geometry>/<setting>.json
├── reference/
│   ├── <geometry>.npz
│   ├── <geometry>.json
│   └── reference_summary.csv
├── circuits/
│   └── <geometry>/
│       ├── manifest.json
│       ├── givens_parameters.csv
│       ├── measurement_map.csv
│       ├── gate_body_radians/*.txt
│       └── originir_full/*.originir
├── gaussian/
│   ├── RUN_GAUSSIAN.md
│   ├── inputs/*.gjf
│   └── optional_relaxed_scan/*.gjf
├── results/
│   ├── parameter_evolution.csv
│   ├── ideal_scan/
│   ├── sample_1000_shots_scan/
│   └── ideal_1000_shot_gap_bootstrap/
└── tests/run_tests.py
```

可用 `sha256sum -c CHECKSUMS.sha256` 核对包内文件；checksum 清单本身不对
自身做递归校验。

审计时建议按以下顺序：

1. `PARAMETER_PROVENANCE.csv`：固定参数为什么是这个值；
2. `data/geometries_paper.json`：每个原子坐标；
3. `reference/<geometry>.json`：积分、两次 SCF 记录、冻结核常数和 RHF 结果；
4. `circuits/<geometry>/givens_parameters.csv`：每个 Givens/RZ 参数；
5. `circuits/<geometry>/measurement_map.csv`：每个 1-RDM 元素由哪两个输出模式给出；
6. `results/<...>/summary.json`：从 JSON 到能量的全部中间量。

---

## 3. 每个固定科学参数从哪里来

完整逐项表见 `PARAMETER_PROVENANCE.csv`。这里给出计算链中最关键的参数。

### 3.1 分子、电荷和粒子数

二氮烯为 `H1-N1=N2-H2`。中性体系的电子数为

\[
N_e=Z_{\rm H}+Z_{\rm N}+Z_{\rm N}+Z_{\rm H}
=1+7+7+1=16.
\]

使用闭壳层 RHF、总自旋 `S=0`，所以多重度

\[
2S+1=1,
\]

并且

\[
N_\alpha=N_\beta=N_e/2=8.
\]

Gaussian 的电荷/多重度行因此是

```text
0 1
```

### 3.2 为什么 STO-3G 是 12 个空间模式

在 STO-3G 最小基中，每个 H 提供一个 `1s` 空间基函数，每个 N 提供
`1s,2s,2p_x,2p_y,2p_z` 五个空间基函数，因此

\[
N_{\rm spatial}=2\times 1+2\times 5=12.
\]

代码还用 `mol.nao_nr()==12` 做硬性检查；若安装的基组或原子顺序错误，
构建会立即失败。

### 3.3 为什么量子部分是 10 qubits、6 particles

公开的 OpenFermion 二氮烯 manifest 指定：先做两次 SCF 更新，再冻结最低
两个空间轨道。两个被冻轨道均为双占据，因此对每个自旋分量：

\[
N_{\rm active\ modes}=12-2=10,
\]

\[
\eta_{\rm active}=8-2=6.
\]

本方案按一个自旋分量编码，一个活性空间模式对应一个 qubit，因此使用
10 qubits、6 particles。RHF 中 alpha 与 beta 的空间 1-RDM 相同，
最终闭壳层能量在能量泛函中显式加入自旋因子 2，而不是再运行一套独立线路。

### 3.4 几何从哪里来

`data/geometries_paper.json` 的 18 个 Cartesian 结构逐数字转录自
Appendix J Tables III/IV：

- 面外路径：H1–N1–N2–H2 二面角扭转；
- 面内路径：论文定义的平面内 H 旋转坐标；
- 坐标单位 Å，保留论文印刷的 5 位小数；
- 路径标签保留论文印刷的 3 位小数。

`code/diazene_core.py::geometry_metrics` 会重新计算三条键长、两个键角和
H1–N1–N2–H2 二面角。面外标签与重算二面角应在坐标印刷精度内一致。
面内标签是路径旋转坐标，不应误当成 HNNH 二面角；其构象本身近似共面。

`data/paper_scan_grid.json` 另存 OpenFermion manifest 中完整 20 点标签。
缺少印刷 Cartesian 坐标的 22 个点不会被静默线性插值，因为那会改变
势能面并破坏“来源可追溯”。

### 3.5 数值阈值与硬件参数

- `conv_tol=1e-12 Ha`、`conv_tol_grad=1e-10`、最大 512 个 SCF 周期：
  是为了生成稳定基准的数值选择，不是分子物理参数。
- `density_tolerance=1e-12`、`energy_tolerance=1e-13 Ha`：
  是本地活性空间 RHF 的停止条件。
- `%mem=4GB`、`%nprocshared=8`：
  只是 Gaussian 资源默认值；只要内存足够，它们不改变科学结果。
- `seed=5210`：
  只用于让 1000-shots 示例可逐位重跑，不是论文参数。
- `1000 shots/setting`：
  是本包的低成本演示。论文硬件数据使用更高的采样预算，因此不能把本包
  的 1000-shot 置信区间说成论文原始误差条。

---

## 4. 构象变化如何逐层改变所有参数

对第 `k` 个构象，Cartesian 坐标记为

\[
\mathbf R_k=\{\mathbf R_A^{(k)}\}_{A=1}^{4}.
\]

每个构象都完整重跑以下映射：

\[
\mathbf R_k
\rightarrow
\{E_{\rm nuc}^{(k)},S^{(k)},h^{(k)},(pq|rs)^{(k)}\}
\rightarrow
C_{\rm pre}^{(k)}
\rightarrow
\{E_0^{(k)},h_{\rm act}^{(k)},g_{\rm act}^{(k)}\}
\rightarrow
\gamma_{\rm RHF}^{(k)}
\rightarrow
\{\theta_\ell^{(k)}\}
\rightarrow
\text{circuits}^{(k)}.
\]

这里没有任何一步把 `k-1` 的积分或角度直接复制到 `k`。

### 4.1 核排斥能

由 Cartesian 坐标直接得到

\[
E_{\rm nuc}^{(k)}
=\sum_{A<B}\frac{Z_AZ_B}
{|\mathbf R_A^{(k)}-\mathbf R_B^{(k)}|},
\]

其中距离在实际积分程序中转换为 bohr。构象改变任意原子坐标，所有核间距
和 `E_nuc` 都会随之改变。

### 4.2 AO 重叠、一电子和二电子积分

\[
S_{\mu\nu}^{(k)}=\langle\chi_\mu^{(k)}|\chi_\nu^{(k)}\rangle,
\]

\[
h_{\mu\nu}^{(k)}
=\left\langle\chi_\mu^{(k)}
\left|-\frac12\nabla^2-\sum_A
\frac{Z_A}{|\mathbf r-\mathbf R_A^{(k)}|}
\right|\chi_\nu^{(k)}\right\rangle,
\]

\[
(\mu\nu|\lambda\sigma)^{(k)}
=\iint\frac{
\chi_\mu^{(k)}(1)\chi_\nu^{(k)}(1)
\chi_\lambda^{(k)}(2)\chi_\sigma^{(k)}(2)}
r_{12}\,d1\,d2.
\]

所以即使基组名称始终是 STO-3G，积分数值也必须对每个几何重新计算。

### 4.3 两次预备 SCF 更新

初始自旋求和 AO 密度使用 PySCF 的确定性 `MINAO` guess：

\[
P^{(0)}=P_{\rm MINAO}.
\]

公开 manifest 说明“两步 SCF”，但没有给出初猜；因此 `MINAO` 是本包显式
记录的实现约定，不伪称是未公开的原始代码细节。

对 `t=0,1`，构造

\[
F^{(t)}=h+J[P^{(t)}]-\frac12K[P^{(t)}],
\]

求广义本征问题

\[
F^{(t)}C^{(t+1)}
=SC^{(t+1)}\varepsilon^{(t+1)},
\]

取最低 8 个轨道并更新

\[
P^{(t+1)}
=2C_{\rm occ}^{(t+1)}
\left(C_{\rm occ}^{(t+1)}\right)^T.
\]

恰好完成两次更新；这里不开 DIIS、不阻尼。每次能量和 Fock–density
对易子范数保存在
`reference/<geometry>.json::preliminary_scf_history`。

### 4.4 冻结最低两个轨道

在第二次预备 SCF 得到的正交 MO 基中，冻结索引 `i,j in {0,1}`。
冻结核常数为

\[
E_0
=E_{\rm nuc}
+2\sum_i h_{ii}
+\sum_{ij}\left[2(ii|jj)-(ij|ij)\right].
\]

活性空间有效一电子积分为

\[
h_{pq}^{\rm eff}
=h_{pq}
+\sum_i\left[2(pq|ii)-(pi|qi)\right],
\]

其中 `p,q` 只遍历剩余 10 个模式；二电子积分直接取活性块

\[
g_{pqrs}^{\rm act}=(pq|rs).
\]

代码实现位于 `build_references.py::frozen_core_integrals`。`E_0` 的四个组成
部分都写入 `frozen_core_components`，因此没有隐藏常数。

### 4.5 10 模活性空间 RHF

以

\[
\gamma^{(0)}=\operatorname{diag}(1,1,1,1,1,1,0,0,0,0)
\]

开始，构造一个自旋分量的闭壳层 Fock 矩阵

\[
F_{pq}
=h_{pq}^{\rm eff}
+2\sum_{rs}\gamma_{rs}(pq|rs)
-\sum_{rs}\gamma_{rs}(pr|qs).
\]

对角化 `F`，取最低 6 个本征向量 `C_occ`，再更新

\[
\gamma=C_{\rm occ}C_{\rm occ}^T.
\]

从第三周期开始使用最多 8 个历史向量的 Pulay DIIS。停止条件为

\[
\|\gamma^{(t+1)}-\gamma^{(t)}\|_F<10^{-12}
\]

且

\[
|E^{(t+1)}-E^{(t)}|<10^{-13}\ {\rm Ha}.
\]

每个周期的数据位于 `active_scf_history`；未在 512 周期内收敛会显式报错。

### 4.6 实际参数变化示例

下表不是拟合值，而是由包内坐标逐点重算：

| 构象 | 路径坐标/° | `E_nuc`/Ha | 完整 RHF/Ha | 10 模 RHF/Ha | 冻结偏差/mHa | Z 线路 Givens 数 |
|---|---:|---:|---:|---:|---:|---:|
| `oop_003p157` | 3.157 | 31.586740069 | -108.545081068 | -108.545080926 | 0.000142 | 24 |
| `oop_095p641` | 95.641 | 31.049789696 | -108.418133273 | -108.418133016 | 0.000257 | 24 |
| `oop_183p522` | 183.522 | 31.648407946 | -108.556668154 | -108.556668104 | 0.000051 | 24 |
| `ip_108p736` | 108.736 | 31.574586295 | -108.544830623 | -108.544830477 | 0.000146 | 24 |
| `ip_182p000` | 182.000 | 31.723477897 | -108.450990803 | -108.450989676 | 0.001127 | 24 |
| `ip_256p105` | 256.105 | 31.658344360 | -108.556849528 | -108.556849479 | 0.000049 | 24 |

18 点的键长、键角、核排斥能、冻结常数、`h1/ERI/gamma` 相邻差和线路角度
统计全部位于 `results/parameter_evolution.csv`。

Givens 角本身存在轨道符号/占据子空间 gauge 自由度：
`C_occ` 的某列乘 `-1` 不改变 \(\gamma\) 或能量，却可能使编译角跳变约
\(\pi\)。因此跨构象判断“物理变化”应优先比较

\[
\|\gamma_k-\gamma_{k-1}\|_F
\]

和能量，而不能只看某个裸 `theta` 是否连续。包中仍保留每个裸角，便于
逐门复核。

---

## 5. 从 RHF 占据轨道到量子线路

### 5.1 目标 Slater 行列式

令 `C_occ` 是 `10 x 6` 的实正交占据轨道矩阵，

\[
C_{\rm occ}^TC_{\rm occ}=I_6.
\]

目标态为

\[
|\Phi\rangle
=\prod_{a=0}^{5}
\left(\sum_{p=0}^{9}C_{pa}a_p^\dagger\right)|{\rm vac}\rangle.
\]

其单自旋 1-RDM 是

\[
\gamma_{pq}
=\langle\Phi|a_q^\dagger a_p|\Phi\rangle
=\sum_{a=0}^{5}C_{pa}C_{qa},
\]

即

\[
\gamma=C_{\rm occ}C_{\rm occ}^T.
\]

它应满足

\[
\gamma^T=\gamma,\quad
\operatorname{Tr}\gamma=6,\quad
\gamma^2=\gamma.
\]

### 5.2 初态

线路先执行

```text
X q[0]
X q[1]
X q[2]
X q[3]
X q[4]
X q[5]
```

得到

\[
|0000\,111111\rangle
\]

（打印 bit 顺序为 `q9...q0`）。这对应前 6 个计算模式占据。

### 5.3 Givens 旋转的定义

相邻模式 `p,p+1` 上使用实旋转

\[
G_{p,p+1}(\theta)
=
\begin{pmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{pmatrix}.
\]

编译器对 `C_occ^T` 做稳定 Givens QR 消元。若当前要处理的两个实数为
`a,b`，代码通过 `hypot(a,b)` 生成归一化的正弦/余弦，使其中一个元素
严格归零；线路按 QR 消元的逆序、逆角度执行。具体实现在
`diazene_core.py::real_givens_matrix_elements` 和
`slater_givens_decomposition`。

占据轨道内部的 `6 x 6` 旋转只改变 Slater 行列式的整体行列式相位，
不改变占据子空间，所以先作为 gauge 自由度消去。剩余必要的
occupied–virtual 旋转上限为

\[
\eta(n-\eta)=6(10-6)=24.
\]

每条线路的 `theta`、并行层、模式对和三个 RZ 参数都在
`circuits/<geometry>/givens_parameters.csv`，没有硬编码在 README 中。

### 5.4 原生 `SQISWAP/RZ` 分解

本包明确规定 `SQISWAP` 在单粒子子空间的矩阵是

\[
\sqrt{i{\rm SWAP}}
=\frac1{\sqrt2}
\begin{pmatrix}
1&i\\
i&1
\end{pmatrix}.
\]

在这个**特定符号约定**下，一个实 Givens 旋转编译为

```text
SQISWAP q[second],q[first]
RZ q[first],(pi + theta)
RZ q[second],(-theta)
SQISWAP q[second],q[first]
RZ q[first],(pi)
```

所有角均打印为 15 位十进制弧度。若平台的 `SQISWAP` 使用 `-i` 分支、
`RZ` 定义为 `exp(-i theta Z/2)` 而不是本包的模式相位约定，必须先做
两模式矩阵校准；不能原样照抄角度。

每条线路通常有：

\[
24\ {\rm Givens}
\Rightarrow
48\ {\rm SQISWAP}+72\ {\rm RZ},
\]

外加 6 个初态 `X`。`ip_164p947` 的一个对称 setting 有一个严格冗余旋转，
所以出现 23 个 Givens；这是消零结果，不是漏门。

### 5.5 为什么测量 setting 也重新编译

非对角元测量需要先把目标轨道对旋到对称/反对称输出模式。为了只使用相邻
双比特门，本包不在态制备后机械追加远程 swap 网络，而是把

\[
C_{\rm measured}=A_{\rm setting}C_{\rm occ}
\]

作为新的占据矩阵，整体重新做一次最优相邻 Givens 分解。因此每个 setting
都有独立角度，且必须随构象重新生成。

---

## 6. 为什么 10 组测量足以得到实 10×10 1-RDM

### 6.1 对角元

Z 基测量给出每个模式占据数

\[
\gamma_{ii}=\langle n_i\rangle.
\]

一条 `z_diagonal` 线路同时得到 10 个对角元。

### 6.2 一个实非对角元

对逻辑模式 `i,j`，定义输出模式

\[
b_L^\dagger
=\frac{a_i^\dagger+a_j^\dagger}{\sqrt2},
\qquad
b_R^\dagger
=\frac{-a_i^\dagger+a_j^\dagger}{\sqrt2}.
\]

则

\[
\langle n_L\rangle
=\frac12\left(
\gamma_{ii}+\gamma_{jj}+\gamma_{ij}+\gamma_{ji}
\right),
\]

\[
\langle n_R\rangle
=\frac12\left(
\gamma_{ii}+\gamma_{jj}-\gamma_{ij}-\gamma_{ji}
\right).
\]

本问题的轨道与 \(\gamma\) 都是实数，\(\gamma_{ij}=\gamma_{ji}\)，所以

\[
\boxed{
\gamma_{ij}
=\frac{\langle n_L\rangle-\langle n_R\rangle}{2}
}.
\]

这正是 `measurement_map.csv` 中每一行使用的估计式。

若使用磁场、复轨道或一般复酉变换，还必须增加带 `i` 相位的分析器以测
\(\operatorname{Im}\gamma_{ij}\)；本包的 10-setting 缩减不能直接外推。

### 6.3 45 个非对角元如何分成 9 组

10 个模式一共有

\[
\binom{10}{2}=45
\]

个独立实非对角元。偶数阶完全图 `K10` 可以分解成 `10-1=9` 个完美匹配；
每个匹配含 `10/2=5` 对互不重叠的模式，因此一条线路可并行测 5 个元素：

\[
9\times5=45.
\]

所以总数是

\[
1\ {\rm diagonal}+9\ {\rm matchings}=10\ {\rm settings}.
\]

`tests/run_tests.py` 会验证 45 条边恰好各出现一次，没有缺失或重复。

---

## 7. 从平台 JSON 到 1-RDM：每个数据操作

### 7.1 支持的输入

每个几何必须有 10 个 JSON，文件名与 manifest 的 `setting` 相同。
支持：

```json
{
  "status": "Completed",
  "shots": 1000,
  "counts": {
    "0000111111": 12
  }
}
```

或

```json
{
  "status": "Completed",
  "probabilities": {
    "0000111111": 0.012
  }
}
```

还支持常见的 `result.key/result.value` 格式。完整模板见
`data/platform_json_template.json`。

### 7.2 概率归一化

若输入为 counts，

\[
\hat p(b)=\frac{c_b}{\sum_{b'}c_{b'}}.
\]

若输入为 probability，代码先检查非负、有限、总和大于零，再除以总和。
当元数据没有 shots 时，代码尝试从概率网格反推最小整数分母；无法可靠
反推时，bootstrap 必须显式传 `--shots`。

### 7.3 bit 顺序

默认字符串从左到右为

```text
q9 q8 q7 q6 q5 q4 q3 q2 q1 q0
```

因此右端字符对应 `q0`。代码不会用“看起来更自然”的方式自动猜顺序；
必须与 manifest 一致。错误 bit 顺序一般仍会保持粒子数，却会把 1-RDM
行列整体置换，所以只做粒子数检查不足以发现该错误。

### 7.4 粒子数后选择

理想线路守恒一个自旋分量的粒子数 `N=6`。保留概率为

\[
P_{\rm pass}
=\sum_{b:\,|b|=6}\hat p(b),
\]

其中 `|b|` 是 bitstring 中 `1` 的个数。后选择分布为

\[
\hat p_{\rm sel}(b)
=
\begin{cases}
\hat p(b)/P_{\rm pass},&|b|=6,\\
0,&\text{其他}.
\end{cases}
\]

必须同时报告 `P_pass`；后选择不能把很低的保留率隐藏成“已修复数据”。

### 7.5 占据数

每个 setting 的模式占据估计为

\[
\langle n_i\rangle
=\sum_b \hat p_{\rm sel}(b)\,b_i.
\]

`z_diagonal` 将它们写入 \(\gamma_{ii}\)；9 个 matching 按第 6.2 节公式
写入 \(\gamma_{ij}\) 和 \(\gamma_{ji}\)。

### 7.6 物理性检查

原始矩阵报告：

\[
\operatorname{Tr}\gamma,\quad
\|\gamma-\gamma^T\|_F,\quad
\|\gamma^2-\gamma\|_F,
\]

以及全部自然占据数。有限 shots 会让本征值略小于 0 或大于 1，也会破坏
幂等性；这不是自动证明有硬件噪声，必须与同 shots 理想抽样比较。

### 7.7 rank-6 物理投影

先做 Hermitian 化

\[
\gamma_H=\frac{\gamma+\gamma^\dagger}{2},
\]

再对角化

\[
\gamma_H=V\Lambda V^\dagger.
\]

保留最大的 6 个本征向量：

\[
\gamma_{\rm proj}
=V_{[:,\,{\rm top}\ 6]}V_{[:,\,{\rm top}\ 6]}^\dagger.
\]

这给出 Frobenius 范数下最近的 rank-6 正交投影矩阵；理论依据是 Hermitian
矩阵的谱截断/Eckart–Young 型最优性。它强制

\[
\operatorname{Tr}\gamma_{\rm proj}=6,\qquad
\gamma_{\rm proj}^2=\gamma_{\rm proj}.
\]

报告中同时保留 raw 与 projected 结果。投影是明确的数据处理步骤，
不能把 projected 能量冒充“原始测量能量”。

### 7.8 Slater 保真度

设目标和重构占据子空间的正交基分别为 `C` 与 `C_tilde`，单自旋 Slater
保真度为

\[
F_\sigma
=|\det(C^T\widetilde C)|^2.
\]

RHF 两个自旋分量相同且独立时，闭壳层行列式保真度报告为

\[
F_{\rm closed}=F_\sigma^2.
\]

### 7.9 TV 距离

每个 setting 还与精确门级分布比较：

\[
D_{\rm TV}
=\frac12\sum_b
|\hat p_{\rm data}(b)-p_{\rm circuit}(b)|.
\]

TV 距离用于定位线路/采样偏差，不直接等于能量误差。

---

## 8. 1-RDM 如何给出 Hartree–Fock 能量

本包的约定是

\[
\gamma_{pq}=\langle a_q^\dagger a_p\rangle
\]

且所有矩阵为实对称。对一个自旋分量的 RHF 密度，活性空间总能量逐项为

\[
E[\gamma]
=E_0
+2\sum_{pq}h_{pq}^{\rm eff}\gamma_{qp}
+2\sum_{pqrs}\gamma_{pq}\gamma_{rs}(pq|rs)
-\sum_{pqrs}\gamma_{pq}\gamma_{rs}(pr|qs).
\]

代码精确对应：

```python
one_body = 2 * einsum("pq,qp->", h1, gamma)
coulomb = 2 * einsum("pq,rs,pqrs->", gamma, gamma, eri)
exchange = einsum("pq,rs,prqs->", gamma, gamma, eri)
E = E0 + one_body + coulomb - exchange
```

因子 2 来自 alpha/beta 两个相同自旋密度；exchange 只在同自旋间出现。
`summary.json::energy_components` 分开保存 `E0`、一电子、Coulomb、exchange
和总能量。

### 8.1 误差分解

定义

\[
\Delta E_{\rm meas}=E_Q-E_{\rm active}^{\rm RHF},
\]

\[
\Delta E_{\rm freeze}
=E_{\rm active}^{\rm RHF}-E_{\rm full}^{\rm PySCF},
\]

\[
\Delta E_{\rm program}
=E_{\rm full}^{\rm PySCF}-E_{\rm full}^{\rm Gaussian}.
\]

则

\[
\boxed{
E_Q-E_{\rm full}^{\rm Gaussian}
=\Delta E_{\rm meas}
+\Delta E_{\rm freeze}
+\Delta E_{\rm program}
}.
\]

这条恒等式是本包判断问题来源的主链：

- `ΔE_meas` 大：测量、bit 顺序、线路或噪声问题；
- `ΔE_freeze` 大：活性空间近似不足；
- `ΔE_program` 大：几何、基组、SCF 收敛或程序定义不一致。

### 8.2 为什么优先报告投影后能量但保留 raw

在自洽 RHF 解附近，满足粒子数和幂等约束的 rank-6 Slater 投影是当前
ansatz 的物理流形；所以主曲线用 `E[gamma_proj]`。但 raw 能量对噪声
更敏感，必须一同保留以审计投影带来的变化。

---

## 9. Gaussian：怎样运行和比较

### 9.1 直接可比的做法：18 个固定几何单点

每个输入使用：

```text
#p RHF/STO-3G SCF=(Tight,XQC,MaxCycle=512) NoSymm Pop=Full
```

含义：

- `RHF/STO-3G`：与完整 12 轨道 PySCF 参考相同的方法和基组；
- `SCF=Tight`：收紧 SCF 收敛；
- `XQC`：常规 SCF 困难时启用二次收敛；
- `MaxCycle=512`：显式上限；
- `NoSymm`：避免不同构象使用不同点群重排而影响逐点审计；
- `Pop=Full`：输出更完整的轨道/占据分析，能量比较本身只需 `SCF Done`。

单个例子：

```bash
mkdir -p gaussian/logs
g16 < gaussian/inputs/oop_095p641.gjf \
  > gaussian/logs/oop_095p641.log
```

Gaussian 09：

```bash
g09 < gaussian/inputs/oop_095p641.gjf \
  > gaussian/logs/oop_095p641.log
```

全部 18 点：

```bash
mkdir -p gaussian/logs
for input in gaussian/inputs/*.gjf; do
  stem=$(basename "$input" .gjf)
  g16 < "$input" > "gaussian/logs/${stem}.log"
done
```

解析：

```bash
python code/parse_gaussian.py gaussian/logs
```

必须检查两件事：

```text
SCF Done:  E(RHF) = ...
Normal termination of Gaussian
```

解析结果写到：

```text
gaussian/gaussian_energy_comparison.csv
gaussian/gaussian_energy_comparison.json
```

### 9.2 应该比较什么

第一比较量：

\[
\Delta E_{\rm program}
=E_{\rm PySCF}^{12orb}-E_{\rm Gaussian}^{12orb}.
\]

解析 CSV 同时保留 `Gaussian-PySCF` 与 `PySCF-Gaussian` 两列；第 8.1 节
总误差分解使用后一种符号。

第二比较量：

\[
\Delta E_{\rm end}
=E_Q^{10mode}-E_{\rm Gaussian}^{12orb},
\]

并按第 8.1 节拆成测量、冻结核和程序三项。

不能直接拿 Gaussian AO density matrix 与量子 10 模 \(\gamma\) 元素逐项相减，
因为它们的基不同。若要比较密度，必须：

1. 从 Gaussian checkpoint 导出 AO density `P_AO`；
2. 确认 AO 排序和归一化；
3. 用与两次预备 SCF 完全相同的 `C_pre` 变换；
4. 去掉冻结的前两个轨道；
5. 将 spin-summed Gaussian 密度除以 2；
6. 再与量子单自旋 10×10 \(\gamma\) 比较。

AO 基非正交时电子数检查是

\[
N_e=\operatorname{Tr}(P_{\rm AO}S),
\]

不是简单的 `Tr(P_AO)`。

### 9.3 完整 PySCF 参考值：Gaussian 的逐点验收靶标

下列是包内同坐标 `RHF/STO-3G` 完整 12 轨道 PySCF 结果。Gaussian
正常收敛后应逐点接近这些值；最终以实际 `SCF Done` 为准。

| geometry | coordinate/° | `E_full^PySCF`/Ha |
|---|---:|---:|
| `oop_003p157` | 3.157 | -108.545081067939 |
| `oop_026p315` | 26.315 | -108.534068556963 |
| `oop_049p473` | 49.473 | -108.506169704875 |
| `oop_072p631` | 72.631 | -108.464525276552 |
| `oop_095p641` | 95.641 | -108.418133273077 |
| `oop_117p611` | 117.611 | -108.477036651745 |
| `oop_139p581` | 139.581 | -108.522186895869 |
| `oop_161p551` | 161.551 | -108.549545172642 |
| `oop_183p522` | 183.522 | -108.556668154482 |
| `ip_108p736` | 108.736 | -108.544830622809 |
| `ip_127p473` | 127.473 | -108.533340607975 |
| `ip_146p210` | 146.210 | -108.499251462128 |
| `ip_164p947` | 164.947 | -108.463927714106 |
| `ip_182p000` | 182.000 | -108.450990802950 |
| `ip_200p526` | 200.526 | -108.467892264206 |
| `ip_219p052` | 219.052 | -108.506656734972 |
| `ip_237p578` | 237.578 | -108.543127845794 |
| `ip_256p105` | 256.105 | -108.556849527808 |

如果差异超过约 `1e-6 Ha`，先检查：

- 是否使用了同一组 Cartesian 坐标而非重新优化；
- 是否是 `RHF` 而不是 `UHF`、DFT 或 correlated method；
- 是否确为 `STO-3G`；
- 是否是 `0 1`；
- 日志是否正常终止；
- 是否读取了最后一个 `SCF Done`；
- Gaussian 是否使用了不同几何单位。

### 9.4 可选 Gaussian 松弛扫描

`gaussian/optional_relaxed_scan/diazene_out_of_plane_relaxed_scan.gjf`
包含：

```text
D 1 2 3 4 S 12 15.0
```

原子编号是 `H1,N1,N2,H2`。含义是从输入构象开始，对 H1–N1–N2–H2
二面角做 12 次、每次 `+15°` 的 relaxed scan；每个点固定当前二面角并
优化其余自由度。

这个 `12×15°` 网格是透明的本地扩展参数，不来源于论文。若要把新扫描点
送入量子管线，必须对**每个 Gaussian 优化后 Cartesian 几何**依次执行：

```text
提取 Cartesian 坐标
→ 加入新的 geometry JSON
→ build_references.py
→ generate_circuits.py
→ 在平台运行新线路
→ analyze_geometry.py
```

不能只把旧线路的二面角标签改掉。

---

## 10. 安装与一键重跑

### 10.1 基础分析环境

Python 3.10+：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

这足以读取预计算积分、模拟门级数据、重构 1-RDM、算能量、画图和跑测试。

### 10.2 从坐标重新计算积分

要重建 PySCF 参考：

```bash
python -m pip install -r requirements-chemistry.txt
```

### 10.3 快速独立重跑

使用包内参考、线路和数据，重新分析全部 18 点并跑测试：

```bash
python code/run_demo.py --mode quick
```

同时重新做 2000 次 mechanism-gap bootstrap：

```bash
python code/run_demo.py --mode quick --bootstrap 2000
```

### 10.4 完整重建

从 Cartesian 坐标开始重建积分、线路、精确数据、1000-shots 数据和结果：

```bash
python code/run_demo.py --mode full --bootstrap 2000
```

Gaussian 不会被脚本自动调用，因为它是许可证软件；按第 9 节单独运行。

---

## 11. 分步命令

所有命令从项目根目录执行。

### 11.1 重新生成 18 个经典参考

```bash
python code/build_references.py
```

仅一个构象：

```bash
python code/build_references.py --geometry-id oop_095p641
```

输出：

```text
reference/<geometry>.npz   # 代码快速读取
reference/<geometry>.json  # 人可审计的全部矩阵和 SCF 历史
reference/reference_summary.csv
```

### 11.2 重新生成线路

```bash
python code/generate_circuits.py
```

输出每个构象 10 条：

```text
circuits/<geometry>/gate_body_radians/*.txt
circuits/<geometry>/originir_full/*.originir
```

`originir_full` 含 `QINIT/CREG/MEASURE`；若平台编辑器只接受门体，使用
`gate_body_radians`。

### 11.3 精确门级模拟

```bash
python code/simulate_data.py
```

输出 `data/ideal/`。

### 11.4 有限 shots 示例

```bash
python code/simulate_data.py --shots 1000 --seed 5210
```

每个 geometry/setting 的实际 seed 由

```text
SHA256("5210|geometry_id|setting")
```

的前 8 bytes 导出，所以不同线路独立且整包可重复。

### 11.5 分析一个构象

精确数据：

```bash
python code/analyze_geometry.py \
  --geometry-id oop_095p641 \
  --input-dir data/ideal \
  --output-dir results/ideal/oop_095p641
```

平台 counts 并做 2000 次 bootstrap：

```bash
python code/analyze_geometry.py \
  --geometry-id oop_095p641 \
  --input-dir my_platform_data \
  --shots 1000 \
  --bootstrap 2000 \
  --seed 5210 \
  --output-dir results/local_oop_095p641
```

加入 Gaussian 日志：

```bash
python code/analyze_geometry.py \
  --geometry-id oop_095p641 \
  --input-dir my_platform_data \
  --shots 1000 \
  --gaussian-log gaussian/logs/oop_095p641.log \
  --output-dir results/local_oop_095p641
```

### 11.6 分析两条曲线

```bash
python code/analyze_scan.py \
  --input-dir my_platform_data \
  --shots 1000 \
  --gaussian-log-dir gaussian/logs \
  --output-dir results/local_scan
```

### 11.7 transition-region 离散点差 bootstrap

```bash
python code/bootstrap_mechanism_gap.py \
  --input-dir data/ideal \
  --shots 1000 \
  --repetitions 2000 \
  --seed 5210 \
  --output-dir results/ideal_1000_shot_gap_bootstrap
```

### 11.8 自动测试

```bash
python tests/run_tests.py
```

测试覆盖：

1. 18 个几何及面外二面角；
2. 电子数、trace、幂等性和能量闭合；
3. `K10` 的 45 条边恰好覆盖一次；
4. 180 条原生线路的 gate 解析、相邻性和 1-RDM 闭合；
5. 180 个精确 JSON 的完整重构；
6. 1000-shots 结果与误差分解恒等式；
7. Gaussian `SCF Done` 解析和参数来源表。

---

## 12. 包内结果以及应该怎样解释

### 12.1 精确门级闭环

- 18 个构象；
- 10 settings/构象；
- 共 180 条线路；
- 线路到目标测量基 1-RDM 的最大元素误差：`2.3313e-9`；
- 精确 JSON 重构后能量相对 10 模 RHF 最大误差：
  `2.84e-11 mHa`；
- 冻结核偏差范围：`0.000049–0.001127 mHa`。

这说明“积分 → RHF → 编译 → 分布 → 1-RDM → 能量”的代码链在数值精度内
自洽。

### 12.2 固定种子的 1000-shots 示例

- mean TV distance：`0.02280–0.03255`；
- 单自旋 Slater fidelity：`0.98974–0.99688`；
- rank-6 能量相对活性 RHF：`+5.453` 到 `+17.493 mHa`；
- 全部点都大于 `1.593601 mHa`，所以逐点化学精度 **FAIL**；
- 全部点都小于 `40 mHa`，只说明该采样规模与论文讨论的路径差异量级相比
  尚可用，不等于每个绝对能量精确。

投影后误差在这组样本中全为正，符合 RHF 自洽极小值附近的变分直觉；
但这不是对任意带噪 raw density 的无条件定理。

### 12.3 两条路径的离散 transition-region 点

在 Appendix J 印刷的中间点中：

```text
out-of-plane: oop_095p641
in-plane:     ip_182p000
```

本包定义差值方向为

\[
\Delta_{\rm path}
=E_{\rm in-plane}-E_{\rm out-of-plane}.
\]

完整 PySCF RHF 给出

\[
\Delta_{\rm path}^{9pt}
=-32.857529873\ {\rm mHa}.
\]

负号表示在这两个离散印刷点上，面内点比面外点低约 `32.86 mHa`。
对理想分布按每 setting 1000 shots 做 2000 次 bootstrap：

\[
{\rm median}=-32.2948\ {\rm mHa},
\]

\[
95\%\ {\rm interval}
=[-42.3935,-22.8635]\ {\rm mHa},
\]

同号比例为 `1.0`。

这不是论文引用的完整连续曲线约 `40.2 mHa` 数值。两者不同的原因是：

1. 本包只使用 Appendix J 可逐坐标审计的 9 点；
2. 真正的路径极值可能位于未印刷的 20 点网格中；
3. 坐标只保留到论文表格的 5 位小数；
4. 本包用 PySCF 重建积分，并明确采用 MINAO 初猜；
5. 本包的 10-setting 编译协议与论文 11-setting 硬件协议不同。

要声明论文 `40.2 mHa` 的数值复现，必须取得完整 20 点 Cartesian/积分文件，
再在相同能量定义下寻找两条曲线的极值；不能对 9 点数据插值后把结果当作
原始论文值。

---

## 13. 常见故障及定位顺序

### 13.1 所有 setting 的粒子数都不对

检查：

1. 是否漏了 `X q[0]...X q[5]`；
2. 平台是否把 qubit 编号反向；
3. `SQISWAP` 是否被替换成不守粒子数的近似门；
4. JSON 是否混入别的任务结果。

### 13.2 粒子数正确，但 1-RDM 像被翻转

最常见是 bit 顺序。默认是 `q9...q0`；若平台返回 `q0...q9`，显式传入正确
顺序或先转换 JSON，不要仅靠粒子数判断。

### 13.3 对角元正确，非对角元符号相反

检查：

- `SQISWAP` 的 `+i/-i` 分支；
- `RZ` 定义和角度单位；
- `physical_left/right` 是否互换；
- 是否错误使用
  `(<n_right>-<n_left>)/2`。

先单独运行一个两模式 Givens 校准，不要直接在 10 qubits 上猜符号。

### 13.4 角度看起来约大 57.3 倍

平台把弧度当成度。所有 `RZ` 文件都是显式弧度；例如 `pi` 必须输入
`3.141592653589793`，不是 `180`。

### 13.5 raw 自然占据数超出 [0,1]

有限 shots 下可能发生。处理顺序：

1. 报告 raw 本征值和幂等误差；
2. 与同 shots 理想 bootstrap 比；
3. 再报告 rank-6 projected；
4. 不删除 raw 结果。

### 13.6 后选择保留率很低

低 `P_pass` 表示明显粒子数泄漏或读出错误。投影不会恢复被丢弃样本的信息。
应报告原始 shots、保留 shots、每个 setting 的 `P_pass`，并将硬件结果与
同 shots 理想统计区间区分。

### 13.7 Gaussian 与 PySCF 不一致

依次检查同坐标、Å 单位、`0 1`、`RHF`、`STO-3G`、正常终止、最后一个
`SCF Done`。不要把 optional relaxed scan 的优化几何拿来与固定论文几何
比较。

### 13.8 SCF 在某构象不收敛

先保留失败日志，不要静默使用最后一次迭代。Gaussian 可检查 `XQC` 是否
进入；PySCF 可比较 `active_scf_history` 的密度差、能量差和对易子。
若改初猜、阻尼或 level shift，必须把它作为新参数记录，因为冻结轨道和
量子角度可能随之改变。

### 13.9 相邻构象的某些 Givens 角跳变接近 π

先比较 \(\gamma\)、能量和 Slater 子空间，再判断是否物理不连续。
MO 列符号和占据子空间旋转是 gauge 自由度，裸编译角不要求逐点连续。

---

## 14. 方法依据与来源

1. F. Arute et al., “Hartree-Fock on a superconducting qubit quantum
   computer,” *Science* 369, 1084–1089 (2020):  
   https://www.science.org/doi/10.1126/science.abb9811
2. 论文预印本及 Supplementary Information（几何表、10-qubit
   二氮烯流程、测量与硬件细节）：  
   https://arxiv.org/abs/2004.04174
3. Google Quantum AI HFVQE 教程（1-RDM 驱动的 HF 工作流）：  
   https://quantumai.google/cirq/experiments/hfvqe
4. Google Quantum AI molecular-data 教程（AO 积分与 RHF 目标）：  
   https://quantumai.google/cirq/experiments/hfvqe/molecular_data
5. OpenFermion 二氮烯 cloud manifest（STO-3G、两次 SCF、冻结最低
   两轨道、完整路径标签）：  
   https://github.com/quantumlib/OpenFermion/blob/main/cloud_library/diazene.txt
6. OpenFermion Slater determinant preparation API（相邻 Givens
   制备约定）：  
   https://quantumai.google/reference/python/openfermion/circuits/slater_determinant_preparation_circuit
7. PySCF SCF API 与 RHF 实现：  
   https://pyscf.org/pyscf_api_docs/pyscf.scf.html  
   https://github.com/pyscf/pyscf/blob/master/pyscf/scf/hf.py
8. Gaussian `Opt` 与 ModRedundant 文档：  
   https://gaussian.com/opt/  
   https://gaussian.com/modred/

---

## 15. 最终验收清单

在把新平台数据称为“二氮烯 HF 复现成功”前，逐项确认：

- [ ] 使用 `data/geometries_paper.json` 的同一 Cartesian 几何；
- [ ] `RHF/STO-3G`, `0 1`；
- [ ] 每个构象使用自己的 10 条线路，未复用其他构象角度；
- [ ] 所有 RZ 参数按弧度解释；
- [ ] 平台 `SQISWAP` 矩阵符号已经校准；
- [ ] bit 顺序确认为 `q9...q0` 或已显式转换；
- [ ] 10 个 JSON 全部完成且 shots 已知；
- [ ] 每个 setting 报告 `P(N=6)`；
- [ ] raw 与 rank-6 1-RDM 都保存；
- [ ] `Tr(gamma)`、幂等误差、自然占据数、fidelity 和 TV 距离均报告；
- [ ] 能量使用与该构象相同的 `h_eff/ERI/E0`；
- [ ] `E_Q-E_Gaussian` 按测量、冻结核、程序差异拆开；
- [ ] Gaussian 日志包含最后的 `SCF Done` 和正常终止；
- [ ] 有限 shots 结论来自 bootstrap，而不是只看一个随机点；
- [ ] 明确区分 9 点离散结果与论文完整 20 点/40.2 mHa 结果。

通过这些检查后，合理结论是：

> 在明确的二氮烯几何、RHF/STO-3G、两轨道冻结和门约定下，量子线路重构
> 的 10 模单自旋 1-RDM 与活性空间 RHF 参考一致；由该 1-RDM 得到的 HF
> 能量在给定 shots 的统计区间内，并且与完整 PySCF/Gaussian RHF 的差异
> 已被分解为测量、冻结核和经典程序三部分。
