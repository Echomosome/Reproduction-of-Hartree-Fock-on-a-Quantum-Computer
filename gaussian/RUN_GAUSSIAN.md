# Gaussian 运行说明

本目录的 `inputs/*.gjf` 是论文 Appendix J 中 18 个二氮烯几何的
**同几何单点** `RHF/STO-3G` 计算。它们是量子能量最直接、定义最清楚的
Gaussian 对照。

## 1. 单个任务

```bash
g16 < gaussian/inputs/oop_095p641.gjf \
  > gaussian/logs/oop_095p641.log
```

Gaussian 09 可把 `g16` 替换为 `g09`。成功日志必须同时包含：

```text
SCF Done:  E(RHF) = ...
Normal termination of Gaussian
```

## 2. 全部 18 个任务

先创建日志目录，再逐个提交：

```bash
mkdir -p gaussian/logs
for input in gaussian/inputs/*.gjf; do
  stem=$(basename "$input" .gjf)
  g16 < "$input" > "gaussian/logs/${stem}.log"
done
```

集群上应把每个输入拆成独立调度任务，不要在登录节点直接串行运行。

## 3. 解析与 PySCF 对照

```bash
python code/parse_gaussian.py gaussian/logs
```

输出：

```text
gaussian/gaussian_energy_comparison.csv
gaussian/gaussian_energy_comparison.json
```

主比较量是

\[
\Delta E_{\rm program}
= E_{\rm PySCF}^{12orb}
- E_{\rm Gaussian}^{12orb}.
\]

CSV 同时输出 `gaussian_minus_pyscf_microhartree` 和符号相反的
`pyscf_minus_gaussian_microhartree`；总误差分解采用后一种符号。

Gaussian 与 PySCF 都必须使用相同的 18 组 Cartesian 坐标、`RHF`、
`STO-3G`、`0 1` 和无对称性重排，才可以逐点比较。

## 4. 可选 relaxed scan

`optional_relaxed_scan/diazene_out_of_plane_relaxed_scan.gjf` 使用
Gaussian 官方 `Opt=ModRedundant` 语法：

```text
D 1 2 3 4 S 12 15.0
```

含义是从输入构象开始，把 `H1-N1-N2-H2` 二面角每次增加 15°，共 12
步；每一步固定该二面角并优化其余自由度。

这条曲线是**独立 Gaussian 松弛扫描**，不是论文表格的直接复现：

- 网格是 13 点而不是论文的 9 点；
- 优化程序从 Psi4 换成 Gaussian；
- 起点坐标只保留到论文印刷的 5 位小数。

因此它用于检查趋势和生成新的路径，不能替代 18 个同几何单点对照。
