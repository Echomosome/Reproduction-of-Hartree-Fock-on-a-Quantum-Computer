# Gaussian：H₂ RHF/STO‑3G 的运行与对比

主输入文件：

```text
inputs/h2_R0p7414_RHF_STO3G.gjf
```

## Linux / 集群

在本材料包根目录运行：

```bash
cd gaussian/inputs
g16 < h2_R0p7414_RHF_STO3G.gjf > h2_R0p7414_RHF_STO3G.log
formchk h2_R0p7414_RHF_STO3G.chk h2_R0p7414_RHF_STO3G.fchk
cd ../..
```

某些集群使用作业队列或包装脚本；这时只需把同一个 `.gjf` 交给本地规定的 Gaussian 16 提交命令。Gaussian 09 则通常把 `g16` 改为 `g09`。

## Gaussian 16W / GaussView

1. 打开 `h2_R0p7414_RHF_STO3G.gjf`；
2. 检查方法是 Restricted Hartree–Fock、基组是 STO‑3G、charge/multiplicity 是 `0 1`；
3. 提交 single-point energy；
4. 在输出中搜索 `SCF Done`。

应得到接近：

```text
SCF Done:  E(RHF) =  -1.116684387...
```

不同程序采用的 Å–Bohr 常数与输出精度可能造成末几位差别；本项目把 Gaussian 与 PySCF 的经典交叉检查阈值设为 \(10^{-6}\) Ha。

## 自动解析

```bash
python code/parse_gaussian.py \
  gaussian/inputs/h2_R0p7414_RHF_STO3G.log \
  --reference reference/h2_r0.7414_sto3g_reference.npz \
  --output gaussian/gaussian_energy_comparison.csv
```

或者把 Gaussian 能量直接加入量子结果分析：

```bash
python code/analyze_results.py \
  --input-dir data/example_ideal_counts \
  --reference reference/h2_r0.7414_sto3g_reference.npz \
  --gaussian-log gaussian/inputs/h2_R0p7414_RHF_STO3G.log \
  --output-dir results_with_gaussian
```

## 键长扫描

`scan_inputs/` 已包含 \(R=0.30\)–\(3.00\) Å、步长 \(0.10\) Å 的 28 个输入文件。也可自行重新生成：

```bash
python code/generate_gaussian_inputs.py \
  --lengths 0.30:3.00:0.10 \
  --output-dir gaussian/scan_inputs
```

批量运行的具体命令取决于所在集群的调度系统。每个 `.log` 都可由 `parse_gaussian.py` 读取。

## 为什么主要比较 `SCF Done`

量子侧使用 PySCF 生成同几何、同 STO‑3G 基组的积分，再把测得的 1‑RDM 代入 RHF 能量泛函。因此最稳健的 Gaussian 比较量是总能量：

\[
\Delta E_Q=E_Q-E_{\rm Gaussian}^{\rm RHF}.
\]

Gaussian 的 AO 密度矩阵是非正交基表示，不能直接与量子线路的正交轨道 1‑RDM 逐元素比较。若要做进阶比较，先从 Gaussian 的 RHF spin-summed AO density \(P_{\rm AO}\) 出发，使用本项目保存的 \(X=S^{-1/2}\)：

\[
D_{\rm orth}^{\rm one\ spin}
=X^{-1}\left(\frac{P_{\rm AO}}2\right)X^{-T}.
\]

然后再与 `gamma_raw.csv` 或 `gamma_projected_rank1.csv` 比较。对 H₂/STO‑3G，AO 排序需保持为输入中的第一个 H、第二个 H。
