# Gaussian 运行说明

默认输入：

```text
inputs/h4_R1p3000_RHF_STO3G.gjf
```

在安装并授权 Gaussian 的机器上运行：

```bash
g16 < inputs/h4_R1p3000_RHF_STO3G.gjf > h4_R1p3000_RHF_STO3G.log
```

如果命令名是 `g09`，将 `g16` 替换为 `g09`。正常结束后应能搜索到：

```text
SCF Done:  E(RHF) = ...
Normal termination of Gaussian
```

从材料包根目录解析：

```bash
python code/parse_gaussian.py \
  gaussian/h4_R1p3000_RHF_STO3G.log
```

把 Gaussian 能量直接并入量子数据分析：

```bash
python code/analyze_results.py \
  --input-dir data/your_run \
  --gaussian-log gaussian/h4_R1p3000_RHF_STO3G.log \
  --shots 1000 \
  --bootstrap 2000 \
  --output-dir results/your_run
```

`scan_inputs/` 中包含 \(R=0.5,0.9,1.3,1.7,2.1,2.5\) Å 六个独立输入。
每个键长都必须重新生成 RHF 参考和量子线路；不能把 \(R=1.3\) Å 的 Givens
角用于其他几何。
