# C2 首批嵌入误差的一阶分解

承接 [C2 首批匹配单点](../../docs/validation/2026-11-07-c2-matched-pilot.md)。结果与解释见 [验证记录](../../docs/validation/2026-11-07-c2-error-mechanism.md)。

`decompose.py` 读取首批的 `inputs/requests_A.npz`（S2 请求）和 `pyscf/` 下的 S2/ENV20/ENV30 结果，对一个候选帧：

1. 经典计算 `dV = V_Ewald − V_S2`：S2 请求中的同一批点电荷，直接库仑求和与立方盒 Ewald（均匀背景，无 QM 自身镜像）之差；在 QM 区内 700 个点上精确计算，用 500 个点拟合 1–4 阶多项式，另 200 个点检验。Ewald 用 (α, kmax) = (1.2, 14) 与 (1.6, 20) 自检。
2. 以首批相同设置重算 PySCF r2SCAN/def2-SVP 的 S2 SCF，核对能量，取电子密度。
3. 一阶能量 `E1 = Σ Z_a dV(R_a) − ∫ n dV`，积分取离 QM 质心 0.6/0.8/1.0 nm 内的 DFT 格点；按质心处多极分出单极、偶极、四极、三阶贡献。

```bash
PSCF=/home/ruigengji/miniforge3/envs/pyscf-env/bin/python
D=experiments/2026-11-07_c2_error_decomp
P=experiments/2026-11-06_c2_matched_pilot
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYSCF_TMPDIR=/dev/shm
for f in 0 4 8; do $PSCF -u $D/decompose.py $P/inputs/requests_A.npz $P/pyscf $D/out $f --threads 8 > $D/frame_$f.log 2>&1 & done; wait
```

2026-11-07 在 181（kasuga01）本机运行，三帧并行、每帧 8 线程；每帧约 20 s（经典 5 s，SCF+密度 13–14 s）。输出 `out/candidate_{00,04,08}.json` 与 `frame_*.log`。

## 320 帧经典场差（无 QM）

`field_stats.py` 从 C2 扫描的每帧捕获 S2 请求（零力捕获后端，同 `prepare_pilot.py`），用 `classical.py` 拟合 `dV`；`analyze_fields.py` 汇总为 `field_stats.json`。

```bash
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
PSCF=/home/ruigengji/miniforge3/envs/pyscf-env/bin/python
D=experiments/2026-11-07_c2_error_decomp
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OPENMM_CPU_THREADS=1
for i in 0 1 2 3 4 5 6 7; do $PY -u $D/field_stats.py $D/fields $i $((i+8)) > $D/fields_$i.log 2>&1 & done; wait
$PSCF $D/analyze_fields.py
```

2026-11-07 在 181 本机运行，8 进程 × 2 窗，约 90 s。输出 `fields/w00–w15.npz`、`fields_*.log`、`field_stats.json`。
