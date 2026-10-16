# T08：PySCF 独立交叉验证

日期：2026-10-14（Asia/Tokyo）。对应 TODO T08、实施计划附录 C、spec §13.4。

## 设置

- 几何：Task 3 fixture。H2O：O(0,0,0)、H(0.96,0,0)、H(−0.24,0.93,0) Å；点电荷 −0.834 e @ (3,0,0) Å，+0.417 e @ (3.5,0.8,0) Å。
- 方法：RHF/def2-SVP（球谐），电荷 0，单重态，点电荷为点模型。
- PySCF：2.14.0，独立环境 `/home/korakuen/micromamba/envs/pyscf-env`（kasuga02），没有装进 `openmm_dev`。`qmmm.mm_charge`，`conv_tol=1e-12`，`conv_tol_grad=1e-8`；点电荷梯度 = `grad_hcore_mm(dm) + grad_nuc_mm()`。
- ORCA：6.1.1（kasuga02 本地安装），经 `ORCAOPIBackend`（openmmorca 0.4.0）运行，scratch 放在 `/dev/shm`。两组设置：
  - `orca_default`：`! HF def2-SVP EnGrad`（与 fixture 相同；TolE 1e-8，TolG 1e-5）
  - `orca_verytight_norijcosx`：再加 `VeryTightSCF NoRIJCOSX`（TolE 1e-9，TolG 2e-6）
  - 两组的 `qm.out` 中，辅助 J 基和 J/K 基都是 "NOT available"，也就是说默认 HF 没有用 RI 或 COSX，两边都是精确积分。
- 单位与符号：两边都是梯度 +∂E/∂R，单位 Eh/bohr。ORCA 一侧由后端返回的力（kJ/mol/nm）乘以 −1/49614.752589 换回梯度。

## 结果

| ORCA | E (Eh) | ΔE（ORCA − PySCF，Eh） | max\|Δg_QM\|（Eh/bohr） | max\|Δg_pc\|（Eh/bohr） |
|---|---|---|---|---|
| fixture（Task 3） | −75.977160671564 | +5.3e-10 | 3.2e-7 | 6.8e-8 |
| orca_default（重算） | −75.977160671564 | +5.3e-10 | 3.2e-7 | 6.8e-8 |
| orca_verytight_norijcosx | −75.977160671594 | +5.0e-10 | 5.7e-8 | 3.3e-9 |

PySCF：E = −75.977160672096 Eh。

附录 C 给出的预期量级是：能量约 1e-6 Eh，梯度约 1e-5 Eh/bohr。实测的能量差小 3 个数量级，QM 梯度差小 1.5–2 个数量级，点电荷梯度差小 2–3 个数量级。逐分量的数值见 `comparison.json`。

- 当前代码在 kasuga02 上重算出的 fixture 能量与原值 12 位一致。
- 默认设置下 QM 梯度差 3.2e-7，改成 VeryTightSCF 后降到 5.7e-8。这说明默认设置下的残差主要来自 ORCA 的 SCF 收敛阈值（TolG 1e-5），不是积分或符号的差异。
- 剩下约 5e-10 Eh 的能量差两组设置都一样，不随 SCF 阈值变化，应该来自两个程序在积分或基组细节上的系统差别。这一点没有进一步排查，它比预期量级小 3 个数量级。

结论范围：H2O + 2 个点电荷的 HF/def2-SVP 单点，验证了 ORCA 后端的能量、QM 梯度和点电荷梯度（包括单位和符号）与 PySCF 一致。这不涉及 DFT 网格、RIJCOSX、截断嵌入或 link atom。

## 产物与命令

`experiments/2026-10-14_t08_pyscf_crosscheck/`：`run.sh`（入口）、`pyscf_side.py`、`orca_side.py`、`compare.py`、`geometry.py`；输出 `pyscf.json`、`orca_*.json`、`orca_*.qm.inp`、`orca_*.qm.out`、`comparison.json`、`comparison.md`、`run.log`。

```bash
ssh korakuen@192.168.0.183 'bash -c "cd /home/ruigengji/openmm-ORCA/experiments/2026-10-14_t08_pyscf_crosscheck && ./run.sh"'
```

整个流程串行运行，计算部分只要几秒。墙钟时间主要花在从 NFS 加载 Python 环境上。
