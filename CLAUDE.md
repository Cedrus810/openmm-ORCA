# openmm-ORCA 项目说明

## 设计稿（开工、验收前全文读完）

- `openmm_orca_opi_design_plan.md`：设计规格（spec）
- `docs/plans/2026-09-26-openmm-orca-implementation-plan.md`：实施计划（含附录 C）
- 后续任务清单：`TODO.md`；验证记录：`docs/validation/`
- `docs/plans/2026-10-15-project-roadmap.md`：当前路线（2026-10-15 起按此推进，阶段 A–D）
- `docs/validation/2026-11-06-current-status.md`：C2 扫描产物核对，16 窗口、320 帧
- `docs/validation/2026-11-07-c2-matched-pilot.md`：最新首批结果；3 构型匹配单点完成，相对嵌入偏差 −10.26/−16.09 kJ/mol，暂停扩大并复核嵌入路线
- `docs/validation/2026-11-07-c2-error-mechanism.md`：首批误差来源；经典势差一阶复现 −10.26/−16.09（≤ 0.15 kJ/mol），主因是 S2 缺少的近似均匀场 × 反应偶极变化；320 帧经典统计：场差各窗几乎不变，窗口平均消不掉
- `docs/plans/2026-11-07-s3-ewald-correction-proposal.md`：S3（S2 + 经典 Ewald 修正势）范围变更提案；范围变更用户 2026-11-07 已同意（设计稿 §1.2、C1 §8 已改），阶数与执行步骤未定
- `docs/validation/2026-11-07-s3-v1-pyscf.md`：S3 V1 PySCF 原型，首批 3 帧相对误差 2 阶 ≤ 0.12、3 阶 ≤ 0.005 kJ/mol；赝电荷需正则化
- `docs/validation/2026-11-07-s3-v2-orca.md`：S3 V2 ORCA 单点，赝电荷可直接用，S3 相对修正量与 PySCF 差 ≤ 0.26 kJ/mol；下一步 V3 需改源码
- `docs/plans/2026-11-07-s3-v3-implementation-plan.md`：S3 可选模式实施计划（Task 0–6），已批准；Task 0–6 全部完成（2026-11-09；酶单帧有限差分最大差 0.034 kJ/mol/nm），见计划末尾"进度"；V4、V5 已完成
- `docs/validation/2026-11-09-s3-v4-b5-forces.md`：S3 V4，B5 20 帧力偏移 S2 +36.7/−53.0 → S3 −0.48/+0.13 kJ/mol/nm；PySCF r2SCAN 解析梯度与能量差分有共同偏差（伸缩约 −13），只用同法相减
- `docs/validation/2026-11-09-s3-v5-c1-windows.md`：S3 V5，C1 窗口统计（PySCF r2SCAN/def2-SVP，5 窗 × 20 帧）：S3 ΔΔE_emb ≤ 0.10 kJ/mol 满足 δE = 5；S2 在 w06/w07/w15 为 −12.3/−8.5/−14.7，不满足
- `docs/validation/2026-11-09-s3-orca-bridge-windows.md`：同 100 帧 ORCA r2SCAN-3c 与 PySCF 的 S3 修正量 ΔΔC 差 +0.07/+0.32/+0.18/+0.11 kJ/mol；ORCA 下嵌入误差为推断
- `docs/validation/2026-11-07-s3-v3-task0.md`：V3 Task 0 通过（解析 Taylor 相对误差 ≤ 0.07 kJ/mol；orca_vpot 用 `qm.scfp`，与有限差分一致到 1e-6；经典每步约 0.8 s）

（前两项路径取自 `researchd.plan.toml` 的 `design` 字段。）

## 环境

| 机器 | 环境 | 是否生产 | 说明 |
|---|---|---|---|
| NFS | `/home/ruigengji/miniforge3/envs/openmm_dev` | 未确认 | 开发环境，openmmorca 可编辑安装；NFS 拥堵时 import 要几分钟 |
| 183 | `/home/korakuen/envs/openmm_dev` | **不是**（用户 2026-10-14 确认） | 本地盘，import 快；有 OpenMM 8.5.2、openff、pdbfixer、mdtraj，**没有 orca-pi**，也没装 openmmorca（可用 `PYTHONPATH=/home/ruigengji/openmm-ORCA`） |
| 183 | `/home/korakuen/micromamba/envs/pyscf-env` | 未确认 | PySCF 2.14.0，T08 交叉验证用 |

PBS 计算节点上用 `module load orca/6.1.1`（用户 2026-10-28 原话：“计算节点上 有 module load orca/6.1.1 你再我们的 pbs里面这么写”；会自动加载 gcc/13.3.0、openmpi/5.0.7_gcc13.3.0）。

ORCA 6.1.1 本地安装：181 `/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2`，183 `/home/korakuen/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2`。不要用 NFS 上的 `/home/ruigengji/ORCA611` 跑计算。ORCA scratch 放 `/dev/shm`（`TMPDIR=/dev/shm`）。

## 运行产物

- 2026-10-05 及以前：`/home/kasuga/openmm-orca-runs/`（181 本地盘）
- 2026-10-14 起：`experiments/<日期>_<主题>/`
