# openmm-ORCA 项目说明

## 设计稿（开工、验收前全文读完）

- `openmm_orca_opi_design_plan.md`：设计规格（spec）
- `docs/plans/2026-09-26-openmm-orca-implementation-plan.md`：实施计划（含附录 C）
- 后续任务清单：`TODO.md`；验证记录：`docs/validation/`
- `docs/plans/2026-10-15-project-roadmap.md`：当前路线（2026-10-15 起按此推进，阶段 A–D）

（前两项路径取自 `researchd.plan.toml` 的 `design` 字段。）

## 环境

| 机器 | 环境 | 是否生产 | 说明 |
|---|---|---|---|
| NFS | `/home/ruigengji/miniforge3/envs/openmm_dev` | 未确认 | 开发环境，openmmorca 可编辑安装；NFS 拥堵时 import 要几分钟 |
| 183 | `/home/korakuen/envs/openmm_dev` | **不是**（用户 2026-10-14 确认） | 本地盘，import 快；有 OpenMM 8.5.2、openff、pdbfixer、mdtraj，**没有 orca-pi**，也没装 openmmorca（可用 `PYTHONPATH=/home/ruigengji/openmm-ORCA`） |
| 183 | `/home/korakuen/micromamba/envs/pyscf-env` | 未确认 | PySCF 2.14.0，T08 交叉验证用 |

ORCA 6.1.1 本地安装：181 `/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2`，183 `/home/korakuen/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2`。不要用 NFS 上的 `/home/ruigengji/ORCA611` 跑计算。ORCA scratch 放 `/dev/shm`（`TMPDIR=/dev/shm`）。

## 运行产物

- 2026-10-05 及以前：`/home/kasuga/openmm-orca-runs/`（181 本地盘）
- 2026-10-14 起：`experiments/<日期>_<主题>/`
