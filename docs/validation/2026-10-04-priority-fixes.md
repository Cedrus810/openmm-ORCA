# P1 修复验证记录

日期：2026-10-04（Asia/Tokyo）。基线提交 `d1ff991`，本次修复位于未提交工作区；源码版本仍为 0.4.0，变更列入 Unreleased。

## 修复范围

- T01：OutOfPlaneSite 按坐标 Jacobian 回分；支持位点之间的依赖顺序。与原生 OpenMM 的物理原子力及全坐标有限差分对照，涵盖平均位点、面外位点与嵌套位点。
- T02：正步数校验；单步可运行；短跑标记 smoke / Task 22 NOT_EVALUATED；完整验收要求至少 2000 步、末尾 500 个温度样本与完整步号。显式反应键独立监控；所配置的稳定键检查通过不改写原全键门槛。
- T03：准备／平衡缓存核对阶段身份及产物哈希，记录参数与软件版本。平衡身份包含有效 MM System 的序列化哈希、准备拓扑与盒；读取 State 检查坐标／速度粒子数与有限性，核对原子映射与盒信息。无 manifest、失配或损坏的缓存保留并拒绝复用。
- T04 部分：CSV 逐步 flush，异常时关闭后端；acceptance.json 记录运行／失败／验收状态，新的失败运行不会沿用旧 PASS。checkpoint／MD 续跑仍未实现。

## 执行结果

环境：`/home/ruigengji/miniforge3/envs/openmm_dev/bin/python`；Python 3.12.13，OpenMM 8.5.2，orca-pi 2.0.0，NumPy 2.4.3，pytest 9.1.1。ORCA 路径 `/home/ruigengji/ORCA611`。

```bash
OPI_ORCA=/home/ruigengji/ORCA611 PYTHONDONTWRITEBYTECODE=1 \
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python -m pytest \
  -m 'not slow' -q -rs -p no:cacheprovider \
  --basetemp=/tmp/openmm-orca-fix-regression
```

终端结果：`196 passed, 5 skipped, 2 deselected in 124.68s (0:02:04)`。

- 通过数包括 166 个非 ORCA 用例与 30 个真实 ORCA 快速用例。
- 32 个新增用例：4 个虚拟位点原生／有限差分对照，28 个酶工作流用例。
- 酶工作流用例覆盖短跑／完整验收、截断或重复步号、非有限温度、反应键、缓存校验、粒子数失配、单步运行与注入失败。还在小水体系上执行了真实 MM 的冷缓存写入及复用（2 步 NVT + 2 步 NPT）；不是完整酶平衡验证。
- 5 个 MPI 用例均因通信探针失败跳过：`pmix_ifinit: socket() failed with errno=1`，并报告无可用通信接口。尚未验证修复版本的多进程 ORCA。
- 2 个 slow 用例（NVE、restart 轨迹比较）未运行；未运行 CUDA 或完整 DhlA 1 ps 应用。
- CLI `--help` 已检查。当前安装元数据仍为 0.2.0，实际源码为 0.4.0；T05 尚待刷新安装与打包核对。

额外 Reference 探针确认：OpenMM 8.5.2 会将原始 PythonForce 位点力分配至父原子，同时在返回数组中保留原位点力。已修正文档中“完全不分配”的旧描述。当前回调显式分配并清零位点项，避免第二次分配。

## 源码与测试哈希（SHA-256）

```text
994c64899146791428e51fe073cec6058716f6f777ac9e755c1d512d9349f9bf  openmmorca/force.py
80fcf032c62dd9e799ebb2a71c727b1439f951b7a6509d5d2a9c087455c37e9e  examples/enzyme_qmmm.py
cf192fe7f8596909c58d0a561725405d114031a35ae48c434d95a60609fd5743  examples/enzyme_workflow.py
7992f4d1c233b249d61a7b80802b4045364e758cb803021072a4542cdc4d7a09  examples/__init__.py
e3d7da2b02ccee4268e7adf2dc64453d25c5c394e5c1dcc151bae2c00dd6a533  tests/test_force_fake.py
bc19ff67be6138b666b3ee606cb8b1bdf04c8397c4f1ed87b05017b62b372123  tests/test_enzyme_workflow.py
```

本文件是本次检查的持久摘要；完整终端输出位于本聊天。历史酶轨迹、长测试原始数据与科学准确性验证不在本次证据范围内。
