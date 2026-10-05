# openmm-orca

[English](README.md) | [简体中文](README.zh-CN.md) | 日本語

OpenMM 主導の QM/MM：**OpenMM が MD を実行**し（力場・積分器・サーモスタット/バロスタット・トラジェクトリ）、**ORCA が QM 領域を計算**します（電子構造 + 静電埋め込み勾配）。OPI（ORCA Python Interface）が ORCA の入出力を担います。インターフェースのスタイルは `openmm-ml` / `openmm-pyscf` に準拠しています。

バージョンごとの変更点：[CHANGELOG.md](CHANGELOG.md)。設計仕様：`openmm_orca_opi_design_plan.md`；実装計画：`docs/plans/2026-09-26-openmm-orca-implementation-plan.md`；ONIOM 計画：`docs/plans/2026-09-28-oniom.md`（いずれも中国語）。

その後の修正・検証・機能拡張は [TODO リスト](TODO.md)（中国語。優先度・証拠・受け入れ条件つき）を参照してください。

**現在の状態（v0.4.0）：** full-QM、QM/MM（静電埋め込み、H リンク原子による共有結合境界、周期 MM の近似カットオフ埋め込み）、2 層 ONIOM（QM:QM、非周期）に加え、restart と失敗バンドル診断が動作しています（M0–M5 + ONIOM）。溶媒和酵素系（DhlA、31,610 原子、`examples/enzyme_qmmm.py`）で検証済みです。

## インストール

必要環境：Python ≥ 3.10、[OpenMM](https://openmm.org/) ≥ 8.5、[orca-pi](https://pypi.org/project/orca-pi/) ≥ 2.0、numpy、および ORCA ≥ 6.1 のインストール（pip では導入できません。ライセンスが必要です）。

```bash
python -m pip install -e .        # リポジトリのルートで実行

# ORCA の場所（必須。orca-pi は import 時に必要とします）
export OPI_ORCA=/path/to/orca     # orca 実行ファイルのあるディレクトリ

# nprocs > 1 の並列には MPI が必要です。システムの mpirun がないノード
# （ログインノードなど）では、OPI_MPI を OpenMPI のインストール（mpirun の
# あるディレクトリ）に向けてください：
export OPI_MPI=/path/to/openmpi

# バッチスケジューラ下では、並列 ORCA の実行にジョブ割り当てが最低でも
# nprocs 分のスロットを提供している必要があります（OpenMPI/PRRTE は
# スケジューラの割り当てを読みます。「Not enough slots available」は
# 要求コア数が nprocs に満たないことを意味します）。
# 意図的にオーバーサブスクライブする場合：OMPI_MCA_rmaps_default_mapping_policy=:oversubscribe
```

ORCA が NFS 上にある場合、混雑するとステップあたりの起動オーバーヘッドが数倍に増大します。MD を実行する前に、ORCA をローカルディスクまたは tmpfs にコピーし（`autoci_*` は不要なので省略可）、`OPI_ORCA` をそのコピーに向けてください：

```bash
D=/dev/shm/orca-$USER; mkdir -p $D
cd /path/to/orca && cp -a lib datasets $D/ && ls | grep -v -e '^autoci_' -e '^lib$' -e '^datasets$' | xargs -I{} cp -a {} $D/
export OPI_ORCA=$D
```

共用クラスタではローカルの ORCA インストールがすでに存在する場合があります——その場合はコピーせず `OPI_ORCA` をそこに向けてください。実行結果は永続的なローカルディスクに置き、`/tmp` には置かないでください。

OpenMPI の `mpirun` は呼び出し元の `taskset` affinity を無視し、常にコア 0 からプロセスを結合します。並列 ORCA を特定のコアに固定したい場合（他ジョブとマシンを共有する場合など）は、`PRTE_MCA_hwloc_default_cpu_list` に希望するコアリストを設定してください。

## 最小例

full-QM（水分子 1 個、20 MD ステップ。`examples/full_qm_water.py` 参照）：

```python
from openmmorca import ORCAPotential

potential = ORCAPotential(method="HF", basis="def2-SVP", extra_keywords=("TightSCF",))
system = potential.createSystem(topology)
```

QM/MM（QM = 水 0、残りはすべて MM、静電埋め込み）：

```python
potential = ORCAPotential(method="HF", basis="def2-SVP", extra_keywords=("TightSCF",))
mixed = potential.createMixedSystem(topology, system, atoms=[0, 1, 2], forceGroup=0)
```

H リンク原子による共有結合 QM/MM 境界（v0.3。リンク原子を通る有限差分が ORCA の解析力と < 0.05 kJ/mol/nm で一致。`examples/link_atom_dipeptide.py` 参照）。切断した各結合を `(q1, m1)` として宣言します（q1 は QM 領域、m1 は MM 領域）。各境界には `R_q1 + g (R_m1 − R_q1)` の位置に水素キャップが置かれ、その力は連鎖律によって q1/m1 に配分されます：

```python
mixed = potential.createMixedSystem(
    topology, system, atoms=qm_atoms,
    boundaryPairs=[(cb, ca)],      # 例：アミノ酸側鎖を CB–CA で切断
    linkRatios=None,               # C–C（1.09/1.526）と C–N（1.09/1.449）にはデフォルトの g あり
)
```

このとき `charge`/`multiplicity` は「QM 原子 + リンク H」を記述します。単結合のみ、q1 あたりリンクは 1 つ。m1 の電荷は埋め込みから取り除かれ、その MM 隣接原子に均等に配分されます（*簡略版* charge shift——文献のスキームにある双極子補償の点電荷対は含みません）。OpenMM の MM–MM 静電相互作用は変更されません。力場残基から切り出した QM 領域では、切断された各残基の QM 部分の力場電荷 x は通常整数ではありません。残余 x − round(x) はその残基の M2 原子に加算され、切断された各残基の MM 残部——および埋め込み全体——が整数電荷を持ちます。残基の QM 部分が半整数に近い場合（丸めが曖昧）や、`charge` が力場が示唆する QM 領域の電荷と一致しない場合は警告が発せられます。

カットオフ埋め込みによる周期 QM/MM（v0.4。酵素の例は `examples/enzyme_qmmm.py`）。周期 System（NonbondedForce が PME、LJPME、Ewald の場合。`CutoffPeriodic` は拒否されます）では、コールバックはカットオフ埋め込みに切り替わります。各ステップで QM 領域（およびリンク原子の m1）を 1 つのイメージに再イメージングし、いずれかの原子が QM 原子の `embeddingCutoff` 以内にある MM 残基だけを、最近接イメージの全体として埋め込みます：

```python
mixed = potential.createMixedSystem(topology, system, atoms=qm_atoms, embeddingCutoff=1.2 * unit.nanometer)
```

**これは近似です**：カットオフ以遠の QM–MM 静電相互作用は無視され（PME と整合しません）、カットオフを跨ぐ残基はエネルギーを不連続にするため、厳密な NVE エネルギー保存は期待できません。`embeddingCutoff` と QM 領域の広がりの合計は、最も狭いボックス幅の半分未満でなければなりません（毎ステップチェック）。タイミングログは毎ステップ `n_embed_groups` と `embed_changed` を記録します。

ONIOM（2 層減算 QM:QM。ここでは水 0 に HF/STO-3G、5 水すべてに xTB——`examples/oniom_water_cluster.py` 参照）：

```python
from openmmorca import ONIOMPotential, ORCAPotential

oniom = ONIOMPotential(
    high=ORCAPotential(method="HF", basis="STO-3G", extra_keywords=("TightSCF",)),  # model 領域
    low=ORCAPotential(method="XTB"),                                                # 全体系
)
system = oniom.createONIOMSystem(topology, atoms=[0, 1, 2], forceGroup=0)
```

エネルギーは標準的な減算の組み合わせ `E_high(model) + E_low(full) − E_low(model)` です。層間は機械的に結合し（QM 層間の点電荷埋め込みなし）、model 領域は分子全体でなければならず、System は力場項を持ちません。各ステップで 3 回の QM 計算（高 1、低 2）が 3 つの独立したバックエンドを通じて行われ、各 restart チェーンのサイズは自己整合します。`high` の charge/multiplicity は model 領域を、`low` は全体系を記述します。低層の model 領域計算は `high` の charge/multiplicity を使うため、low 層のみの原子は帯電・開殼でも構いません。`createONIOMSystem` を呼ぶたびにこれら 3 つのバックエンドが新規作成されます。`oniom.summarize_timings()` / `oniom.close()` が両層を集約します。

QM/MM 水クラスター NVE（1 ps、エネルギー保存チェック）：`examples/qmmm_water_cluster_nve.py`。ONIOM 版：`examples/oniom_water_cluster.py`（デフォルト 400 ステップ = 0.1 ps。1 ps は `--steps 4000`）。リンク原子ジペプチド NVT：`examples/link_atom_dipeptide.py`。`examples/analyze_nve.py DIR` は `DIR` 内の証拠 CSV（「テストの実行」参照）から NVE ドリフト/標準偏差と restart 再現性の差を再計算し、入力ハッシュつきの `summary.json` を書き出します。

酵素の実行は 2000 ステップ（1 ps）未満ではスモークチェックのみを報告し、`Task 22 acceptance: NOT_EVALUATED` となります。1 ステップの実行も可能です。`--reaction-bond I J` を繰り返し指定すると、ゼロ始まりの OpenMM 原子インデックスで反応性結合を個別に監視できます。設定された安定性チェックは残りの QM 結合を対象とし、元の Task 22 の全 QM 結合ゲートは別途報告されます。終了コードはスモーク/設定された安定性チェックに従い、完全な受け入れ判定は `acceptance.json` に記録されます。`steps.csv` の各行は実行中にフラッシュされ、失敗時も完了済みの行が保持されバックエンドは閉じられます。

準備/平衡化キャッシュには対応する `.manifest.json` が必要です。ステージ設定、入力/成果物ハッシュ、バージョン、原子マッピング、ボックス情報を検証します。manifest のない旧キャッシュや設定が一致しないキャッシュは保持された上で拒否されます。新しい `--outdir` で再構築してください。後から QM 手法や領域を変えても MM キャッシュの同一性は無効になりません。QM/MM 実行は既存の実行を黙って上書きしません。実行成果物（`run.json`、`steps.csv`、トラジェクトリ、`checkpoints/`、最終 State）のある出力ディレクトリは、`--archive-existing` で `archive/run-<UTC 時刻>/` に移さない限り拒否されます。

`examples/analyze_enzyme_run.py OUTDIR [--json report.json]` は完了した実行を成果物から再計算します：実行長、温度、結合偏差、埋め込み変化、SCF サイクル/リトライ、タイミング。さらに DCD フレームから QM 結合偏差を独立に再計算し（mdtraj が必要）`steps.csv` と照合します。再計算した Task 22 判定を `acceptance.json` と比較し、`run.json` の由来情報（バージョン、ホスト、コマンド、git コミット）を含め、不整合があれば終了コード 1 で終了します。

Checkpoint と再開：`--checkpoint-interval` ステップごと（デフォルト 100。10 ステップの DCD 間隔の倍数）と最終ステップで、OpenMM checkpoint と移植可能な State XML を書き出し、`steps.csv` の長さ・ハッシュとともに `run.json` に記録します。`run.json` には状態（running / completed / failed）と全ての試行も記録され、完了すると `final_state.xml` と `final.pdb` を書き出します。`--resume checkpoint` は実行の同一性を確認した上で新しいプロセスで継続し、Langevin 乱数状態を復元します。同じプラットフォームと OpenMM バージョンが必要です。決定論的バックエンドでは中断なしのトラジェクトリを正確に再現します。ORCA の場合、再開の最初のステップは fresh SCF 初期推定を使うため、一致は SCF 収束精度内にとどまります。`--resume state` は State から新しいシードで再開し、トラジェクトリは再現しません。checkpoint 前の `steps.csv` はバイト単位で保持され、checkpoint 後に再計算された行は `steps.superseded.attempt-NNN.csv` に保存されます。各試行は独自のトラジェクトリファイルを書き、`--steps` を大きくすると完了済みの実行を延長できます。

## パラメータ（`ORCAPotential.__init__`）

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `method` | 必須 | ORCA の simple keyword。例：`"HF"`、`"PBE0"`、`"XTB"`、`"r2SCAN-3c"` |
| `basis` | `None` | 基底関数系。複合手法 / xTB では None |
| `charge` / `multiplicity` | 0 / 1 | QM 領域の総電荷と多重度（v0.3 以降は「QM 原子 + リンク H」で数えます） |
| `nprocs` | 1 | >1 で `%pal` を書き、MPI 環境変数を設定（ユーザー設定値は上書きしません） |
| `maxcore_mb` | 2000 | MPI プロセスあたりのメモリ（ORCA `%maxcore` 準拠） |
| `extra_keywords` | `()` | そのまま追加する simple keywords（`RIJCOSX`、`def2/J`、`TightSCF`…）。タスク種別のキーワード（`SP`/`Opt`/`EnGrad`…）は拒否 |
| `extra_blocks` | `()` | そのまま追加する `% block` 文字列。`%pointcharges`/`%pal`/`%maxcore`/`%moinp`/`%output` は拒否（バックエンドが管理） |
| `scratch_root` | `None` | デフォルト `/tmp/openmmorca-<user>`（tmpfs） |
| `restart` | `True` | `.gbw` を MORead 初期推定に使用（失敗時は fresh SCF に 1 回自動フォールバック） |
| `keep_failed` | `True` | ステップ失敗時に失敗バンドルを保持 |
| `max_fresh_retries` | 1 | MORead 失敗後の fresh-SCF リトライ回数 |
| `timeout_s` | `None` | 1 回の ORCA 呼び出しのタイムアウト（秒） |
| `orca_path` | `None` | デフォルトで `OPI_ORCA` または PATH を参照 |
| `backend_factory` | `None` | カスタムバックエンドの注入（テスト用） |

**`createSystem` / `createMixedSystem` / `createONIOMSystem` を呼ぶたびに、専用のスクラッチディレクトリを持つ新しいバックエンドインスタンス（ONIOM は 3 つ）が作成されます**。1 つのバックエンドがサービスする Context は 1 つだけです。

## スクラッチディレクトリと失敗バンドル

```text
<scratch_root>/<uuid>/
├── current/     毎ステップ上書き：qm.inp qm.out qm.engrad qm.pcgrad qm.gbw pc.pc guess.gbw qm.property.json
├── restart/     last_good.gbw（MORead 初期推定）
├── failures/    failure_step_NNNNNN/ ← 失敗の現場
└── timings.csv  ステップごとの所要時間（write/orca/read/total、SCF サイクル数、restart 使用状況）
```

失敗時には `ORCACalculationError` / `ORCAOutputError` / `ORCATimeoutError` が送出され、メッセージ末尾に失敗バンドルのパスが付きます。再現：

```bash
cd <scratch>/failures/failure_step_000123 && $OPI_ORCA/orca qm.inp > rerun.out
```

`ORCAPotential.summarize_timings()` はバックエンドごとの mean/P50/P95 所要時間を返します（コールドスタートの最初のステップは自動的に除外）。

## 既知の制限

- 周期系は上記の近似カットオフ埋め込みのみ。ONIOM は非周期です。QM/MM 領域はリンク原子で単結合を切断できます（上記参照）。ONIOM の model 領域は分子全体のままです。ONIOM の層間は機械的結合（埋め込みなし）です。
- PythonForce を含む System は XML シリアライズできません（コールバックがロックと一時ディレクトリを保持するため）。
- 毎ステップにプロセス起動の固定オーバーヘッドがあります（シリアル ≈0.4 s）。QM 領域が小さい場合は無視できません。
- NPT では `MonteCarloBarostat` の各試行が 2 回の追加 QM 計算を引き起こします（デフォルト頻度 25 で QM コスト +8%）。
- 埋め込み電荷は `%pointcharges` ファイルを経由します。inline `Q` は禁止です（MM–MM 静電気を二重計算し pcgrad も得られません）。
- SCF 失敗時に古い力を返すことはありません。リトライが失敗すればハードエラーとし、MD を停止します。
- 酵素サンプルの checkpoint 再開は、決定論的バックエンド・Reference プラットフォーム上でのみビット単位の一致を検証済みです。中断された ORCA 実行の実際の再開はまだ行っていません。再開の最初のステップは fresh SCF 初期推定のため、ORCA 再開トラジェクトリは中断なし実行と SCF 収束精度内でのみ一致します。

## テストの実行

```bash
export OPI_ORCA=/path/to/orca
python -m pytest                    # デフォルト：ユニットテスト + 高速 ORCA テスト（slow を除く）
python -m pytest -m orca -v         # ORCA が必要なテストのみ
python -m pytest -m slow -v         # NVE と restart 再現性（長時間）

# slow テストの機械可読な証拠を保存し（T06）、再計算する：
OPENMMORCA_EVIDENCE_DIR=$DIR python -m pytest -m slow -v     # qmmm_nve.csv、restart.csv を書き出す
python examples/oniom_water_cluster.py $DIR/oniom_nve.csv --steps 4000
python examples/analyze_nve.py $DIR
```

ORCA がなければ `orca` マークのテストは自動的にスキップされます（注意：デスクトップ Linux では `/usr/bin/orca` がスクリーンリーダーの場合があります。conftest は ELF バイナリのみ受け付けます）。`tests/test_platform_consistency.py` は倍精度 CUDA と Reference でリンク原子・周期イメージング・仮想サイトを比較し、使用可能な CUDA デバイスがなければスキップします。`tests/test_packaging.py` はインストール済みメタデータが古い場合に失敗します。`pip install -e . --no-deps` で修復してください。`examples/analyze_enzyme_run.py` の DCD チェックには mdtraj が必要で、未インストールならその項目をスキップします。

現在の検証状態と証拠：[TODO.md](TODO.md)（状態表）と [docs/validation/](docs/validation/)。

## 謝辞

- [openmm-pyscf](https://github.com/Gallicchio-Lab/openmm-pyscf)（Gallicchio Lab）：本プロジェクトの OpenMM 側の系修正ロジックはこれを移植したものであり、結合項削除ルールと O(N²) exception 処理に修正を加えています。Gallicchio Lab に感謝します。
