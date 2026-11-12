# SuperMimic 官方开源与本地迁移审查

核查日期：2026-11-12，Asia/Shanghai。结论：**论文作者团队已经公开 SuperMimic 实现，值得成为下一阶段的官方对照基线；当前发布包仍有依赖与复现入口缺口，不能直接替换社区实现或接续旧权重。**

截至 2026-11-13 02:20，官方源码与独立 Windows/Linux Conda 环境均已部署，分支 `ljd/official-supermimic` 已推送至现有公开仓库。Windows 与 Linux 各 217 项官方测试通过。原始 compiled 目标拼接在两次 H100 试跑中均于至少 201 次采样后发生 CUDA illegal memory access；同步报错定位于 `_compiled_assemble_goal` 的 Torch/Triton 内核。切换该函数为作者原始 eager 运算后，以下四轮训练和全40评估均完成，最终检查点可读且数值有限。此对照支持编译路径相关的判断，尚未证明具体底层根因。

LAFAN40 使用同一冻结动作集、4096个训练环境、seed 42，评估从每条动作首帧开始，关闭噪声与域随机化，采用作者原生终止条件；所有失败均保留。

| 训练配置 | 采样步数 | 完成整段动作 | 等权动作平均归一化回报 | 训练时间 |
|---|---:|---:|---:|---:|
| 公开默认参数 + eager goal | 1,001 | 0/40 | 0.030332 | 205.42秒 |
| 公开默认参数 + eager goal | 16,001 | 39/40 | 0.843299 | 2129.56秒 |
| 本地论文参数适配 + eager goal | 1,001 | 17/40 | 0.285147 | 593.65秒 |
| 本地论文参数适配 + eager goal | 16,001 | 39/40 | 0.790469 | 2104.16秒 |

归一化分母为完整参考帧数乘每步最大正回报0.1，提前失败后的剩余参考贡献零；并另存按参考帧数加权的汇总。论文汇总权重未获确认。公开默认16k唯一失败为 `fightAndSports1_subject4`，在131.06秒触发 `anchor_pos`。这是训练动作覆盖和原生回报，尚无严格姿态/接触质量或留出集结论。只有一个种子；适配版同时改变优化器、观测历史与调度，不能把提升归因于单项改动，编译缓存也使上述时长不适合直接比较算法效率。

**重复性失败已确认：** `1113c` 对同一个默认16k检查点采用40环境、相同索引顺序的新评估入口，完成了40条动作，但得到38/40，与上述原始39/40不同。一致性检查按预设条件失败，后续24环境分批验证没有执行。随后未修改的旧入口重跑得到40/40，证明当前运行环境下旧入口自身也不重复；这不能反向证明新入口等价。保留全部原始结果与轨迹。因此39/40目前是一次观测，不是已验证的确定性成绩。 新诊断已验证两次数据及检查点哈希一致，40条动作的首帧状态和第一次动作逐位一致；后续状态在1至13帧内出现超过1e-5的分量差异。新增失败是 `fightAndSports1_subject1`（原12257步成功，新1250步失败）。[逐条轨迹诊断](../official-reproduction-20261112/original-vs-batched-trajectories.json) 与 [成绩比较](../official-reproduction-20261112/original-vs-batched-comparison.json) 已保存。`1113d` 已完成，旧入口重复运行的逐条最大归一化回报差为0.380654，最大长度差为5704步。作者依赖的 `mjlab/utils/random.py` 也注明 MuJoCo Warp 尚非完全确定性；具体到哪个内核仍未归因。

收据与逐条结果：[默认1k](../official-reproduction-20261112/lafan40-eager-pilot-summary.json)、[默认16k](../official-reproduction-20261112/lafan40-eager-16k-summary.json)、[适配1k](../official-reproduction-20261112/lafan40-paper-pilot-summary.json)、[适配16k](../official-reproduction-20261112/lafan40-paper-16k-summary.json)。默认16k和适配1k的逐条JSON由各自完整的40行云端日志重建，并与最终计数核对，记录了日志哈希；它们不是云端原始结果文件的逐字节下载副本。

适配16k的训练、配置、评估与协议，以及旧入口重跑的评估、协议和两份比较文件，现已通过短日志片段完整取回，八份文件均与服务器原始字节的SHA256一致，见 [收据清单](../official-reproduction-20261112/cloud-receipts-1113e-ledger.json)。适配16k唯一失败为 `fallAndGetUp1_subject5`，在57.70秒触发 `anchor_pos`；按参考帧数加权回报为0.796026。[旧入口重跑](../official-reproduction-20261112/lafan40-legacy-repeat-summary.json) 得到40/40、等权回报0.852879。重跑与原始运行仅有一条动作的首帧状态出现1.86e-9差异，首次动作全部逐位一致。只有单个训练种子和两次旧入口评估，尚不能可靠估计分布或判断配置优劣。

BONES 原始 G1 数据已完成下载与 SHA256 核验，142,220 条 CSV 全部转换成功，得到 51,915,358 个 50Hz 帧。转换仅适配原始厘米、角度和 Euler 字段，再调用作者的重采样及速度计算代码；未做名称筛选、可行性筛选或地面校准。5 条抽查与作者格式入口逐数组完全一致；内存有界打包在 16 条样本上与作者打包器完全一致。完整 13,575,450,990 字节压缩包已上传个人 AFS，服务器 SHA256 与本地一致；九个数据文件解包时的哈希均通过。解包器对附带校验清单的处理已修复并完成 16 条样本回归验证，云端九个数据文件已全部通过独立 SHA256 读回，最终目录为 `data/bones-packed-raw`，收据见 [bones-cloud-readback.json](../official-reproduction-20261112/bones-cloud-readback.json)。这些是未筛选的原始数据，尚不等于论文筛选后 129,514 条动作实验已复现。

BONES 原始全库已在单张H100上完成1,001步论文参数适配版试跑，平台状态为成功，三个最终检查点数值有限。训练耗时358.59秒，PyTorch分配峰值约30.85GiB（不包含Warp及其他CUDA分配）。这证明当前配置可载入全库并进行训练；没有完成142,220条动作的全库策略评估。见 [BONES试跑收据](../official-reproduction-20261112/bones-raw-pilot-result.json)。

新增 [分批全库评估入口](../official-reproduction-20261112/evaluate_motion_corpus.py) 复用上述原生评估协议，逐批重置环境及策略历史，并保留每条动作的结果和轨迹。按长度排序只改变批次顺序，不筛掉任何动作；完整结果必须含每个原始索引恰好一次。原始BONES在1024环境下，按原始顺序各批最长长度之和为184,425步，按长度排序为57,293步；这是参考长度计算，不是实际GPU速度实测。`1113c` 的首项一致性检查未通过，详见前述重复性问题；后续 `1113e` 在24环境排序评估的第二批重置处失败：第一批生成的传感器历史缓存属于 inference tensor，旧脚本在退出 `inference_mode` 后尝试原地清零。修复仅把所有批次的重置、观测及步进放入同一个推理上下文，模型构建与载入仍在其外，官方文件不变。使用固定依赖的真实 `ContactSensor.reset()` 在CPU上复现原错误，修复后的连续三次重置通过，见 [针对性验证](../official-reproduction-20261112/batch-inference-reset-validation.json)。修复版 `1113g` 已于02:14:30在单卡启动，两份脚本哈希通过，第一批已观察到第5,000步；两批验证尚未完成，首帧映射和完整轨迹等价性需分别报告。GPU分批一致性和全库执行尚未验收，见 [验证状态](../official-reproduction-20261112/corpus-evaluator-validation.json)。

以下静态审查与历史对照保留初次审查时的证据边界；当前执行合同见 [experiment-contract.json](../official-reproduction-20261112/experiment-contract.json)，数据验证见 [bones-data-validation.json](../official-reproduction-20261112/bones-data-validation.json)。官方源文件未修改；离线训练包装器替换 W&B 产物上传/清理逻辑，并提供显式 `--goal-backend eager` 对照开关；该开关调用作者相同的目标计算函数，绕过其单独的 Torch 编译包装。公开默认配置保持作者学习参数，论文适配版另改优化器、观测历史与学习率调度，不能用这些实验宣称官方原始性能复现。

## 1. 官方身份与发布日期

- [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic) 公开、Apache-2.0，README 链接原论文及项目页，版权声明为 The SuperMimic Authors；Git 历史含论文第一作者 qiayuanl 的实现提交。
- [论文 arXiv:2610.28480](https://arxiv.org/abs/2610.28480) 的题名、作者、项目链接与仓库一致。当前 v2 更新于 2026-11-09。
- GitHub API 的仓库创建时间为 `2026-11-10T10:46:56Z`，北京时间 11 月 10 日 18:46:56；这证明该仓库此时创建，**不据此推定精确转为公开的时刻**。历史提交时间也不能用来推定代码当时已公开。
- 必需依赖 [HybridRobotics/mjlab_research](https://github.com/HybridRobotics/mjlab_research) 现在也能公开获取，本轮 HTTPS fetch 成功。旧 issue 中的依赖不可访问状态不应沿用为当前结论。
- 两个仓库本次 API 查询均没有 GitHub Release。主仓库 README 未给出可直接核验的论文模型下载清单；未测试作者 W&B 模型或数据是否允许匿名访问。

因此，之前“没有完整官方训练实现，只能选择社区版”的判断需要更新；此前研究记录应保留为当时的历史事实。

## 2. 本轮实际 Git 结果

工作仓库：`publication/SuperMimic-Reproduction`；外层 SuperMimic 目录本身不是 Git 仓库。

| 项目 | 固定值 |
|---|---|
| 新分支 | `ljd/official-supermimic`（原名 `codex/official-supermimic-review-20261112`） |
| 起点 | `codex/sports-video-xy-20261110`，`2efb0fcdce28e9920c16986b6456d0bcbeb060ca` |
| 官方主仓库 | `abc124c1bd5976341034668af3d12c087f2f1f55` |
| 官方 mjlab 依赖 | `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd` |
| 引入位置 | `upstream/supermimic/` 与 `upstream/mjlab_research/` |
| 导入量 | 583 个文件，62,938,387 字节；原始 Git 文件模式保留 |

起点工作区干净。检查了 origin 的实时分支：main 为 `9f6a70d`，回放研究分支为 `b190405`，与本地相应跟踪记录一致。两套实现没有共同 Git 祖先，因此通过独立目录导入固定版本，未合并无关历史。官方 Git 历史保留在 `official/main` 与 `official-mjlab/main`。既有实验代码、检查点清单和历史结果不因导入而改变。

## 3. 五个会话与现有实验的关系

本轮通过 `read_thread` 读取五个指定会话的相关回合，并以本地源码和实验文档补充核对。以下训练结果是历史证据，本轮没有重新跑模型或核实云端状态。

| 会话 | 已有积累 | 官方开源后的定位 |
|---|---|---|
| 调研 SuperMimic 社区复现 | Uni-SuperMimic、G1/LAFAN1 旧40；16K 基线；22K/26K/30K；专项与教师约束 | 继续保存社区结果，建立官方实现的独立基线；不同控制器、后端及参考版本不能串成同一训练曲线 |
| SuperMimic 社区复现历史 | 统一55、连续120秒、失败尾部、接触及回退账本 | 评测经验和失败分析可复用，原有分数不能直接当官方分数 |
| 评估 SuperMimic 工程创新 (2) | ARX5 操作、任务专用世界模型、泛化边界 | 当前官方仓库主要是人形动作跟踪；不会自动提供 ARX5/LIBERO 的统一实现 |
| 整理到github | 公开证据仓库、旧经验回放四比例×三种子的课程实验 | 保持已冻结协议；所读最新会话确认12组训练完成并进入统一评测，不能凭该回合断言最终效果已齐备 |
| 评估 SuperMimic 视频复现方案 | GEM-X/SOMA-X 五段视频1956帧；F0/F1 体育续训；旧能力回退诊断 | 前端与数据质量问题独立于控制器，官方代码不能自动修复身份、尺度、地面及接触误差 |

现有 [体育实验记录](../experiments/sports-video/ACTUAL-SPORTS-PILOT.md) 还限定了：F0/F1 都完成单条14.22秒动作；选定旧动作上 F0 在13.36秒失败、F1完成168.2秒。旧40批量诊断有预算截断和数值一致性失败，F1另一个旧动作失败尚未由单环境清除。**这不是全旧40保留，也没有证明 XY 输入总体占优。**

原16K、统一101D分支、专项失败与体育结果都保留各自语义。LIBERO 暂停任务没有因本次导入而恢复。

## 4. 官方代码提供了哪些有价值的对照

| 方面 | 本地社区路线 | 本次官方源码 |
|---|---|---|
| 框架 | UniLab / Uni-SuperMimic | mjlab_research / SuperMimic |
| 状态表达 | 社区世界模型及99D/101D策略接口 | GenCoord 与 MaxCoord 两条表达路线 |
| 观测 | 历史模型为99D，XY扩展101D | 默认 `actor_obs=rep_state`、历史长1；可选 sensor proprio 为93D |
| 策略 | 单目标编码器与共享解码器 | `MultiEncoderAE`，motion/velocity 目标编码器及共享解码器 |
| 世界模型训练 | 短窗口动力学拟合及想象训练 | 多步状态损失、解析积分/FK、物理上下文条件化 |
| 优化 | 既有社区配置 | hidden Linear 使用 Muon，其余 Adam；torch.compile 和 CUDA graph |
| 重置采样 | 固定/混合旧新库及回放实验 | 基于低回报分箱的 `ReturnSampler`，含 explore/focus/keep |
| 数据 | G1 CSV 与社区预处理 | root/joint NPZ 或 packed NPY，显式 joint_names 与 FK 校验 |

数字需按接口分别解释：29关节时，GenCoord 全状态为71D、表示状态68D；sensor proprio 为 `3+3+29+29+29=93D`，另有目标窗口。默认并非“93D＋10帧历史”的部署配置。参考窗口为10帧、间隔5个控制步，约覆盖未来0.9秒。详见 [representations.py](../upstream/supermimic/src/supermimic/representations.py) 和 [runner 默认配置](../upstream/supermimic/src/supermimic/runners/supermimic_runner.py)。

官方仍采用“真实仿真采集 → 拟合动力学 → 世界模型内BPTT优化策略”，与社区版方法主线一致。新结构和实现选择值得实验比较；**源码存在不能证明它在我们55条动作或体育素材上更好。**

## 5. Review 发现：优先解决哪些问题

### P1：两个当前仓库的依赖锁不一致

[supermimic/uv.lock](../upstream/supermimic/uv.lock) 记录 mjlab 1.5.0；[mjlab_research/pyproject.toml](../upstream/mjlab_research/pyproject.toml) 已是1.6.0。用版本约束解析逐项检查，发现8个直接依赖冲突：

| 依赖 | SuperMimic lock | 当前 mjlab 要求 |
|---|---|---|
| Torch | 2.10.0 | >=2.14.0 |
| Warp | 1.14.0 | >=1.15.0 |
| MuJoCo | 3.10.0 | ~=3.16.0 |
| MuJoCo-Warp | 3.10.0.1 | ~=3.16.0 |
| rsl-rl-lib | 5.4.0 | ==5.5.2 |
| viser | 1.0.27 | >=1.1.1 |
| mjviser | 0.0.14 | >=0.0.16 |
| mediapy | 1.2.6 | >=1.2.7 |

影响：旧 lock 无法证明这两个当前提交的组合已验证；允许重新解依赖又会改变环境版本。这里核实的是元数据冲突，尚未执行 `uv sync`，不能虚构具体安装报错。下一步应在独立 Linux/Python3.13 环境确认兼容组合、生成实验专用锁并运行测试，不能复用旧社区环境解释官方结果。

### P1：warm-start 不是完整续训

[runner 的 load_wandb_checkpoint](../upstream/supermimic/src/supermimic/runners/supermimic_runner.py)（第201行）只载入 `world_model.pt`；随后重新构建 actor/critic。第729行起保存世界模型、actor、critic 参数，但没有保存/恢复优化器、回放池、ReturnSampler、迭代数及 RNG 的完整训练状态。

影响：不能用这个参数声称“从原策略接着训练”，也不能作为我们中断恢复与增量学习实验的等价入口。正式续训前需要单独实现并验证恢复合同。旧16K/22K的 state_dict、normalizer 与接口都不同，也没有现成官方转换器。

### P1（论文复现范围）：默认命令缺少全量实验闭环

[train.py](../upstream/supermimic/src/supermimic/scripts/train.py) 默认引用一条 `lafan_run1_subject5`，支持本地多NPZ及 packed 数据，但没有随仓库提供论文全库固定清单、对应可核验的模型包、完整复现实验配置和结果导出命令。公开的 [benchmark.py](../upstream/supermimic/src/supermimic/scripts/benchmark.py) 测量仿真/世界模型速度，不能替代129,514条动作的逐条成功与回报评测。

SuperMimic 训练入口使用单设备，没有找到自己的 DDP/torchrun 联合训练实现。依赖框架的 PPO 多GPU入口不等于此算法已接通论文双卡流程。4卡可先用于独立实验并行，不能直接宣称官方四卡训练已可运行。

### P2：演示、离线评测与数据转换入口需要分开

[play.py](../upstream/supermimic/src/supermimic/scripts/play.py) 要求 W&B run，读取模型和配置，并显式启用 actor observation corruption；它是交互演示入口。尚未验证各观测路径下的实际噪声效果，不应直接当成固定无噪声评测。

依赖中确实有 [csv_to_npz.py](../upstream/mjlab_research/src/mjlab/scripts/csv_to_npz.py) 和 [pack_motions.py](../upstream/mjlab_research/src/mjlab/scripts/pack_motions.py)。前者按 xyzw 输入、wxyz 输出，支持30→50Hz，并生成 root pose/twist、joint state 和 joint_names；其 CLI 写 `/tmp/motion.npz` 后上传 W&B。后续本地转换可复用 `MotionLoader.arrays()`，另设显式输出路径，避免误把转换命令当成纯本地读写。本轮未调用上传入口。

旧CSV和体育CSV有复用可能，但必须核验关节顺序、四元数、真实帧率、root link与COM、坐标系及地面质量，不能只换扩展名。

### P2：发布CI没有覆盖默认分支

[ci.yml](../upstream/supermimic/.github/workflows/ci.yml) 的 push/PR 过滤为 `master`，而实际默认分支为 `main`；依赖检出仍使用 `MJLAB_PAT`。因此不能据现有配置认定公开 main 的每次提交都经过这些检查。这是源码配置审查，本轮未宣称上游测试通过或失败。

## 6. 对后续研究的判断

**建议采用官方实现作为新的对照底座，先完成一个可比较的小基线。** 官方 native 终止与我们的严格位置/接触门槛分别报告。论文项目页也明确全库动作是训练动作，因此其全覆盖不能被表述为未见动作泛化，更不能自动等价于严格运动质量。

持续学习仍值得研究，但研究问题应从“增加一些工程机制”收敛到：**旧动力学知识的保留和策略对旧状态的覆盖，哪一个决定旧技能是否遗忘？** 官方已有多编码器、速度指令接口及带 keep 项的困难分箱采样，不能再把这些概念本身包装成我们的独有创新。ReturnSampler 的 keep 也不等于显式旧/新任务回放，仍可在固定采集分布下设计世界模型回放 × 策略初始状态回放的2×2对照。

体育路线的关键仍是可靠参考和独立评估。原始视频提取、相机尺度、根轨迹与接触质量不会因为换了控制器而自动改善。官方 actor 目标接口主动去掉 root 相对位置；是否增加场地 XY 反馈是需要单变量验证的设计选择，需明确仿真/真机位置来源。

建议顺序：

1. 固定兼容的官方主仓库/依赖/环境组合，先过单元测试和单动作有限运行。
2. 接入同一旧40数据，建立官方 native + 现有质量指标的双报告；再接入新15，固定分母并保留失败。
3. 取得完整保存/恢复功能后开展旧新技能实验；社区16K保留为历史控制，官方模型独立初始化或用可核验的官方权重。
4. 最后复用经过质量审核的体育动作，并按比赛/选手分割训练与留出集。

上述是后续实验建议，没有在本轮执行。既有课程项目先保持冻结版本，避免在分析中途替换代码、数据或评测合同。

## 7. 本轮验证与边界

- 全部583个导入文件与对应 upstream Git blob 一致；每文件SHA256已记录，Git树与文件模式保留。
- Python3.13.9 对351个官方/依赖Python文件完成语法解析；无源码修改。
- 检查8个直接依赖版本约束冲突；这是已证实的环境元数据问题。
- 原仓库 `check_release.py` 检查633个Python文件、695个JSON、209个报告链接及351个视频；唯一报错是31个既有、被Git忽略的大型训练产物。逐一确认均未被跟踪，未删除或提交它们。该脚本原始退出码为1，不能写成全套检查通过；未报告语法、JSON、链接、历史源码或视频哈希错误。
- [验证脚本](../tools/verify_official_import.py) 无需第三方训练依赖；[完整证据](../validation/official-upstream-review-20261112.json) 记录源版本、API元数据、静态核查和仓库检查结果。
- 初次静态审查时尚未安装运行依赖；后续安装、测试与云端试跑结果见文首的执行状态。本节保留初次导入检查，不代表当前环境仍不可用。

源码同步成功与实验复现成功是两个独立结论。初次审查完成源码同步；后续执行继续沿用这些固定版本，训练与评测结论必须另行验收。

## 8. 论文配置适配与数据筛选补充核查

`run_lafan_pilot.py --training-profile paper-a5-single-gpu` 通过独立的
[paper_profile.py](../official-reproduction-20261112/paper_profile.py) 配置全参数 Adam、
93D 本体感知的10帧历史，以及250,000次采样内衰减至初值0.03倍的余弦学习率。
官方导入文件保持原样；此入口是本地对论文 Tables A.4/A.5 的显式解释，不能称为作者发布的完整配置。
学习率以零起始的仿真采样步计数，在该步之后的模型/策略更新前生效；论文 iteration 与此计数方式的对应尚未获作者确认。
运行长度由命令显式指定，1k试跑不等于论文320k完整训练。

两项CPU数值检查通过：各网络参数仅进入一个Adam优化器，变化学习率下与标准Adam的更新一致；
调度端点及单调性正确，学习率张量地址保持不变。11月13日00:02启动的单卡 `1112f`
又通过三项适配层检查（15.68秒），包括CUDA graph重放使用变化后的学习率。
该任务已完成1,001步训练及完整全40评估，17条成功、23条失败；平台成功、退出码0，三个最终模型可读且数值有限。
该入口尚未解决完整状态恢复、双卡同步、速度编码器配置或全库筛选缺失。

按当前 SONIC 官方筛选脚本的固定版本
[`1e1ec4f`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/gear_sonic/data_process/filter_and_copy_bones_data.py)
对142,220个原始动作名做只读匹配，排除12,435条、剩余129,785条，与论文名称筛选后的129,667条不一致。
当前SONIC规则还含 `handstand`、`cartwheel` 等关键词，不能未经成员核对就复制成SuperMimic筛选器。
论文只说明沿用SONIC并补充少数词，没有给出确切版本和完整关键词；后续可行性筛选也有未量化门槛。
因此暂不通过凑齐条数来建立“论文同一数据集”。详细计数与源文件哈希见
[只读筛选核查](../official-reproduction-20261112/sonic-name-filter-audit.json)，原始数据未改变。

另有未在适配层改动的差异：论文D.1把10%的keep项描述为历史softmax权重，
而固定源码 [reset_sampling.py](../upstream/supermimic/src/supermimic/runners/reset_sampling.py)
的 `_distribution()` 累加 `focus.double().square()` 后归一化。
这会更偏重曾经高度集中的困难分箱；它不是直接累计权重的同一个公式。
目前保留作者发布的实现并明确记录，不能声称此适配层已经逐项还原论文。

## 9. 当前有限作业（11月13日02:20）

原始 BONES 全量16k任务 `1113f`（UID `af1bd6ee0f7c415abfd1f7c14143a82f`）已于02:04:45启动，已观察到3,891次采样。它使用全部142,220条动作、51,915,358帧、4096环境、seed42和同一单卡论文参数适配；只增加训练长度，不改变数据筛选或参考。单卡、4vCPU、64GiB、零自动重启，程序最长3600秒，截止03:04:45，超时后另给30秒终止进程。当前运行状态不代表训练完成或评估成功。

批次修复诊断 `1113g`（UID `1b3745ad2a5e484e805a459424f0be4d`）最长1800秒，截止02:44:30。它使用24环境覆盖全部40条动作及末批填充，只修复推理上下文范围，并检查完整成员与非空轨迹文件。BONES 1k模型的全142,220条评估命令已准备，单卡最长7200秒；尚未提交，需先取得分批诊断的首帧映射与完整性证据。
