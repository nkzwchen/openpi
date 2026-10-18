# SO101 OpenPI：绝对位置采集与 HG-DAgger

当前任务是 `pick blocks from table`。此实现采用主视角和腕部视角的完整 RGB 画面，控制全部六个关节（包括夹爪），不裁 ROI、不冻结关节、不做 DeltaActions/AbsoluteActions 转换。它取代旧交接文档中的“试管插入、两个 ROI、18维状态、3维增量”实施方案；旧数据不能直接改标签后混入。

训练吞吐、JAX INT8/FP8实测、双卡epoch估算和10Hz/RTC建议见 [2026-10-18性能报告](benchmark-20261018.md)。

## 仓库与环境

本地和云端共同使用 `git@github.com:nkzwchen/openpi.git` 的 `feature/so101-openpi-dagger` 分支，通过提交号核对版本。

- 本地：`/home/nkzwchen/pi_project/openpi`。
- 云端本次开发与验收：`/root/gpufree-data/openpi-dagger`，是同一个 Git 仓库的独立 worktree。
- 云端原目录 `/root/gpufree-data/openpi` 保留原 `main` 和未提交的镜像源锁文件修改。
- 本地已有的 client 依赖和锁文件修改也保留，未纳入本功能提交。
- 数据、权重、验证产物在 `data/`、`assets/`、`checkpoints/`，不入 Git；凭据不要写进脚本或文档。

采集环境使用本地 LeRobot 的 `SOFollower` / `SOLeader` API；训练环境使用 OpenPI 固定依赖。两端通过轻量 WebSocket 协议通信。不要为了安装训练依赖而升级现有机器人环境。

当前本地 `lerobot` 环境有相机/机器人相关依赖，但缺少 `websockets` 和 `msgpack`。离线预检与纯人工采集不需要它们；运行远程策略前应在**独立环境或依赖目录**补齐这两个客户端依赖，并加入 `PYTHONPATH`，不必安装完整训练包。

以下命令均在仓库根目录运行，使用对应环境的 Python，并设置：

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:${PYTHONPATH:-}"
```

## 数据契约

版本：`so101_openpi_absolute6_v1`；采集频率 10 Hz。

| 字段 | 含义 |
|---|---|
| `observation.images.cam_main` | 完整主视角 RGB 视频 |
| `observation.images.cam_wrist` | 完整腕部视角 RGB 视频 |
| `observation.state` | 从臂六个关节的实测位置，非18维派生状态 |
| `action` | 发送至从臂的六个绝对位置目标，非增量 |
| `task` / prompt | `pick blocks from table` |

六维顺序固定为 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`。主从臂都显式设置 LeRobot `use_degrees=False`：前五维是标定范围 `[-100,100]`，夹爪 `[0,100]`，不是弧度或角度。`max_relative_target` 只是 LeRobot 对单次目标变化的限制，不改变绝对动作语义。原始记录同时保存请求和实际发送目标；受限动作首版不纳入专家训练。

原始 pkl 保留全尺寸图像和逐步审计信息，异步写入；导出后为两路 MP4 + parquet。训练的 `ResizeImages(224,224)` 只做模型输入尺寸适配，不是 ROI。当前输出是供本库本地 v3 reader 使用的 LeRobot v3 子集，不承诺可直接上传为具备全部 Hub 元数据的通用 LeRobot 数据集。

## A：离线复核与导出

原始记录格式为 `{transitions, meta}`；只加载自己可信的 pickle。成功/失败标签不自动成为专家动作标签。纯人工示范也需要复核；策略自主动作默认不用于监督。

先查看新记录的字段与契约：

```bash
python -m openpi.so101.export inspect data/recordings/<batch>/ep_0000.pkl
```

打开离线复核窗口，两路画面、状态和动作逐帧显示；使用开始/结束按钮选取值得模仿的人工区间，结束按钮包含当前帧。保存产生独立 `.expert.json`，重做复核会备份上一版本，不修改原 pkl：

```bash
python examples/so101/review_experts.py \
  data/recordings/<batch>/ep_0000.pkl --reviewer <操作者>
```

也可生成待填写的 JSON 模板（默认 `pending`，不自动批准）：

```bash
python -m openpi.so101.export review-template data/recordings/<batch>/ep_0000.pkl
```

JSON 中区间使用原 `transitions` 下标，半开区间 `[start,end)`。批准必须有匹配的原文件 SHA256、`reviewed=true`、`status=approved`、非空 reviewer 和明确 intervals。整段排除、未复核、模型步、质量异常、受限/拒绝动作不会被偷偷训练。

导出到一个不存在的新目录：

```bash
python -m openpi.so101.export export data/recordings/<batch>/*.pkl \
  --output data/blocks_v001 --horizon 10
```

实际使用时应传入多个独立采集批次；同一批次的全部 episode 固定分到同一 split，避免相邻窗口泄漏。导出报告 `report.json` 明确列出 train/validation、有效起点、短区间、各拒绝原因；`manifest.jsonl` 保留原文件 hash、原帧号、请求与发送动作、操作者、模型版本等信息。不同标定/相机契约分开导出。输出不覆盖原目录，失败目录标记 `INCOMPLETE`。

只允许完整 H 步专家窗口进入训练。人工/模型切换、暂停、丢步、时间跳变、done、复核区间边界均断开序列；不压紧跨越被筛掉的帧。短段和尾部不足 H 的起点单独计数，不重复末尾动作补齐。完整区间仍保留尾部帧供前面有效起点查询。

## B：模型配置、统计与训练

新增配置 `pi05_so101_absolute6`；旧 `pi05_so101` 不改语义。模型原生投影仍为32维，state/action 补齐；`loss_action_dim=6` 只监督真实六维，避免补零维度稀释损失。未使用的动作通道在加噪和采样积分时保持为0，避免未监督通道的随机漂移重新影响有效动作。π₀.₅ 通过离散 state tokens 使用六维实测状态。

归一化只从训练 split 的有效窗口计算，asset_id 与导出报告 hash 绑定。生产训练入口拒绝合成数据、旧契约、空 train/validation、未完成导出。以下命令需要已有的新任务人工数据；本次没有启动完整训练：

```bash
python scripts/so101/train.py --dataset data/blocks_v001/train \
  --exp-name blocks_v001 --compute-norm-only

python scripts/so101/train.py --dataset data/blocks_v001/train \
  --exp-name blocks_v001 --steps 20000 --batch-size 2
```

默认双骨干 LoRA；`--weights` 可指定兼容的 π₀.₅ 预训练 `params` 路径，默认使用官方 GCS 地址。成功训练后，实验根目录写入 `so101_contract.json`，绑定数据版本、标定/相机契约、LoRA设置及统计 asset_id。服务入口按绑定恢复，避免加载时丢失 LoRA 参数。需要完整微调可加 `--no-lora`，应另行确认显存容量。

## C：离线预检、假机器人和网络测试

默认预检不会打开串口、相机或 ROS：

```bash
python examples/so101/record_openpi_so101.py
python -m unittest discover -s tests/so101 -v
python scripts/so101/offline_acceptance.py --output data/acceptance-local-new
```

在有 OpenPI 环境的云端运行模型检查：

```bash
python scripts/so101/offline_acceptance.py \
  --output data/acceptance-cloud-new --model-smoke
```

`--output` 必须不存在。合成样本显式标记，不得冒充真实示范。模型检查使用随机初始化的小型 Gemma + 真实 SigLIP，验证视频读取、完整窗口采样、归一化往返、有限 loss、动作输出层非零梯度、一次参数更新、参数形状兼容、保存重载和 `(10,6)` 策略输出。它不证明预训练权重已成功加载，不是完整 LoRA/SFT 验收，也不证明任务成功率。

控制测试使用假机器人与假策略，覆盖两阶段空格接管、超时/断网停止、迟到结果失效、chunk 单次消费、按键优先于命令、接管上限、GUI key gate 的跨阶段输入过滤、异步写盘快照，以及真实 loopback WebSocket 的编码与接收超时。

2026-10-18 验收：云端24项测试全部通过；本地23项通过，1项依赖OpenPI/JAX的服务配置测试在本地跳过、在云端通过。云端模型检查的 loss 为5.9213、动作输出层梯度范数13.1281，未使用通道梯度为0，保存重载的固定输入采样误差为0。完整摘要与所测代码提交号见 [offline_validation.json](offline_validation.json)。这些数字仅验证合成数据链路，不表示真实任务训练效果。

## 未来部署与真机边界

本次没有执行本节的真机命令，也未打开现场 GUI 或相机。必须先核对串口、标定、两路真实相机、主从位置映射、全关节限位、夹爪和停止行为。新实现采用绝对 LeRobot 主从位置映射，不复用试管插入的差分锚点、冻结维度或自动归位。

纯人工采集的命令模板（设备参数需要现场确认）：

```bash
python examples/so101/record_openpi_so101.py --hardware \
  --follower-port <从臂串口> --leader-port <主臂串口> \
  --main-camera <主相机设备> --wrist-camera <腕相机设备> \
  --out data/recordings
```

GUI/终端逻辑：READY 按 C 开始；策略模式下第一次空格进入暂停且不发送新模型命令，第二次空格进入人工控制；人工时空格交回模型并清空旧 chunk。D 保存、Q 整段排除且保留原始记录、X 退出。RETURN 由人工回程，回程不记录，C 回到 READY；不自动运动归位。GUI 接管次数设置独立存放在 `~/.config/openpi-so101/control_panel.json`。终端模式使用 `--terminal-controls --max-takeovers N`。

训练完成后在云端启动服务（选取实际存在的 checkpoint 步目录）：

```bash
python scripts/so101/serve.py --dataset data/blocks_v001/train \
  --checkpoint checkpoints/pi05_so101_absolute6/blocks_v001/<step> \
  --bind 127.0.0.1 --port 8000
```

通过 SSH 端口转发连接服务，再给录制入口增加 `--policy-uri ws://127.0.0.1:8000`。默认预测 H=10、实际执行1步，两者独立。推理图像在客户端等比例letterbox为224×224，原始录制保留全分辨率；可用 `--inference-image-size 0` 关闭传输缩放。模型预热在设备和控制窗口打开前进行；worker 中执行远程推理，控制循环不会等待它。没有新结果时保持，不重复播放旧动作；暂停/人工/交回/结束/reset 都使在途请求失效。超时、断网、非法动作进入停止流程，不自动重连。可调 `--timeout-s` 和 `--max-age-s`，需用端到端延迟实测确定，不能保证远程10Hz。

两个相机由 LeRobot 分别读取，当前没有硬件级同步保证；原始记录包含控制时序和慢步标记，但还需现场确认相机时间戳/新鲜度。网络推理不会阻塞按键，但串口、相机读取和发送仍是同步驱动调用，其最坏延迟需要真机阶段测量。

## 代码来源

GUI、键盘、接管设置和异步录制器复制自本地 `hil-serl` 的2026-10-18工作区（该工作区包含未提交修复），保留独立副本并做格式及任务说明调整。来源清单见 `source_provenance.json`。原 HIL-SERL 文件、旧录制入口、原数据和权重均未修改。
