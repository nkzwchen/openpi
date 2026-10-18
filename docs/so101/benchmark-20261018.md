# 2026-10-18：冻结 VLM 训练、JAX 量化与 10 Hz 部署评估

本报告区分单卡实测、双卡估算与尚未完成的真机验证。云端实际只有一张 NVIDIA L40S（46068 MiB），没有租用或模拟第二张卡。JAX 0.5.3、Flax 0.10.2，保留现有 CUDA/依赖环境。所有输入和视频均为合成数据，没有打开机械臂、串口或相机；本报告不评价抓积木任务成功率。

官方 JAX 权重来自 `gs://openpi-assets/checkpoints/pi05_base/params`，已完整下载、可加载，云端缓存为 `/root/gpufree-data/openpi-bench-cache/pi05_base_ranged`。原 `/data/models/pi05_base` 的 PyTorch 权重未改动。

## 1. 测试设置

- 完整 π₀.₅：Gemma 2B + action expert + SigLIP；共3,353,433,872参数。
- 冻结全部 SigLIP 与 PaliGemma，包括其适配器；训练完整 action expert、时间与动作投影，共430,098,464参数。不是只训练动作输出头，也不是 LoRA 性能数字。
- AdamW，无 EMA；H=10，模型投影维度32，有效动作维度6，语言长度上限200，本次实际39个有效 token。
- 两路真实输入槽为主视角与腕部视角。当前代码还计算第三个 mask=False 的空图槽，因此实测包含3次视觉编码及相应前缀长度；没有把它当作“真正只算两槽”的结果。
- 640×480双路完整 RGB 合成视频，mp4v 编码、PyAV解码；模型输入 letterbox 到224×224。视频有细节和运动，原始示范明确标记 synthetic。
- 视频管线重复同一小型数据集以维持足够多批次、避免每几个批次重启 worker。不是实测30分钟真实数据的完整 I/O；真实视频编码、GOP、图像复杂度与 episode 长度会影响读取成本。
- 训练测20个稳定步；推理每项3次预热后40次，显式等待 GPU 完成。初始化、编译、加载权重、worker启动、保存 checkpoint 均不计入稳定吞吐。

## 2. 单卡训练结果

| 配置 | 平均训练步耗时 | 吞吐 |
|---|---:|---:|
| batch=2，GPU计算，完整尺寸初始化模型 | 237.8 ms | 8.41窗口/s |
| batch=8，GPU计算，官方预训练权重 | 约777 ms | 10.30窗口/s |
| batch=8，双路640×480，原生异步循环，0 worker | 约0.95 s | 约8.45窗口/s |
| batch=8，双路640×480，原生异步循环，4 workers | 约0.78 s | 约10.2窗口/s |

首批编译约48秒，4 workers冷启动约20秒。编译只计入首次启动；当前loader设置persistent_workers=False，跨数据epoch时还会重启worker，这部分约20秒/epoch另计。表中数据管线数值取稳定实测，最终完整 JSON 会保留各次结果与权重来源。

额外做了完全串行的对照：读取、解码、预处理 batch=8 约0.9秒，等待训练完成另约0.78秒，合计约1.7秒/步。**该串行对照不是原 `train.py` 的正常吞吐**：原脚本异步提交 JAX 更新后立即取下一批，CPU读取与GPU计算可重叠。4 workers 实测可基本隐藏合成视频的读取成本。

建议先使用 `--freeze-vlm --no-lora --batch-size 8 --num-workers 4`。新增 `--freeze-vlm` 冻结整个视觉语言骨干；该模式关闭EMA，避免维护冻结模型的无效副本。原 LoRA 路径仍保留，不指定这两个训练模式参数时原行为不变。

```bash
python scripts/so101/train.py --dataset data/blocks_v001/train \
  --exp-name blocks_frozen_vlm --freeze-vlm --no-lora \
  --batch-size 8 --num-workers 4 \
  --weights /root/gpufree-data/openpi-bench-cache/pi05_base_ranged
```

该生产入口仍要求真实、复核通过的新任务数据，不能将 benchmark 合成数据当作正式训练集。

## 3. 双卡与30分钟数据的估算

假设两张同型号 L40S、每卡batch=8、全局batch=16；以单卡端到端约10.2窗口/s为基准，按70%–90%的双卡扩展效率估计，吞吐约14.3–18.4窗口/s。通信拓扑、PCIe/P2P能力、CPU加载和内存容量尚未双卡实测。

冻结VLM后可优先采用数据并行（本库 `fsdp_devices=1`，两张GPU在batch轴并行），而非先将整个冻结骨干做FSDP。若显存不足再分片，分片通信开销需重新测试。两卡并不保证2倍加速；本机没有第二张GPU，因此以下均是估算。

30分钟×60秒×10Hz = **18,000个采样时刻**。两路相机不会把样本数翻倍；H=10、stride=1的窗口不会把样本数除以10。

若全部为已复核的连续专家动作、共E个episode，且每段至少H步：

```text
N_valid = Σ(L_episode − H + 1) = 18000 − 9E
每epoch更新步数 = floor(N_valid / 全局batch)   # 当前drop_last=True
保留最后不足一批的样本时使用ceil
```

| 每段时长假设 | episode数 | 有效窗口 | 双卡batch16步数：当前 / 保留尾批 | 双卡每epoch估算 |
|---|---:|---:|---:|---:|
| 10秒 | 180 | 16,380 | 1023 / 1024 | 约15–19分钟 |
| 30秒 | 60 | 17,460 | 1091 / 1092 | 约16–20分钟 |
| 60秒 | 30 | 17,730 | 1108 / 1109 | 约16–21分钟 |

以60秒/段为例，若按原批次留20%验证集，训练24段、14,184有效窗口：当前 **886步/epoch**（保留尾批887步），约 **13–17分钟/epoch**。首次启动再加约1–2分钟的权重读取、编译和worker启动；checkpoint写盘、导出、统计与权重首次下载另计。

这是假设全部30分钟都是专家监督数据。HG-DAgger中的自主模型步、被排除区间、暂停、短于H的人工区间不计入上述窗口数，实际应使用导出的 `report.json` 的 `valid_starts`。

## 4. JAX低精度推理实测

采用官方预训练权重、相同固定噪声与合成观测，比较BF16、VLM线性层量化和VLM+action expert线性层量化。这里“全主干”不是每个算子都变成整数：Embedding、Norm、Softmax、注意力QK/AV、卷积patch embedding以及小型NNX动作投影仍为浮点。

实验直接运行 INT8×INT8→INT32 或 FP8 E4M3FN→FP32 的矩阵点积，再按scale还原。StableHLO记录了低精度输入；FP8微基准的GPU编译产物确认使用 `__cublas$lt$matmul$f8`。不是伪量化后继续用BF16点积。

**这是动态量化算子实验**：每次前向包含激活及权重量化成本，仍保存浮点权重；没有预打包权重缓存，不提供INT8/INT4 checkpoint存储或显存压缩收益。它不能代替专门部署后端的测试。

| 模式 | 去噪5步中位耗时 | 去噪10步中位耗时 |
|---|---:|---:|
| BF16 | 54.6 ms | 66.4 ms |
| VLM线性层 INT8 | 76.0 ms | 87.5 ms |
| VLM+action expert线性层 INT8 | 81.8 ms | 97.5 ms |
| VLM线性层 FP8 | 74.4 ms | 86.1 ms |
| VLM+action expert线性层 FP8 | 82.3 ms | 98.9 ms |

本次实现未获得正向加速，5步时比BF16增加约36%–51%延迟。当前部署先选 **BF16 + 5步去噪**，再通过独立任务成功率验证是否需要10步。5→10步耗时不会翻倍，因为视觉编码和语言前缀每次请求只运行一次。

相同合成观测上，5步量化输出相对BF16的归一化动作RMSE约0.036–0.048，最大绝对差约0.090–0.113。这不是关节角误差，也不是抓积木成功率，不能据此确认部署精度。

当前库固定Flax0.10.2；[Qwix当前依赖Flax≥0.12](https://github.com/google/qwix/blob/main/pyproject.toml)，本次没有升级现有训练环境。[Qwix](https://github.com/google/qwix)提供预打包PTQ等集成方向，但需单独适配和实测；当前动态量化慢并不能用于推断所有低比特部署后端的表现。

## 5. 10Hz执行与RTC

10Hz表示每100ms执行一次带时间戳的关节目标，不表示每100ms必须重新计算整个VLA。原始采集仍保存完整双路画面；推理请求可在客户端等比例letterbox到224×224，保留完整视野，减少网络传输。

两路uint8图像像素数据：224×224约0.30MB，640×480约1.84MB。端到端测量包含实际本地→SSH转发→云端WebSocket→本地路径，使用完整模型和官方预训练权重。实测记录见本报告关联JSON。

实际请求链路（含本地 `prepare_inputs`、序列化、SSH网络、服务端预处理、GPU与返回；不含真机相机采集）：

| 请求图像尺寸 | 双路payload | 中位数 | 样本P95 | 样本P99 |
|---|---:|---:|---:|---:|
| 从640×480完整画面letterbox为224×224 | 301,315字节 | 126.2 ms | 157.9 ms | 174.2 ms |
| 原始640×480 | 1,843,463字节 | 277.5 ms | 433.1 ms | 523.3 ms |

已将客户端传输前letterbox加入 `AsyncChunkPolicy`，默认224；录制器支持 `--inference-image-size 224`，`0`表示发送原分辨率。只处理传输副本，原始录制帧不变；保留左右视野的回归测试通过。此前只测预先构造的224输入，得到约104ms中位数/140ms P99；最终建议使用上表包含客户端准备的更完整结果。

224链路普通推理可先保留3个控制周期的延迟预算；完整RTC加入guidance后必须重测P99，必要时将d调到4。原分辨率链路按P99加1步余量得到d=7，H=10没有满足RTC约束的执行窗口，不宜直接采用。

推荐初始目标（不是当前代码已有的完整RTC开关）：

| 参数 | 起始设置或规则 |
|---|---|
| 控制频率 | 10Hz，Δt=100ms |
| 去噪步数 | 5，BF16 |
| 预测长度H | 10步，即1秒；长延迟时考虑训练H=20 |
| 延迟d | ceil(RTC端到端P99 / 100ms)，另留1步抖动余量 |
| 执行/重规划窗口s | 先从4步开始；d暂取3，RTC实测后必要时取4，必须满足 d≤s≤H−d |
| soft prefix schedule | exponential |
| 最大guidance权重β | 5 |
| delay buffer | 10个近期延迟样本起步，同时保留更长窗口的P99监控 |
| 停止与接管 | 暂停/人工/交回/reset立即使旧chunk与在途请求失效 |

[RTC原论文](https://www.physicalintelligence.company/download/real_time_chunking.pdf)给出 `d≤s≤H−d` 的约束，采用exponential soft masking、β=5、delay buffer=10。RTC含denoising VJP，耗时会高于本次普通采样；本报告未实测本库的完整RTC。

当前 `AsyncChunkPolicy` 是可取消的异步推理适配器，**没有旧chunk前缀inpainting/guidance，也没有提前重叠请求的完整RTC调度**。它在chunk用完后请求下一块，仍会停等；`execute_steps=1` 即使推理小于100ms也会交替请求和执行，不能等同于持续10Hz控制。修改几个CLI数值不足以实现RTC。

完整RTC需要：旧chunk尚在执行时按预计d提前发起请求；对确定会执行的前缀进行条件约束；新结果按执行时间轴对齐；每个动作只消费一次。过期判断应按动作的计划执行时间，而非直接沿用当前适配器对整个chunk的统一观测年龄限制。

若暂用当前适配器，chunk有效年龄至少覆盖 `端到端延迟 + (execute_steps−1)×100ms + 调度余量`，否则后几个动作会在执行前被丢弃。它适合接管与链路验证，不代表真实连续10Hz RTC已验收。

完整指标和估算输入见 [benchmark-results-20261018.json](benchmark-results-20261018.json)。云端26项回归测试通过；本地24项通过，2项需要OpenPI/JAX环境的测试在本地跳过、在云端通过。

## 6. 复现与证据

主要脚本位于 `scripts/benchmarks/`：

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:${PYTHONPATH:-}"

python scripts/benchmarks/train_frozen_vlm.py --batch-size 8 --steps 20 \
  --weights /root/gpufree-data/openpi-bench-cache/pi05_base_ranged \
  --dataset data/benchmarks/video640/dataset/train \
  --output data/benchmarks/train-b8-pretrained-pipeline640.json

JAX_PLATFORMS=cpu python scripts/benchmarks/test_jax_int8.py
python scripts/benchmarks/probe_native_kernels.py --output data/benchmarks/native-kernels.json

python scripts/benchmarks/inference_quantization.py \
  --weights /root/gpufree-data/openpi-bench-cache/pi05_base_ranged \
  --modes bf16 vlm all --num-steps 5 10 --iterations 40 \
  --output data/benchmarks/inference-pretrained-int8.json
```

FP8实验使用 `--modes bf16 vlm_fp8 all_fp8`。大文件和数据留在云端忽略目录；仓库只保存脚本、说明与精简结果。首次下载脚本要求目标目录独立，避免修改已有权重目录。


推理链路复现：先运行 `scripts/benchmarks/serve_latency_probe.py`（绑定云端127.0.0.1:19090，metadata显式synthetic），用SSH转发19090端口，再运行本地 `scripts/benchmarks/remote_latency_probe.py --iterations 40 --output data/benchmarks/remote-latency-with-prep.json`。该probe端点被生产接管适配器拒绝，不能用作真机策略。

关节目标执行频率与关节物理速度是不同参数。当前 `max_relative_target=5` 是标定位置单位/步的限幅，10Hz时对应最高约50标定单位/秒的目标变化，不等于50度/秒，也不是增量动作契约。本轮没有测串口、相机阻塞或真实关节/夹爪速度，需在真机阶段分别确认。
