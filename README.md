# 脚本使用说明

以下命令均在仓库根目录执行，使用已安装对应依赖的 Python 环境。采集端使用机器人环境，训练与推理服务使用 OpenPI 环境。

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:${PYTHONPATH:-}"
```

命令中的 `<...>` 请替换为实际路径或设备参数；脚本的完整参数可通过 `--help` 查看。

## SO101 采集与人工接管

默认离线预检，不打开机器人或相机：

```bash
python examples/so101/record_openpi_so101.py
```

使用主从臂与双路相机录制人工示范：

```bash
python examples/so101/record_openpi_so101.py --hardware \
  --follower-port <从臂串口> --leader-port <主臂串口> \
  --main-camera <主相机设备> --wrist-camera <腕相机设备> \
  --out data/recordings
```

当前采集任务为 `pick blocks from table`，记录两路完整视频、六维实测位置和六维绝对目标位置。按 `C` 开始，`D` 保存，`Q` 排除当前轨迹但保留原始记录，`X` 退出。策略运行时，第一次空格暂停，第二次空格进入人工接管；人工模式下按空格交回策略。

接入策略服务时，在上面的采集命令后追加：

```bash
--policy-uri ws://127.0.0.1:8000 --inference-image-size 224 \
--execute-steps 1 --timeout-s 1.0 --max-age-s 0.3
```

`--inference-image-size` 控制发送给策略的图像尺寸，设为 `0` 发送原图；录制视频保留原始分辨率。`--max-takeovers` 限制接管次数，默认 `0` 表示不限。

## 专家区间复核与数据导出

检查录制文件，打开双路视频复核界面：

```bash
python -m openpi.so101.export inspect data/recordings/<批次>/*.pkl
python examples/so101/review_experts.py data/recordings/<批次>/<轨迹>.pkl \
  --reviewer <复核人>
```

在复核界面标记人工示范或人工纠正区间并保存，生成同名 `.expert.json`。也可以先生成待复核模板：

```bash
python -m openpi.so101.export review-template data/recordings/<批次>/*.pkl
```

导出已通过复核的专家区间；`--output` 必须是尚不存在的目录。提供多个独立采集批次，以便划分训练集和验证集。

```bash
python -m openpi.so101.export export \
  data/recordings/<批次1>/*.pkl data/recordings/<批次2>/*.pkl \
  --output data/blocks_v001 --horizon 10
```

## 冻结 VLM 训练

在训练环境执行：

```bash
python scripts/so101/train.py \
  --dataset data/blocks_v001/train \
  --exp-name blocks_frozen_vlm \
  --weights <pi05_base的params目录> \
  --freeze-vlm --no-lora \
  --batch-size 8 --num-workers 4 --steps 20000
```

`--freeze-vlm` 冻结完整视觉语言模型，`--no-lora` 使用普通参数训练动作专家。仅计算归一化统计时追加 `--compute-norm-only`；通过 `--checkpoint-root` 指定检查点根目录。

## 启动推理服务

使用训练时绑定的同一数据集：

```bash
python scripts/so101/serve.py \
  --dataset data/blocks_v001/train \
  --checkpoint checkpoints/pi05_so101_absolute6/blocks_frozen_vlm/<step> \
  --num-steps 5 --bind 127.0.0.1 --port 8000
```

`--num-steps` 为去噪步数。远程部署时，在采集端建立 SSH 转发，再使用前面的 `--policy-uri` 参数：

```bash
ssh -N -L 8000:127.0.0.1:8000 -p <SSH端口> <用户名>@<服务器地址>
```

## 离线验收

```bash
python -m unittest discover -s tests/so101 -v
python scripts/so101/offline_acceptance.py --output data/acceptance-new
```

验收输出目录必须尚不存在。在完整 OpenPI GPU 环境中，可追加 `--model-smoke` 执行模型冒烟检查。

## 性能与量化测试

下载测试所需 JAX 权重，输出目录必须尚不存在：

```bash
python scripts/benchmarks/fetch_jax_weights.py \
  --output <权重目录> --workers 16
```

准备合成双路视频数据；`--norm-source` 指向包含 `norm_stats.json` 的目录：

```bash
python scripts/benchmarks/prepare_video_fixture.py \
  --output data/benchmarks/video-fixture \
  --norm-source <归一化统计目录>
```

测试冻结 VLM 的单卡训练与数据加载速度：

```bash
python scripts/benchmarks/train_frozen_vlm.py \
  --weights <JAX权重目录> \
  --dataset data/benchmarks/video-fixture/dataset/train \
  --batch-size 8 --steps 20 \
  --output data/benchmarks/train.json
```

测试 JAX INT8 运算、GPU 原生矩阵运算和模型量化推理：

```bash
JAX_PLATFORMS=cpu python scripts/benchmarks/test_jax_int8.py
python scripts/benchmarks/probe_native_kernels.py \
  --output data/benchmarks/native-kernels.json
python scripts/benchmarks/inference_quantization.py \
  --weights <JAX权重目录> \
  --modes bf16 vlm all vlm_fp8 all_fp8 \
  --num-steps 5 10 --iterations 40 \
  --output data/benchmarks/inference.json
```

`bf16` 为基线；`vlm`、`all` 分别测试 VLM 与全部受支持线性层的动态 INT8 运算，`vlm_fp8`、`all_fp8` 使用对应范围的 FP8 运算。这些模式用于性能实验。

测试端到端通信延迟，在服务端运行：

```bash
python scripts/benchmarks/serve_latency_probe.py \
  --weights <JAX权重目录> --num-steps 5 --port 19090
```

在客户端建立转发并运行探针：

```bash
ssh -N -L 19090:127.0.0.1:19090 -p <SSH端口> <用户名>@<服务器地址>
# 另开终端执行：
python scripts/benchmarks/remote_latency_probe.py \
  --uri ws://127.0.0.1:19090 --iterations 40 \
  --output data/benchmarks/remote-latency-with-prep.json
```

该延迟服务仅接收合成测试输入。
