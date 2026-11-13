# π0.5 推理性能：真实观测离线对照

实验日期 2026-10-18；2026-11-13 公开既有记录并做 CPU 数值复核，未重新运行 GPU。证据：[FP8 原始输出与汇总](../../evidence/fp8-real-20261018/summary.json)、[provenance](../../evidence/fp8-real-20261018/provenance.json)、[复核脚本](../../scripts/benchmarks/verify_published_results.py)。公开导出仅替换机器绝对目录，数值和警告保留，文件哈希见 `evidence/manifest.json`。

## 条件与结果

L40S；π0.5 基础权重；双路 224×224 图像；horizon 10、6 维有效动作；固定输入与噪声比较输出。校准和评估各 64 组真实观测，episode 不重叠。每配置预热 100 次、3 轮各 200 次；重复计时次数不是独立任务试验次数。

| 后端 | 去噪步数 | 中位 ms | P99 ms |
|---|---:|---:|---:|
| JAX BF16 | 5 | 48.48 | 50.52 |
| FlashRT BF16 | 5 | 27.21 | 28.50 |
| FlashRT VLM FP8＋专家 BF16 | 5 | 21.07 | 24.42 |
| JAX BF16 | 10 | 60.82 | 63.45 |
| FlashRT BF16 | 10 | 34.11 | 35.26 |
| FlashRT VLM FP8＋专家 BF16 | 10 | 28.46 | 28.98 |

FlashRT 计时从 host 图像/状态开始，包含提示词更新、图像准备及 H2D，直到同步得到 host 动作；JAX 从准备好的设备图像及 host 状态开始。均排除初次加载/编译。具体边界见每个文件 `metadata.timing_scope`。因此跨后端加速是此测试方法下的整体比较，不能全部归功于 FP8，也不包含公网传输或机械臂执行。

相对 JAX，FP8 配置的归一化动作 RMSE 约 0.044 / 0.059，最大偏差约 0.337 / 0.545（5 / 10 步）。动作误差不是成功率。校准存在异常尺度层警告；使用校准侧替代状态归一化，未绑定任务微调 checkpoint；不能宣称已经验证量化后的抓取精度。

## 为什么动态 INT8 未必更快

[算子结果](../../evidence/int8-20261018/results.json) 比较 `(968×2048) @ (2048×16384)`：BF16 约 0.361 ms，纯 INT8 GEMM 约 0.329 ms，预量化权重但包含激活量化/还原约 0.398 ms，动态量化完整流程约 0.550 ms。完整流程每次求尺度并量化，额外开销抵消矩阵乘法收益。HLO 保存在同目录，不能将该结论推广到所有 INT8 内核或完整模型。

## 复核与重跑

CPU 复核现有文件的哈希、样本一致性、RMSE、加速比例：

```bash
python scripts/benchmarks/verify_published_results.py
```

GPU 重跑使用 [命令指南](commands.md) 中 `prepare_real_fixture.py` 和 `fp8_runtime.py`。需要相同权重、数据源、FlashRT 提交、GPU 与归一化；未提供源视频/模型时，只能复核已公开数值，不能完整重现推理。不得用随机输入重跑值替换本表的真实观测结果。

图表可用 `python scripts/benchmarks/plot_published_latency.py` 重建，需要额外安装 matplotlib。
