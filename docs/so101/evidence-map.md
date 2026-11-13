# SO101 扩展：实现与证据对应

核对日期：2026-11-13。范围是本分支公开的代码和历史实验文件，不以对话描述代替实验。

| 工作 | 可检查位置 | 证据边界 |
|---|---|---|
| 主从臂采集与接管 | [入口](../../examples/so101/record_openpi_so101.py)、[状态机](../../src/openpi/so101/collector.py) | LeRobot 原生硬件 IO；不能把这一路描述为 ROS 串口控制 |
| 人工纠正回流 | [区间复核](../../examples/so101/review_experts.py)、[导出](../../src/openpi/so101/export.py) | 专家区间审核与离线聚合；不是每步自动在线学习 |
| π0.5 适配 | [训练](../../scripts/so101/train.py)、[模型](../../src/openpi/models/pi0.py) | 六维有效动作及数据契约；完整训练效果需要微调权重和评测 |
| 异步动作块 | [runtime](../../src/openpi/so101/runtime.py) | 实现响应时效与切换失效管理；RTC 专门方法尚无对应证据 |
| FlashRT / FP8 | [性能说明](performance.md)、[数值文件](../../evidence/fp8-real-20261018/summary.json) | L40S 基础模型离线基准，不证明真机量化精度 |
| 成功率 80%→90%、新增 30 条纠正 | 待补独立评测表及数据版本 | 2026-11-13 用户确认暂时没有可提供的记录，不作为公开验证结论 |

提交时间记录代码演进，不自动等同于项目实际起止时间。复现先从 README 的无硬件检查开始。

## 后续评测需要保存什么

每个 trial 记录 `trial_id, checkpoint, dataset_version, scene, seed, success, intervention_count, duration_s, video`；明确抓取成功、任务完成时限和是否允许接管。按位置、干扰物划分测试场景，保持比较条件和试验次数一致，分别报告纯策略与人工协助结果。新增样本的采集批次不得泄漏到评测集。

RTC 需要提供实际算法实现和同权重、同延迟条件下的对照；可另开发，但不能将新实现追溯为已有结果。
