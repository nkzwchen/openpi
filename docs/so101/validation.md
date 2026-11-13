# 本次离线验证（2026-11-13）

使用现有 Python 3.12 环境、CPU 软件路径；未连接机器人、相机或云端 GPU。

| 检查 | 结果 |
|---|---|
| `unittest discover -s tests/so101 -v` | 26 项，24通过、2明确跳过 |
| `offline_acceptance.py` | 合成数据导出通过，模型冒烟未运行 |
| `verify_published_results.py` | 哈希、历史延迟数值、加速比、episode 划分、跨后端输入一致性及动作误差复算通过 |

跳过的是 WebSocket 回环（该环境未安装 websockets）和完整 serving 配置测试（缺少 augmax 等 OpenPI 模型依赖）；不能将其写为已通过。本次修正了 serving 测试只凭安装 JAX 就认定依赖齐备的判定。

轻量独立环境安装尝试受 PyPI/镜像 TLS 连接故障阻断，未关闭证书验证。README 中的 requirements 入口已提供，但本次不声称完成了全新环境安装验收。

测试覆盖专家区间边界、异步响应失效、人工接管状态机、写盘快照、数据导出与冻结参数筛选；这些不代表模型已学会任务。历史 GPU 实验仍标注2026-10-18，本次只重新计算既有结果。

复核命令见 README。`evidence/offline-validation-20261113/` 保留脱敏测试输出和合成导出报告，原始本地路径替换为 `<WORKSPACE>`；合成生成物不能用于正式真机训练。
