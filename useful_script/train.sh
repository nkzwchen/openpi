# 设置 JAX 内存分配（推荐）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 基础训练命令
uv run scripts/train.py <config_name> --exp_name <实验名称>

# 示例：训练 Pi0 模型
uv run scripts/train.py pi0_aloha_sim --exp_name my_pi0_experiment

# 多GPU 训练（使用 FSDP）
uv run scripts/train.py <config_name> --exp_name <实验名称> --fsdp-devices <GPU数量>

# 示例：使用 4 张 GPU 训练
uv run scripts/train.py pi0_aloha_sim --exp_name my_experiment --fsdp-devices 4