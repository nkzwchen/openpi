# OpenPI Training Data Flow Learning Notes

这份笔记用于从训练命令出发，顺着 OpenPI 的源码理解一次训练的数据流。推荐阅读路径是：

```text
useful_script/train.sh
  -> scripts/train.py
  -> src/openpi/training/config.py
  -> src/openpi/training/data_loader.py
  -> src/openpi/transforms.py
  -> src/openpi/policies/*_policy.py
  -> src/openpi/models/model.py
  -> src/openpi/models/pi0.py
```

## 1. 训练命令在做什么

`useful_script/train.sh` 中的基础命令是：

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py <config_name> --exp_name <实验名称>
```

示例：

```bash
uv run scripts/train.py pi0_aloha_sim --exp_name my_pi0_experiment
```

多 GPU 示例：

```bash
uv run scripts/train.py pi0_aloha_sim --exp_name my_experiment --fsdp-devices 4
```

各部分含义：

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`：让 JAX/XLA 最多预分配 90% GPU 显存，训练大模型时通常更稳定。
- `uv run`：用项目环境运行 Python 脚本。
- `scripts/train.py`：JAX 版训练入口。
- `<config_name>`：训练配置名，例如 `pi0_aloha_sim`、`pi05_so101`。
- `--exp_name` / `--exp-name`：实验名，用于 checkpoint 路径和 wandb run 名称。
- `--fsdp-devices`：使用多少张 GPU 做 FSDP 参数分片。

在源码里，`scripts/train.py` 最后是：

```python
if __name__ == "__main__":
    main(_config.cli())
```

也就是说命令行参数会被 `tyro` 解析成一个 `TrainConfig`，然后传给 `main()`。

## 2. TrainConfig 决定训练什么

核心配置类在：

```text
src/openpi/training/config.py
```

`TrainConfig` 里最重要的字段：

```python
name                 # config 名字，例如 pi0_aloha_sim
model                # 模型配置，例如 Pi0Config / Pi0FASTConfig
data                 # 数据配置工厂，例如 LeRobotAlohaDataConfig
weight_loader        # 从哪里加载预训练权重
batch_size           # 全局 batch size
num_train_steps      # 训练步数
lr_schedule          # 学习率计划
optimizer            # 优化器
fsdp_devices         # FSDP 分片设备数
```

以 `pi0_aloha_sim` 为例，它表示：

```text
模型：Pi0Config()
数据：lerobot/aloha_sim_transfer_cube_human
prompt：Transfer cube
初始权重：gs://openpi-assets/checkpoints/pi0_base/params
训练步数：20000
```

你当前代码里还加了 `pi05_so101`，它的意图大概是：

```text
模型：pi0.5
数据：/data/wwq/lerobot_smolvla/datasets/pick_place_pencil_smolvla
batch_size：2
初始权重：gs://openpi-assets/checkpoints/pi05_base/params
```

注意：当前 `config.py` 末尾重新写了一次 `_CONFIGS = [...]`，但前面的 `_CONFIGS_DICT` 已经生成过了。这可能导致新加的 `pi05_so101` 不会被 CLI 识别。训练前建议把新 config 合并到原来的 `_CONFIGS` 列表中，或者在重定义 `_CONFIGS` 后重新生成 `_CONFIGS_DICT`。

## 3. main() 是训练总控

入口函数在：

```text
scripts/train.py
```

主流程可以概括成：

```text
main(config)
  1. 初始化日志
  2. 检查 batch_size 是否能被 GPU 数量整除
  3. 创建 JAX mesh / FSDP sharding
  4. 初始化 checkpoint 目录
  5. 初始化 wandb
  6. 创建 data_loader
  7. 取第一个 batch
  8. 初始化模型和训练状态
  9. jax.jit 包装 train_step
  10. for step in range(num_train_steps):
        train_step
        log
        next batch
        save checkpoint
```

最关键的两行是：

```python
data_loader = _data_loader.create_data_loader(
    config,
    sharding=data_sharding,
    shuffle=True,
)
batch = next(data_iter)
```

这就是训练数据真正进入训练循环的位置。

## 4. 数据加载入口：create_data_loader

数据加载代码在：

```text
src/openpi/training/data_loader.py
```

主函数：

```python
def create_data_loader(config, ...):
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(...)
    return create_torch_data_loader(...)
```

这里的关键点是：`TrainConfig.data` 不是最终的数据配置，而是一个 `DataConfigFactory`。它会根据 `assets_dirs` 和 `model_config` 创建真正的 `DataConfig`。

`DataConfig` 里最重要的字段：

```python
repo_id              # LeRobot 数据集路径或 HuggingFace repo
norm_stats           # 归一化统计
repack_transforms    # 原始字段名 -> openpi 中间字段名
data_transforms      # 机器人相关变换
model_transforms     # 模型相关变换
action_sequence_keys # 动作序列字段名
prompt_from_task     # 是否从 LeRobot task 里拿 prompt
```

## 5. LeRobot 数据如何被读取

普通 LeRobot 数据会走：

```text
create_data_loader
  -> create_torch_data_loader
  -> create_torch_dataset
```

`create_torch_dataset()` 里会创建：

```python
dataset_meta = LeRobotDatasetMetadata(repo_id)

dataset = LeRobotDataset(
    data_config.repo_id,
    delta_timestamps={
        key: [t / dataset_meta.fps for t in range(action_horizon)]
        for key in data_config.action_sequence_keys
    },
)
```

这里有一个很重要的概念：`action_horizon` 决定每个训练样本中包含多少步未来动作。

例如 `Pi0Config` 默认：

```python
action_horizon = 50
action_dim = 32
```

那么训练时的动作标签大概是：

```text
actions: [batch, 50, 32]
```

如果 SO101 配置设置：

```python
action_horizon = 10
```

那么动作标签大概是：

```text
actions: [batch, 10, action_dim]
```

原始 SO101 动作可能只有 6 或 7 维，但后面会通过 `PadStatesAndActions` pad 到模型需要的 `action_dim`。

## 6. Transform 是数据流核心

数据集读出来后，会进入 `transform_dataset()`：

```python
return TransformedDataset(
    dataset,
    [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ],
)
```

这串 transform 是数据流里最重要的一段：

```text
原始 LeRobot sample
  -> repack_transforms
  -> data_transforms
  -> Normalize
  -> model_transforms
  -> 模型需要的 dict
```

主要源码在：

```text
src/openpi/transforms.py
```

## 7. repack_transforms：字段重命名

`RepackTransform` 的作用是把数据集原始字段改成 OpenPI 期待的中间字段。

例如 SO101 配置里类似：

```python
RepackTransform(
    {
        "observation/image": "observation.images.top",
        "observation/wrist_image": "observation.images.xie",
        "observation/state": "observation.state",
        "actions": "action",
        "prompt": "task_index",
    }
)
```

含义是：

```text
原始 observation.images.top   -> 新字段 observation/image
原始 observation.images.xie   -> 新字段 observation/wrist_image
原始 observation.state        -> 新字段 observation/state
原始 action                   -> 新字段 actions
原始 task_index               -> 新字段 prompt
```

这一步只负责改字段结构，不负责模型语义。

## 8. data_transforms：机器人格式转模型通用格式

以 SO101 为例，相关代码在：

```text
src/openpi/policies/so101_policy.py
```

`SO101Inputs` 会把 SO101 数据整理成 OpenPI 模型通用结构：

```python
{
    "state": ...,
    "image": {
        "base_0_rgb": ...,
        "left_wrist_0_rgb": ...,
        "right_wrist_0_rgb": ...,
    },
    "image_mask": {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": True,
    },
    "actions": ...,
    "prompt": ...,
}
```

这个结构非常关键，因为后面的 `Observation.from_dict()` 就认这个格式。

你还加了 `PromptIndexToText`，它在：

```text
src/openpi/transforms.py
```

作用是把数字任务 ID 变成文本 prompt：

```text
0 -> "Place the pencil in the box."
```

也就是：

```text
task_index: 0
  -> prompt: "Place the pencil in the box."
```

## 9. Normalize：状态和动作归一化

接下来会经过：

```python
Normalize(norm_stats, use_quantiles=...)
```

真实数据训练时，如果没有归一化统计，会报错：

```text
Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`.
```

所以正常训练前要先跑：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_so101
```

归一化通常作用在：

```text
state
actions
```

图片不会用这个 norm stats 归一化。图片会在 `Observation.from_dict()` 中从 `uint8 [0,255]` 转成 `float [-1,1]`。

## 10. model_transforms：变成模型真正可吃的输入

`ModelTransformFactory` 在：

```text
src/openpi/training/config.py
```

对于 Pi0 / Pi0.5，通常会做：

```python
InjectDefaultPrompt(...)
ResizeImages(224, 224)
TokenizePrompt(...)
PadStatesAndActions(model_config.action_dim)
```

也就是：

```text
补默认 prompt
图片 resize 到 224x224
文本 prompt tokenize
state/actions pad 到 action_dim
```

最终单个样本大概长这样：

```python
{
    "image": {
        "base_0_rgb": [224, 224, 3],
        "left_wrist_0_rgb": [224, 224, 3],
        "right_wrist_0_rgb": [224, 224, 3],
    },
    "image_mask": {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": True,
    },
    "state": [action_dim],
    "actions": [action_horizon, action_dim],
    "tokenized_prompt": [max_token_len],
    "tokenized_prompt_mask": [max_token_len],
}
```

## 11. batch 和 sharding

单个样本 transform 完以后，会进入 `TorchDataLoader`。

`_collate_fn()` 做的事情是：

```python
np.stack(..., axis=0)
```

也就是把多个单样本堆成 batch：

```text
image:   [224, 224, 3]          -> [batch, 224, 224, 3]
state:   [action_dim]           -> [batch, action_dim]
actions: [horizon, action_dim]  -> [batch, horizon, action_dim]
prompt:  [max_token_len]        -> [batch, max_token_len]
```

如果是 JAX 训练，还会执行：

```python
jax.make_array_from_process_local_data(self._sharding, x)
```

这一步把 numpy batch 变成带 sharding 的 JAX array。

## 12. dict 转 Observation

最后在 `DataLoaderImpl.__iter__()`：

```python
yield _model.Observation.from_dict(batch), batch["actions"]
```

也就是说，训练循环拿到的 batch 是：

```python
(
    Observation(...),
    actions
)
```

`Observation` 定义在：

```text
src/openpi/models/model.py
```

里面包含：

```python
images
image_masks
state
tokenized_prompt
tokenized_prompt_mask
token_ar_mask        # pi0-fast 用
token_loss_mask      # pi0-fast 用
```

`actions` 单独作为 label 传给模型。

## 13. train_step：真正训练一步

回到 `scripts/train.py`：

```python
observation, actions = batch
loss, grads = nnx.value_and_grad(loss_fn)(model, train_rng, observation, actions)
```

`loss_fn` 里最核心的是：

```python
chunked_loss = model.compute_loss(rng, observation, actions, train=True)
```

所以数据最终进入模型的位置是：

```text
model.compute_loss(rng, observation, actions, train=True)
```

对于 Pi0，代码在：

```text
src/openpi/models/pi0.py
```

大概流程是：

```text
observation/actions
  -> preprocess_observation
       图片增强
       检查三路 image key
       mask 整理
  -> 给 actions 加 noise
  -> 采样 flow matching 时间 t
  -> embed_prefix(observation)
       图像 + prompt + state
  -> embed_suffix(observation, noisy_actions, time)
       noisy action tokens
  -> PaliGemma forward
  -> action_out_proj
  -> MSE(v_t, u_t)
```

这个模型不是普通的“直接预测 action”，而是 flow matching / diffusion 风格：训练模型预测从 noisy action 回到真实 action 的向量场。

## 14. 一句话总数据流

完整训练数据流可以记成：

```text
train.sh
  -> scripts/train.py main()
  -> TrainConfig
  -> config.data.create()
  -> DataConfig
  -> LeRobotDataset / RLDS dataset
  -> RepackTransform
  -> robot-specific Inputs, e.g. AlohaInputs / SO101Inputs
  -> Normalize
  -> ResizeImages + TokenizePrompt + PadStatesAndActions
  -> TorchDataLoader collate
  -> JAX sharded batch
  -> Observation.from_dict(batch), actions
  -> train_step
  -> model.compute_loss
  -> grads
  -> optimizer update
  -> checkpoint / wandb log
```

## 15. 推荐源码阅读顺序

建议按这个顺序读：

1. `useful_script/train.sh`
   先知道命令怎么进入训练脚本。

2. `scripts/train.py`
   重点看 `main()` 和 `train_step()`。

3. `src/openpi/training/config.py`
   重点看 `TrainConfig`、`DataConfig`、`ModelTransformFactory`、你的 `LeRobotSO101DataConfig`。

4. `src/openpi/training/data_loader.py`
   重点看 `create_data_loader()`、`create_torch_dataset()`、`transform_dataset()`、`DataLoaderImpl.__iter__()`。

5. `src/openpi/transforms.py`
   重点看 `RepackTransform`、`Normalize`、`TokenizePrompt`、`PadStatesAndActions`、`DeltaActions`。

6. `src/openpi/policies/so101_policy.py`
   看你的机器人数据怎么被整理成 OpenPI 标准输入。

7. `src/openpi/models/model.py`
   看 `Observation` 的标准格式。

8. `src/openpi/models/pi0.py`
   看 `compute_loss()`，理解模型怎么消费 batch。

## 16. 当前 SO101 配置的注意点

你的 `SO101Inputs` 里读取了：

```python
data["observation/right_image"]
```

但当前 `LeRobotSO101DataConfig` 的 `RepackTransform` 里没有映射这个字段。也就是说，如果原始数据没有被 repack 出 `observation/right_image`，训练时会在这里 KeyError。

建议重点检查：

```text
LeRobotSO101DataConfig.repack_transform
  是否产出：
    observation/image
    observation/wrist_image
    observation/right_image
    observation/state
    actions
    prompt
```

另外，当前 `config.py` 中存在一个需要留意的结构问题：

```text
原始 _CONFIGS
  -> 生成 _CONFIGS_DICT
  -> 后面又重新定义 _CONFIGS
```

如果要让 `pi05_so101` 被 CLI 正常识别，需要让它出现在生成 `_CONFIGS_DICT` 之前，或者在最后重新执行：

```python
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}
```

否则命令：

```bash
uv run scripts/train.py pi05_so101 --exp-name so101_test
```

可能找不到这个 config。
