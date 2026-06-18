# `wan_video_action.parsers` API

## 概览

`wan_video_action/parsers.py` 负责训练和推理入口的参数注册、YAML 配置合并、模型权重路径解析，以及根据模式推导数据读取键。

## 配置合并

### `merge_yaml_and_args(yaml_path, parser, args, cli_args=None)`

合并 YAML 配置和命令行参数。

优先级：

```text
CLI args > YAML config > parser defaults
```

输入：

- `yaml_path`：YAML 配置文件路径。
- `parser`：`argparse.ArgumentParser`。
- `args`：已解析出的 argparse namespace。
- `cli_args`：可选，显式传入命令行参数列表。

输出：

- 返回更新后的 `args`。

## 模型路径解析

### `_expand_safetensors_index(path)`

如果输入是 `.safetensors.index.json`，读取其中的 `weight_map`，展开成实际 shard 文件列表；否则原样返回路径。

### `resolve_model_paths(args)`

根据 model config 中的 `modules` 和 `tokenizer_subdir` 字段，解析模型权重路径和 tokenizer 路径。

输入：

- `args.model_root_path`：模型根目录。
- `args.model_config_path`：模型配置 YAML。
- `args.load_modules`：需要加载的模块名列表。

写入：

```python
args.model_paths_list
args.tokenizer_path
```

## 数据键解析

### `resolve_data_keys(args)`

根据训练/推理阶段和模型模式决定 dataset 实际读取哪些字段。

规则：

```text
stage=infer 或 vae=raw -> video
否则 -> latents

action != none -> 追加 action
text = emb -> 追加 prompt_emb, negative_prompt_emb
```

结果写入：

```python
args.data_keys
```

## 参数组注册

### `add_dataset_base_config(parser)`

注册 dataset 相关参数：

- `--dataset_name`：数据集构造器名称。
- `--dataset_base_path`：数据集根目录。
- `--dataset_metadata_path`：样本索引文件路径。
- `--dataset_repeat`：每个 epoch 重复数据集的次数。
- `--dataset_num_workers`：DataLoader worker 数量。

### `add_video_size_config(parser)`

注册视频尺寸和时间窗口参数：

- `--height`：输入帧高度。
- `--width`：输入帧宽度。
- `--max_pixels`：动态分辨率下的最大像素数。
- `--num_frames`：单个样本的视频帧数。
- `--resize_mode`：图像缩放方式，支持 `crop` 和 `fit`。
- `--num_history_frames`：历史条件帧数量。
- `--time_division_factor`：时间维对齐因子。
- `--time_division_remainder`：时间维对齐余数。
- `--spatial_division_factor`：空间尺寸对齐因子。
- `--enable_first_frame_anchor`：历史条件中是否固定加入第 0 帧。

### `add_model_config(parser)`

注册模型加载参数：

- `--model_root_path`：预训练模型根目录。
- `--model_id_with_origin_paths`：模型 ID 与原始权重路径映射。
- `--load_modules`：需要从模型配置中加载的模块名。
- `--extra_inputs`：额外传入 pipeline 的输入字段。
- `--fp8_models`：使用 FP8 精度的模型列表。
- `--offload_models`：需要 offload 的模型列表。
- `--model_config_path`：模型配置 YAML 路径。
- `--initialize_model_on_cpu`：是否先在 CPU 上初始化模型。
- `--stage`：运行阶段，用于解析数据读取键。
- `--text_mode`：文本条件模式。
- `--vae_mode`：视频 latent 来源模式。
- `--action_mode`：动作条件注入模式。

### `add_action_config(parser)`

注册 action 条件参数：

- `--action_type`：动作或状态表示类型。
- `--action_stat_path`：动作归一化统计文件路径。
- `--action_dim`：动作向量维度。

### `add_training_config(parser)`

注册训练超参数：

- `--learning_rate`：学习率。
- `--num_epochs`：训练 epoch 数。
- `--find_unused_parameters`：DDP 是否查找未使用参数。
- `--weight_decay`：权重衰减系数。
- `--task`：训练任务类型。
- `--seed`：随机种子。
- `--deterministic`：是否启用严格确定性算法。
- `--mixed_precision`：混合精度模式。
- `--max_timestep_boundary`：最大 timestep 边界。
- `--min_timestep_boundary`：最小 timestep 边界。
- `--max_train_steps`：最大 optimizer step 数。
- `--batch_size`：单卡 batch size。

### `add_output_config(parser)`

注册输出和 checkpoint 参数：

- `--output_path`：输出目录。
- `--remove_prefix_in_ckpt`：保存 checkpoint 时移除的参数名前缀。
- `--save_steps`：checkpoint 保存间隔。
- `--ckpt_path`：初始化训练权重的模型 checkpoint。
- `--resume_from`：完整训练状态恢复路径。

### `add_lora_config(parser)`

注册 LoRA 参数：

- `--lora_base_model`：挂载 LoRA 的基础模型。
- `--lora_target_modules`：应用 LoRA 的目标层。
- `--lora_rank`：LoRA rank。
- `--lora_checkpoint`：LoRA checkpoint 路径。
- `--preset_lora_path`：预置 LoRA 路径。
- `--preset_lora_model`：融合预置 LoRA 的模型名。

### `add_gradient_config(parser)`

注册梯度相关参数：

- `--use_gradient_checkpointing`：是否启用梯度检查点。
- `--use_gradient_checkpointing_offload`：是否将梯度检查点 offload 到 CPU。
- `--gradient_accumulation_steps`：梯度累积步数。
- `--max_grad_norm`：梯度裁剪上限。

### `add_tracking_config(parser)`

注册日志追踪参数：

- `--use_wandb`：是否启用 Weights & Biases。
- `--use_swanlab`：是否启用 SwanLab。
- `--run_name`：远程追踪的 run 名称。

### `add_infer_config(parser)`

注册推理参数：

- `--cfg_scale`：CFG guidance scale。
- `--num_inference_steps`：推理去噪步数。
- `--negative_prompt`：负向 prompt。
- `--negative_prompt_emb`：预提取负向 prompt embedding 路径。
- `--quality`：输出视频质量。
- `--enable_chunk_infer`：是否启用分 chunk 推理。
- `--fps`：输出视频帧率。
- `--enable_metrics`：是否启用评测指标。

### `add_debug_config(parser)`

注册调试和抽样参数：

- `--start_index`：从第几条 metadata 样本开始处理。
- `--max_samples`：最多处理多少条样本，`0` 表示不限制。

### `add_config_support(parser)`

注册通用配置文件参数：

- `--config`：YAML 配置文件路径。

### `add_general_config(parser)`

总入口。按顺序注册所有参数组：

```text
config
dataset
video
model
action
training
output
lora
gradient
tracking
infer
debug
```

训练和推理入口通常只需要调用这个函数。
