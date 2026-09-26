import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, List, Literal
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer
from cs336_alignment.grpo_train_step_standard_on_policy import grpo_train_step

PROJECT_ROOT = (Path(__file__).parent.parent).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class GRPOConfig:
    # Hugging Face 模型目录；训练模型和 vLLM 推理服务都从这里加载初始权重。
    model_path: str = str((PROJECT_ROOT / "model" / "OLMo-2-0425-1B").resolve())
    # GSM8K 训练集 JSONL，每行包含 question 和 answer。
    train_file: str = str((PROJECT_ROOT / "data" / "gsm8k" / "train.jsonl").resolve())
    # 用于验证的 JSONL；本作业使用 GSM8K 的 test.jsonl 作为验证集。
    val_file: str = str((PROJECT_ROOT / "data" / "gsm8k" / "test.jsonl").resolve())
    # 每轮训练使用的训练样例数；课程建议使用 6400 条。
    n_train_examples: int = 6400
    # 训练日志、验证记录和生成样例的输出目录。
    output_dir: str = str((PROJECT_ROOT / "outputs").resolve())
    # Hugging Face 策略模型所在的训练设备。
    train_device: str = "cuda:0"
    # vLLM 推理服务使用的 GPU 编号；与训练 GPU 分开。
    vllm_gpu: int = 1
    # 随机种子，用于控制数据抽样和训练中的随机过程。
    seed: int = 42
    # 训练步数；每步生成一批新回答并执行一次策略更新。
    n_grpo_steps: int = 200
    # AdamW 的学习率，决定每次优化器更新的步幅。
    learning_rate: float = 1e-5
    # 优势标准差归一化时加到分母上的小数，避免除以零。
    advantage_eps: float = 1e-6
    # 每个 rollout batch 的回答总数；这里包含同一题的多个采样回答。
    rollout_batch_size: int = 256
    # 每道题采样的回答数，也就是 GRPO 中每组的大小。
    group_size: int = 8
    # 训练 rollout 的采样温度；较高温度通常增加回答多样性。
    sampling_temperature: float = 1.0
    # nucleus sampling 的概率阈值；1.0 表示不通过 top-p 截断候选词。
    sampling_top_p: float = 1.0
    # 每个回答允许生成的最大 token 数。
    sampling_max_tokens: int = 512
    # 遇到这些字符串时停止生成；元组便于作为不可变默认值保存。
    sampling_stop_strings: tuple[str, ...] = ("</answer>",)
    # 每批新 rollout 重复训练的轮数；标准 on-policy GRPO 固定为 1。
    epochs_per_rollout_batch: int = 1
    # 一次优化器更新使用的回答数；标准 on-policy 设置与 rollout_batch_size 相同。
    train_batch_size: int = 256
    # 每次优化器更新前累积的 microbatch 数；256 / 32 = 每个 microbatch 8 条回答。
    gradient_accumulation_steps: int = 32
    # 梯度范数上限；超过时裁剪梯度以限制单次更新幅度。
    max_grad_norm: float = 1.0
    # Prompt 模板文件路径；其中的题目占位符会替换为 GSM8K 问题。
    prompt_path: str = str((PROJECT_ROOT / "cs336_alignment" / "prompts" / "r1_zero.prompt").resolve())
    # 组内奖励 baseline；mean 表示从每个回答奖励中减去本组平均奖励。
    baseline: Literal["mean"] = "mean"
    # 优势归一化方式；std 表示再除以本组奖励的标准差。
    advantage_normalizer: Literal["std"] = "std"
    # rollout 与当前策略的概率校正方式；none 表示标准 on-policy，不做重要性加权。
    importance_reweighting_method: Literal["none"] = "none"
    # Loss 聚合方式；sequence 表示先对每条回答的有效 token 求平均，再对回答求平均。
    loss_normalization: Literal["sequence"] = "sequence"
    # vLLM 可使用的 GPU 显存比例，包含模型权重和 KV cache。
    gpu_memory_utilization: float = 0.9
    # 每隔多少个训练步在验证集上评估一次。
    val_every_steps: int = 10
    # 每次验证使用的样例数；验证指标会基于这些样例计算。
    val_size: int = 1024
    # 验证生成的温度；0.0 表示贪心解码，便于稳定比较各步结果。
    eval_sampling_temperature: float = 0.0
    # 每个记录步保存多少条训练 rollout 示例供人工查看。
    num_example_rollouts: int = 3
    # 保存 rollout 示例的训练步数，逗号分隔；默认每 40 步记录一次。
    example_rollout_steps: str = "40,80,120,160,200"
    # WandB 项目名称，用于归集本次训练的曲线和样例。
    wandb_project: str = "cs336-assignment5-grpo"
    # WandB 单次运行名称；留空时应在启动训练时根据 seed 生成名称。
    wandb_run_name: str = ""

# Purpose: Parse command-line options and construct the GRPO configuration.
# Inputs: None explicitly; reads command-line arguments from the current process.
# Output: GRPOConfig containing one value for each training, sampling, and path option.
def parse_args() -> argparse.Namespace:
    defaults = GRPOConfig()
    parser = argparse.ArgumentParser(description="GRPO Training Configuration")
    parser.add_argument("--model_path", type=str, default=defaults.model_path, help="Path to the Hugging Face model directory.")
    parser.add_argument("--train_file", type=str, default=defaults.train_file, help="Path to the training JSONL file.")
    parser.add_argument("--val_file", type=str, default=defaults.val_file, help= "Path to the validation JSONL file.")
    parser.add_argument("--n_train_examples", type=int, default=defaults.n_train_examples, help="Number of training examples per round.")
    parser.add_argument("--output_dir", type=str, default=defaults.output_dir, help="Directory to save training logs and outputs.")
    parser.add_argument("--train_device", type=str, default=defaults.train_device, help="Device for training the Hugging Face model.")
    parser.add_argument("--vllm_gpu", type=int, default=defaults.vllm_gpu, help="GPU index for the vLLM inference server.")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Random seed for reproducibility.")
    parser.add_argument("--n_grpo_steps", type=int, default=defaults.n_grpo_steps, help="Number of GRPO training steps.")
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate, help="Learning rate for the AdamW optimizer.")
    parser.add_argument("--advantage_eps", type=float, default=defaults.advantage_eps, help="Small constant to avoid division by zero in advantage normalization.")
    parser.add_argument("--rollout_batch_size", type=int, default=defaults.rollout_batch_size, help="Total number of responses in each rollout batch.")
    parser.add_argument("--group_size", type=int, default=defaults.group_size, help="Number of responses sampled per question (group size).")
    parser.add_argument("--sampling_temperature", type=float, default=defaults.sampling_temperature, help="Sampling temperature for generating rollouts.")
    parser.add_argument("--sampling_top_p", type=float, default=defaults.sampling_top_p, help="Nucleus sampling probability threshold; 1.0 means no top-p truncation.")
    parser.add_argument("--sampling_max_tokens", type=int, default=defaults.sampling_max_tokens, help="Maximum number of tokens allowed in each generated response.")
    parser.add_argument("--sampling_stop_strings", type=str, nargs='+', default=defaults.sampling_stop_strings, help="Strings that indicate when to stop generation; multiple strings can be provided.")
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=defaults.epochs_per_rollout_batch, help="Number of epochs to train on each new rollout batch; standard on-policy GRPO is fixed at 1.")
    parser.add_argument("--train_batch_size", type=int, default=defaults.train_batch_size, help="Number of responses used in each optimizer update; standard on-policy is the same as rollout_batch_size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=defaults.gradient_accumulation_steps,help="Number of microbatches to accumulate before each optimizer update; e.g., 256 / 32 = 8 responses per microbatch.")
    parser.add_argument("--max_grad_norm", type=float, default=defaults.max_grad_norm, help="Maximum gradient norm; gradients exceeding this will be clipped to limit the update step size.")
    parser.add_argument("--prompt_path", type=str, default=defaults.prompt_path, help="Path to the prompt template file; the question placeholder will be replaced with GSM8K questions.")
    parser.add_argument("--baseline", type=str, choices=["mean"], default=defaults.baseline, help="Baseline for group rewards; 'mean' subtracts the mean reward of the group from each response reward.")
    parser.add_argument("--advantage_normalizer", type=str, choices=["std"], default=defaults.advantage_normalizer, help="Method for advantage normalization; 'std' divides by the standard deviation of the group rewards.")
    parser.add_argument("--importance_reweighting_method", type=str, choices=["none"], default=defaults.importance_reweighting_method, help="Method for importance reweighting; 'none' indicates standard on-policy GRPO without importance weighting.")
    parser.add_argument("--loss_normalization", type=str, choices=["sequence"], default=defaults.loss_normalization, help="Method for loss aggregation; 'sequence' averages over valid tokens in each response, then averages over responses.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=defaults.gpu_memory_utilization, help="Fraction of GPU memory to use for vLLM, including model weights and KV cache.")  
    parser.add_argument("--val_every_steps", type=int, default=defaults.val_every_steps, help="Number of training steps between each validation evaluation.")
    parser.add_argument("--val_size", type=int, default=defaults.val_size, help="Number of examples to use for validation; metrics will be computed based on these examples.")
    parser.add_argument("--eval_sampling_temperature", type=float, default=defaults.eval_sampling_temperature, help="Sampling temperature for validation generation; 0.0 indicates greedy decoding for stable comparison of results.")
    parser.add_argument("--num_example_rollouts", type=int, default=defaults.num_example_rollouts, help="Number of training rollout examples to save for manual inspection at each logging step.")
    parser.add_argument("--example_rollout_steps", type=str, default=defaults.example_rollout_steps, help="Comma-separated list of training steps at which to save rollout examples; default is to log every 40 steps.")
    parser.add_argument("--wandb_project", type=str, default=defaults.wandb_project, help="WandB project name for aggregating training curves and examples.")
    parser.add_argument("--wandb_run_name", type=str, default=defaults.wandb_run_name, help="WandB run name; if left empty, a name will be generated based on the seed when starting training.")
    return GRPOConfig(**vars(parser.parse_args()))

class GSM8KBatchBuilder:
    # Purpose: Store the GRPO configuration and prompt template for GSM8K batch construction.
    # Inputs: cfg (GRPOConfig); prompt_template (str) containing the "{question}" placeholder.
    # Output: None; stores both inputs on this instance.
    def __init__(self, cfg: GRPOConfig, prompt_template: str) -> None:
        self.cfg = cfg
        self.prompt_template = prompt_template

    # Purpose: Load a JSONL file and parse each line into a Python dictionary.
    # Inputs: file_path (str), the filesystem path to the JSONL file.
    # Output: list[dict], length N where N is the number of JSON records in the file.
    @staticmethod
    def load_jsonl(file_path: str) -> list[dict]:
        """Load a JSONL file and return a list of dictionaries."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    # Purpose: Convert raw GSM8K records into question and final-answer records.
    # Inputs: rows (List[dict]), length N; each dict must contain string keys "question" and "answer".
    # Output: List[dict], length N; each dict contains "question" and "ground_truth_answer" strings.
    @staticmethod
    def prepare_gsm8k(rows: List[dict]) -> List[dict]:
        examples = []
        for row in rows:
            answer = row["answer"]
            if"####"not in answer:
                raise ValueError(f"Answer does not contain '####' separator: {answer}")
            ground_truth_answer = answer.split("####")[-1].strip()
            examples.append(
                {
                    "question": row["question"],
                    "ground_truth_answer": ground_truth_answer,
                }
            )
        return examples

    # Purpose: Shuffle the selected GSM8K examples and group them into per-step prompt batches.
    # Inputs: None explicitly; reads GRPO settings and the prompt template from this instance.
    # Output: List[tuple[list[str], list[str]]], length cfg.n_grpo_steps; each tuple contains
    #         P prompts and P matching answers, where P = cfg.rollout_batch_size // cfg.group_size.
    def build_train_batches(self) -> List[tuple[list[str], list[str]]]:
        """Build training batches from the GSM8K dataset."""
        cfg = self.cfg
        prompt_template = self.prompt_template

        train_rows = self.load_jsonl(cfg.train_file)
        train_examples = self.prepare_gsm8k(train_rows)
        if len(train_examples) < cfg.n_train_examples:
            raise ValueError(f"Not enough training examples: {len(train_examples)} < {cfg.n_train_examples}")
        # Randomly sample n_train_examples from the training set
        rng = random.Random(cfg.seed)
        sampled_examples = rng.sample(train_examples, cfg.n_train_examples)
        #6400 examples, 256 rollout batch size, 8 group size => 32 prompts per step totaly 200 steps
        train_batches = []
        prompts_per_step = cfg.rollout_batch_size // cfg.group_size # our questions 256 // 8 = 32
        for idx in range(cfg.n_grpo_steps): # 200 steps
            start_idx = idx * prompts_per_step
            end_idx = start_idx + prompts_per_step
            batch_examples = sampled_examples[start_idx:end_idx]
            prompts = [prompt_template.format(question = item["question"]) for item in batch_examples]
            ground_truth_answers = [item["ground_truth_answer"] for item in batch_examples]
            train_batches.append((prompts, ground_truth_answers))
        return train_batches # [200*( 32,32 )]

# Purpose: Generate grouped rollout responses and align them with their prompts and answers.
# Inputs: server (VLLMServer), already started with weight sync initialized;
#         prompts (list[str]) of length P;
#         cfg (GRPOConfig), where group_size is G;
#         ground_truth_answers (list[str]) of length P.
# Output: tuple[list[str], list[str], list[str]]:
#         repeated prompts, responses, and repeated answers, each expected to have length P * G.
def generate_rollouts(
    server: VLLMServer,
    prompts: list[str],
    cfg: GRPOConfig,
    ground_truth_answers: list[str],
) -> tuple[list[str], list[str], list[str]]:
    if len(prompts) != len(ground_truth_answers):
        raise ValueError("prompts and answers must have equal lengths")

    sampling_params = {
        "temperature": cfg.sampling_temperature,
        "max_tokens": cfg.sampling_max_tokens,
        "n": cfg.group_size,
        "seed": cfg.seed,
        "stop": list(cfg.sampling_stop_strings),
        "include_stop_str_in_output": True,
    }

    completions = server.generate_completions(prompts, sampling_params)
    rollout_responses = [completion.text for completion in completions]

    expected = len(prompts) * cfg.group_size
    if len(rollout_responses) != expected:
        raise ValueError(
            f"Expected {expected} rollouts, got {len(rollout_responses)}"
        )

    repeated_prompts = [
        prompt for prompt in prompts for _ in range(cfg.group_size)
    ]
    repeated_ground_truths = [
        answer
        for answer in ground_truth_answers
        for _ in range(cfg.group_size)
    ]

    return repeated_prompts, rollout_responses, repeated_ground_truths


# Purpose: Run GRPO updates, periodic validation, and W&B logging with initialized resources.
# Inputs: cfg (GRPOConfig); train_batches (List of S batches, each containing P prompts and P answers);
#         reward_fn (Callable[[str, str], dict[str, float]]); server (started VLLMServer with weight sync initialized);
#         policy (torch.nn.Module); tokenizer (PreTrainedTokenizerBase); optimizer (torch.optim.Optimizer).
# Output: dict[str, Any] containing per-step training metrics, validation metrics, and selected generations.
def run_grpo_train_loop(
    cfg: GRPOConfig,
    train_batches: List[tuple[list[str], list[str]]],
    reward_fn: Callable[[str, str], dict[str, float]],
    server: VLLMServer,
    policy: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    
    # --- W&B setup: run tracking only; this does not initialize or update the policy. ---
    run_name = cfg.wandb_run_name or f"grpo-seed-{cfg.seed}"
    # W&B API: init creates one run and records its project, name, and config.
    wandb_run = wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        # asdict converts the GRPOConfig dataclass into a dictionary for run metadata.
        config=asdict(cfg),
    )

    history: dict[str, Any] = {"train": [], "val": [], "examples": []}
    example_rollout_steps = {
        int(value.strip())
        for value in cfg.example_rollout_steps.split(",")
        if value.strip()
    }

    try:
        prompt_template = Path(cfg.prompt_path).read_text(encoding="utf-8")
        val_rows = GSM8KBatchBuilder.load_jsonl(cfg.val_file)
        val_examples = GSM8KBatchBuilder.prepare_gsm8k(val_rows)
        if cfg.val_size > 0:
            val_examples = val_examples[:cfg.val_size]
        if not val_examples:
            raise ValueError(f"No validation examples found in {cfg.val_file}")

        validation_prompts = [
            prompt_template.format(question=item["question"])
            for item in val_examples
        ]
        validation_ground_truths = [
            item["ground_truth_answer"]
            for item in val_examples
        ]

        # --- Training loop: rollout generation and policy updates. ---
        for step, (prompts, ground_truth_answers) in enumerate(train_batches, start=1):
            # Use the current policy to sample fresh on-policy rollouts.
            server.sync_policy_weights(policy)
            
            (
                repeated_prompts,
                rollout_responses,
                repeated_ground_truths,
            ) = generate_rollouts(
                server,
                prompts,
                cfg,
                ground_truth_answers,
            )

            # Training API: computes rewards/loss, accumulates gradients, and updates policy once.
            loss, metadata = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                max_grad_norm=cfg.max_grad_norm,
                reward_fn=reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=cfg.group_size,
                baseline=cfg.baseline,
                advantage_eps=cfg.advantage_eps,
                advantage_normalizer=cfg.advantage_normalizer,
                importance_reweighting_method=cfg.importance_reweighting_method,
                loss_normalization=cfg.loss_normalization,
            )

            # --- W&B logging preparation only: convert results to serializable scalars. ---
            step_metrics: dict[str, float] = {
                "loss": float(loss.detach().float().cpu().item()),
            }
            for key, value in metadata.items():
                if isinstance(value, torch.Tensor):
                    step_metrics[key.replace("-", "_")] = float(
                        value.detach().float().cpu().item()
                    )
                elif isinstance(value, (int, float)):
                    step_metrics[key.replace("-", "_")] = float(value)

            log_data = {
                f"train/{key}": value
                for key, value in step_metrics.items()
            }
            history["train"].append({"step": step, **step_metrics})

            # W&B-only logging: collect selected generations for inspection.
            if step in example_rollout_steps:
                # W&B API: Table creates a tabular log object; add_data appends rows below.
                train_table = wandb.Table(
                    columns=[
                        "step",
                        "prompt",
                        "response",
                        "reward",
                        "format_reward",
                        "answer_reward",
                    ]
                )
                num_examples = min(cfg.num_example_rollouts, len(prompts))
                for group_idx in range(num_examples):
                    response_idx = group_idx * cfg.group_size
                    scores = reward_fn(
                        rollout_responses[response_idx],
                        repeated_ground_truths[response_idx],
                    )
                    example = {
                        "step": step,
                        "prompt": repeated_prompts[response_idx],
                        "response": rollout_responses[response_idx],
                        "reward": float(scores["reward"]),
                        "format_reward": float(scores["format_reward"]),
                        "answer_reward": float(scores["answer_reward"]),
                    }
                    history["examples"].append(example)
                    # W&B API: add_data appends one generation record to the table.
                    train_table.add_data(
                        example["step"],
                        example["prompt"],
                        example["response"],
                        example["reward"],
                        example["format_reward"],
                        example["answer_reward"],
                    )
                log_data["train/generations"] = train_table

            # --- Validation only: evaluate the updated policy without optimizer updates. ---
            if step % cfg.val_every_steps == 0 or step == len(train_batches):
                # Evaluate the updated policy, and log metrics and sample generations.
                server.sync_policy_weights(policy)
                val_sampling_params = {
                    "temperature": cfg.eval_sampling_temperature,
                    "max_tokens": cfg.sampling_max_tokens,
                    "n": 1,
                    "seed": cfg.seed,
                    "stop": list(cfg.sampling_stop_strings),
                    "include_stop_str_in_output": True,
                }
                val_completions = server.generate_completions(
                    validation_prompts,
                    val_sampling_params,
                    batch_size=cfg.rollout_batch_size,
                )
                if len(val_completions) != len(val_examples):
                    raise ValueError(
                        f"Expected {len(val_examples)} validation responses, "
                        f"got {len(val_completions)}"
                    )

                val_scores = [
                    reward_fn(completion.text, ground_truth)
                    for completion, ground_truth in zip(
                        val_completions,
                        validation_ground_truths,
                        strict=True,
                    )
                ]
                val_metrics = {
                    "step": step,
                    "mean_reward": sum(score["reward"] for score in val_scores) / len(val_scores),
                    "mean_format_reward": sum(score["format_reward"] for score in val_scores) / len(val_scores),
                    "mean_answer_reward": sum(score["answer_reward"] for score in val_scores) / len(val_scores),
                }
                history["val"].append(val_metrics)
                log_data.update(
                    {
                        f"validation/{key}": value
                        for key, value in val_metrics.items()
                        if key != "step"
                    }
                )

                # W&B API: Table creates the validation-generation log object.
                val_table = wandb.Table(
                    columns=[
                        "step",
                        "prompt",
                        "response",
                        "reward",
                        "format_reward",
                        "answer_reward",
                    ]
                )
                num_val_examples = min(
                    cfg.num_example_rollouts,
                    len(val_examples),
                )
                for idx in range(num_val_examples):
                    val_example = {
                        "step": step,
                        "prompt": validation_prompts[idx],
                        "response": val_completions[idx].text,
                        "reward": float(val_scores[idx]["reward"]),
                        "format_reward": float(val_scores[idx]["format_reward"]),
                        "answer_reward": float(val_scores[idx]["answer_reward"]),
                    }
                    history["examples"].append(val_example)
                    # W&B API: add_data appends one validation example to the table.
                    val_table.add_data(
                        val_example["step"],
                        val_example["prompt"],
                        val_example["response"],
                        val_example["reward"],
                        val_example["format_reward"],
                        val_example["answer_reward"],
                    )
                log_data["validation/generations"] = val_table

            # W&B API: log uploads this step's metrics and any tables in log_data.
            wandb_run.log(log_data, step=step)

        return history
    finally:
        # W&B API: finish flushes pending records and closes the run.
        wandb_run.finish()


# Purpose: Parse configuration, load the prompt template, prepare batches, and iterate over them.
# Inputs: None explicitly; obtains all settings from command-line arguments.
# Output: None; the current per-step training body is still a placeholder.
def main():
    cfg = parse_args()
    # Load prompt template
    with open(cfg.prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    gsm8k_batch_builder = GSM8KBatchBuilder(cfg, prompt_template)
    train_batches = gsm8k_batch_builder.build_train_batches()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code = True)
    
    # Initialize vLLM server and Hugging Face model here (not shown)
    # For each training step, generate rollouts, compute rewards, and update the policy
    server = VLLMServer(
        model_id = cfg.model_path,
        host = "127.0.0.1",
        port = 8000,
        gpu = cfg.vllm_gpu,
        seed = cfg.seed,
        gpu_memory_utilization = cfg.gpu_memory_utilization,
    )
    
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype = torch.bfloat16,
        trust_remote_code = True
    ).to(cfg.train_device)
    
    policy.train()
    
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr = cfg.learning_rate,
    )
    
    try:
        server.start()
        server.init_weight_sync(
            policy_device = cfg.train_device
        )
        history = run_grpo_train_loop(
            cfg,
            train_batches,
            r1_zero_reward_fn,
            server,
            policy,
            optimizer = optimizer,
            tokenizer = tokenizer
        )
    finally:
        server.stop()
if __name__ = "__main__":
    main()