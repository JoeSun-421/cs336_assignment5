#!/usr/bin/env python3
"""Evaluate the three Assignment 5 GSM8K prompting baselines.

This script assumes that a vLLM OpenAI-compatible server is already running
at the configured URL. It saves every prompt, response, and reward so that
the automatically misparsed examples can be inspected later.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from cs336_alignment.vllm_utils import generate_completions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "allenai/OLMo-2-0425-1B"
DEFAULT_DATA_PATH = ROOT / "data/gsm8k/test.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "outputs/prompting_baselines.jsonl"

CATEGORY_ORDER = [
    "category_1_correct_and_formatted",
    "category_2_formatted_but_wrong",
    "category_3_unformatted",
    "unexpected_format_zero_answer_one",
]
CATEGORY_LABELS = {
    "category_1_correct_and_formatted": "1: correct + formatted",
    "category_2_formatted_but_wrong": "2: formatted, answer wrong",
    "category_3_unformatted": "3: unformatted",
    "unexpected_format_zero_answer_one": "unexpected",
}

PROMPT_SPECS = {
    "question_only": {
        "path": ROOT / "cs336_alignment/prompts/question_only.prompt",
        "reward_fn": question_only_reward_fn,
        "stop": None,
        "include_stop_str_in_output": False,
    },
    "r1_zero": {
        "path": ROOT / "cs336_alignment/prompts/r1_zero.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    },
    "r1_zero_three_shot": {
        "path": ROOT / "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OLMo-2-0425-1B prompting baselines on GSM8K."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(PROMPT_SPECS),
        help="Evaluate only the selected prompt; repeat this option if needed.",
    )
    parser.add_argument(
        "--slow-grader",
        action="store_true",
        help="Also run the slower math_verify fallback in the grader.",
    )
    return parser.parse_args()


def load_examples(data_path: Path, max_examples: int | None) -> list[dict[str, str]]:
    examples = []
    with data_path.open(encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            answer = item["answer"]
            if "####" not in answer:
                raise ValueError(
                    f"Missing GSM8K final-answer marker on line {line_number}."
                )
            ground_truth = answer.rsplit("####", maxsplit=1)[-1].strip()
            examples.append(
                {
                    "question": item["question"],
                    "ground_truth": ground_truth,
                }
            )
            if max_examples is not None and len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError(f"No examples found in {data_path}.")
    return examples


def render_prompts(template_path: Path, examples: list[dict[str, str]]) -> list[str]:
    template = template_path.read_text(encoding="utf-8")
    return [template.format(question=example["question"]) for example in examples]


def classify_reward(scores: dict[str, float]) -> str:
    format_correct = bool(scores["format_reward"])
    answer_correct = bool(scores["answer_reward"])
    if format_correct and answer_correct:
        return "category_1_correct_and_formatted"
    if format_correct and not answer_correct:
        return "category_2_formatted_but_wrong"
    if not format_correct and not answer_correct:
        return "category_3_unformatted"
    return "unexpected_format_zero_answer_one"


def summarize(records: list[dict]) -> dict:
    summary = {}
    grouped_records = defaultdict(list)
    for record in records:
        grouped_records[record["prompt_name"]].append(record)

    for prompt_name, prompt_records in grouped_records.items():
        counts = Counter(record["category"] for record in prompt_records)
        category_counts = {
            category: counts.get(category, 0) for category in CATEGORY_ORDER
        }
        num_examples = len(prompt_records)
        summary[prompt_name] = {
            "num_examples": num_examples,
            "category_counts": category_counts,
            "format_rate": (
                category_counts["category_1_correct_and_formatted"]
                + category_counts["category_2_formatted_but_wrong"]
            )
            / num_examples,
            "answer_reward_rate": category_counts[
                "category_1_correct_and_formatted"
            ]
            / num_examples,
            "mean_reward": sum(
                record["reward"] for record in prompt_records
            )
            / num_examples,
        }
    return summary


def write_details_csv(records: list[dict], path: Path) -> None:
    fields = [
        "prompt_name",
        "example_index",
        "category",
        "ground_truth",
        "format_reward",
        "answer_reward",
        "reward",
        "finish_reason",
        "question",
        "response",
        "prompt",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: record.get(field, "") for field in fields}
            for record in records
        )


def _markdown_cell(value: object, max_length: int | None = None) -> str:
    text = str(value).replace("\r", "").replace("\n", " ").replace("|", "\\|")
    if max_length is not None and len(text) > max_length:
        text = text[: max_length - 3] + "..."
    return text


def write_markdown_report(records: list[dict], summary: dict, path: Path) -> None:
    lines = [
        "# GSM8K Prompting Baselines",
        "",
        "The complete prompts and responses are in the accompanying details CSV.",
        "",
        "## Summary",
        "",
        "| Prompt | N | Category 1 | Category 2 | Category 3 | Format rate | Answer reward rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for prompt_name, prompt_summary in summary.items():
        counts = prompt_summary["category_counts"]
        lines.append(
            "| {prompt} | {n} | {c1} | {c2} | {c3} | {format_rate:.2%} | {answer_rate:.2%} |".format(
                prompt=prompt_name,
                n=prompt_summary["num_examples"],
                c1=counts["category_1_correct_and_formatted"],
                c2=counts["category_2_formatted_but_wrong"],
                c3=counts["category_3_unformatted"],
                format_rate=prompt_summary["format_rate"],
                answer_rate=prompt_summary["answer_reward_rate"],
            )
        )

    lines.extend(
        [
            "",
            "## Detailed Generations",
            "",
            "The response column is shortened here for readability; the CSV has the full text.",
            "",
            "| Prompt | Example | Category | Ground truth | Format | Answer | Finish | Response preview |",
            "| --- | ---: | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for record in records:
        lines.append(
            "| {prompt} | {index} | {category} | {ground_truth} | {format_reward:.0f} | {answer_reward:.0f} | {finish_reason} | {response} |".format(
                prompt=_markdown_cell(record["prompt_name"]),
                index=record["example_index"],
                category=_markdown_cell(CATEGORY_LABELS[record["category"]]),
                ground_truth=_markdown_cell(record["ground_truth"]),
                format_reward=record["format_reward"],
                answer_reward=record["answer_reward"],
                finish_reason=_markdown_cell(record["finish_reason"]),
                response=_markdown_cell(record["response"], max_length=500),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # 先读取命令行配置和 GSM8K 示例，再为每个 prompt 模板构造请求。
    args = parse_args()
    examples = load_examples(args.data_path, args.max_examples)
    selected_names = args.only or list(PROMPT_SPECS)
    records = []

    for prompt_name in selected_names:
        # 每个模板单独生成、评分和记录，便于比较格式正确率与答案正确率。
        spec = PROMPT_SPECS[prompt_name]
        prompts = render_prompts(spec["path"], examples)
        sampling_params = {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "n": 1,
            "seed": args.seed,
            "stop": spec["stop"],
            "include_stop_str_in_output": spec["include_stop_str_in_output"],
        }

        completions = generate_completions(
            vllm_base_url=args.base_url,
            model_id=args.model_id,
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=args.batch_size,
        )
        if len(completions) != len(examples):
            raise RuntimeError(
                f"Expected {len(examples)} completions for {prompt_name}, "
                f"but received {len(completions)}."
            )

        reward_fn = spec["reward_fn"]
        for example_index, (example, prompt, completion) in enumerate(
            zip(examples, prompts, completions), start=1
        ):
            scores = reward_fn(
                completion.text,
                example["ground_truth"],
                fast=not args.slow_grader,
            )
            records.append(
                {
                    "prompt_name": prompt_name,
                    "example_index": example_index,
                    "question": example["question"],
                    "ground_truth": example["ground_truth"],
                    "prompt": prompt,
                    "response": completion.text,
                    "finish_reason": completion.finish_reason,
                    "format_reward": float(scores["format_reward"]),
                    "answer_reward": float(scores["answer_reward"]),
                    "reward": float(scores["reward"]),
                    "category": classify_reward(scores),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize(records)
    summary_path = args.output.with_name(args.output.stem + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    details_path = args.output.with_name(args.output.stem + ".details.csv")
    report_path = args.output.with_name(args.output.stem + ".report.md")
    write_details_csv(records, details_path)
    write_markdown_report(records, summary, report_path)

    print(f"Saved records to {args.output}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved detailed CSV to {details_path}")
    print(f"Saved Markdown report to {report_path}")
    for prompt_name, prompt_summary in summary.items():
        print(prompt_name)
        for category in CATEGORY_ORDER:
            count = prompt_summary["category_counts"][category]
            if category != "unexpected_format_zero_answer_one" or count:
                print(f"  {CATEGORY_LABELS[category]}: {count}")
        print(f"  format rate: {prompt_summary['format_rate']:.2%}")
        print(f"  answer reward rate: {prompt_summary['answer_reward_rate']:.2%}")


if __name__ == "__main__":
    main()
