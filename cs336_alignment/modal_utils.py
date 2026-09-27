"""Modal helpers for running assignment 5 jobs.

Example usage:

import sys
from cs336_alignment.modal_utils import app, quote_command, submit_commands

def build_run_commands(args):
    # Suppose args.seeds = '0,1,2,3'
    return [
        [sys.executable, "-u", "scripts/grpo.py", "--seed", seed]
        for seed in args.seeds.split(',')
    ]

@app.local_entrypoint(name=...)
def modal_main(*argv: str) -> None:
    args = make_parser().parse_args(list(argv))
    commands = build_run_commands(args)
    submit_commands(commands)
"""

from __future__ import annotations

import shlex
import subprocess

import modal


# SUNET_ID 用于隔离课程作业的 Modal app、W&B secret 和远程资源命名空间。
SUNET_ID = "TODO"  # NOTE: modal_utils.py should remain unchanged other than adding your SUNET_ID.
if SUNET_ID == "TODO":
    raise ValueError("Please set SUNET_ID in cs336_alignment/modal_utils.py before running Modal jobs.")


# 这些常量同时控制远程 GPU 数量、容器并发和单个任务超时。
GPU = "B200:2"
MAX_CONTAINERS = 4
REMOTE_ROOT = "/root"
RUN_TIMEOUT_SECONDS = 60 * 60
WANDB_SECRET_NAME = "wandb"

app = modal.App(f"cs336-a5-rlvr-{SUNET_ID}")
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)

image = (
    # 远程镜像包含 CUDA、项目依赖和训练所需的本地目录。
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .uv_sync(extras=["gpu"])
    .workdir(REMOTE_ROOT)
    .add_local_dir("cs336_alignment", f"{REMOTE_ROOT}/cs336_alignment")
    .add_local_dir("data", f"{REMOTE_ROOT}/data")
    .add_local_dir("experiments", f"{REMOTE_ROOT}/experiments")
    .add_local_dir("scripts", f"{REMOTE_ROOT}/scripts")
    .add_local_file("pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml")
    .add_local_file("uv.lock", f"{REMOTE_ROOT}/uv.lock")
)
image = image.add_local_file("AGENTS.md", f"{REMOTE_ROOT}/AGENTS.md")
image = image.add_local_file("CLAUDE.md", f"{REMOTE_ROOT}/CLAUDE.md")


def quote_command(command: list[str]) -> str:
    """把参数列表安全地转换成可复制的 shell 命令字符串。"""
    return " ".join(shlex.quote(part) for part in command)


@app.function(
    image=image,
    gpu=GPU,
    timeout=RUN_TIMEOUT_SECONDS,
    max_containers=MAX_CONTAINERS,
    secrets=[wandb_secret],
)
def run_command(command: list[str]) -> str:
    """在一个 Modal GPU 容器中执行单条命令，并让失败状态向上抛出。"""
    command_str = quote_command(command)
    print(command_str, flush=True)
    subprocess.run(command, check=True)
    return command_str


def submit_commands(commands: list[list[str]]) -> None:
    """并发提交命令，收集所有失败任务后统一返回失败状态。"""
    print(
        f"Submitting {len(commands)} Modal jobs "
        f"with max_containers={MAX_CONTAINERS}, gpu={GPU}, "
        f"timeout={RUN_TIMEOUT_SECONDS}s.",
        flush=True,
    )
    failures = []
    for idx, result in enumerate(run_command.map(commands, return_exceptions=True)):
        command = commands[idx]
        command_str = quote_command(command)
        if isinstance(result, BaseException):
            print(f"Failed: {command_str}", flush=True)
            print(f"Error: {result!r}", flush=True)
            failures.append(command_str)
        else:
            print(f"Completed: {result}", flush=True)

    if failures:
        print(f"{len(failures)} of {len(commands)} Modal jobs failed.", flush=True)
        raise SystemExit(1)
