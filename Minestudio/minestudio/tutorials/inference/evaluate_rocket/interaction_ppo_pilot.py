import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-group", type=str, default="rocket2_official")
    parser.add_argument("--task-group-path", type=str, default="/home/gyulab/envgen2/ROCKET-2/env_conf")
    parser.add_argument("--env-source", type=str, default="rocket2_official", choices=["local", "rocket2_official"])
    parser.add_argument("--protocol", type=str, default="ours_v1")
    parser.add_argument("--tasks", type=str, default="hunt_cow_do_not_touch_sheep")
    parser.add_argument("--eval-episodes-per-task", type=int, default=10)
    parser.add_argument("--collect-episodes-per-task", type=int, default=10)
    parser.add_argument("--train-iters", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--episode-retries", type=int, default=3)
    parser.add_argument("--stop-on-auto-success", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--model-path", type=str, default="CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
    parser.add_argument("--sam-path", type=str, required=True)
    parser.add_argument("--molmo-id", type=str, default="allenai/MolmoE-1B-0924")
    parser.add_argument("--molmo-url", type=str, default="managed-local")
    parser.add_argument("--molmo-api-key", type=str, default="EMPTY")
    parser.add_argument("--molmo-managed-env", type=str, default="molmo-pointing")
    parser.add_argument("--molmo-loader", type=str, default="manual")
    parser.add_argument("--molmo-torch-dtype", type=str, default="float16")
    parser.add_argument("--molmo-autocast-dtype", type=str, default="float16")
    parser.add_argument("--refresh-on-tracker-loss", action="store_true")
    parser.add_argument("--min-tracker-area", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-learning-rate", type=float, default=1e-5)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--policy-coef", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--normalize-advantage", action="store_true")
    parser.add_argument("--clip-vloss", action="store_true")
    parser.add_argument("--cleanup-stale-pointing-servers", action="store_true", default=True)
    parser.add_argument("--out-dir", type=str, default="outputs/evaluate_rocket/interaction_ppo_pilot")
    return parser.parse_args()


def latest_subdir(path: Path) -> Path:
    children = sorted([child for child in path.iterdir() if child.is_dir()])
    if not children:
        raise RuntimeError(f"No run directories found under {path}")
    return children[-1]


def run_command(cmd: List[str], env):
    print("[interaction-ppo-pilot] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def cleanup_pointing_servers():
    pattern = "python -m minestudio.tutorials.inference.evaluate_rocket.pointing_server"
    subprocess.run(
        ["pkill", "-f", pattern],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)


def load_summary(summary_json_path: Path) -> Dict[str, Dict]:
    rows = json.loads(summary_json_path.read_text(encoding="utf-8"))
    return {row["benchmark_name"]: row for row in rows}


def build_common_molmo_args(args) -> List[str]:
    return [
        "--molmo-id",
        args.molmo_id,
        "--molmo-url",
        args.molmo_url,
        "--molmo-api-key",
        args.molmo_api_key,
        "--molmo-managed-env",
        args.molmo_managed_env,
        "--molmo-loader",
        args.molmo_loader,
        "--molmo-torch-dtype",
        args.molmo_torch_dtype,
        "--molmo-autocast-dtype",
        args.molmo_autocast_dtype,
    ]


def build_eval_cmd(args, out_dir: Path, model_path: str) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_runner",
        "--env-source",
        args.env_source,
        "--protocol",
        args.protocol,
        "--task-group",
        args.task_group,
        "--task-group-path",
        args.task_group_path,
        "--tasks",
        args.tasks,
        "--episodes-per-task",
        str(args.eval_episodes_per_task),
        "--base-seed",
        str(args.base_seed),
        "--episode-retries",
        str(args.episode_retries),
        "--model-path",
        model_path,
        "--sam-path",
        args.sam_path,
        *build_common_molmo_args(args),
        "--out-dir",
        str(out_dir),
    ]
    if args.skip_video:
        cmd.append("--skip-video")
    if args.stop_on_auto_success:
        cmd.append("--stop-on-auto-success")
    return cmd


def build_collect_cmd(args, out_dir: Path, model_path: str) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "minestudio.tutorials.inference.evaluate_rocket.interaction_posttrain_runner",
        "--env-source",
        args.env_source,
        "--protocol",
        args.protocol,
        "--task-group",
        args.task_group,
        "--task-group-path",
        args.task_group_path,
        "--tasks",
        args.tasks,
        "--episodes-per-task",
        str(args.collect_episodes_per_task),
        "--base-seed",
        str(args.base_seed),
        "--episode-retries",
        str(args.episode_retries),
        "--model-path",
        model_path,
        "--sam-path",
        args.sam_path,
        *build_common_molmo_args(args),
        "--min-tracker-area",
        str(args.min_tracker_area),
        "--save-ppo-fragments",
        "--out-dir",
        str(out_dir),
    ]
    if args.refresh_on_tracker_loss:
        cmd.append("--refresh-on-tracker-loss")
    if args.skip_video:
        cmd.append("--skip-video")
    return cmd


def build_update_cmd(args, run_dir: Path, out_dir: Path, model_path: str) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "minestudio.tutorials.inference.evaluate_rocket.interaction_ppo_update",
        "--run-dir",
        str(run_dir),
        "--model-path",
        model_path,
        "--output-dir",
        str(out_dir),
        "--epochs",
        str(args.ppo_epochs),
        "--learning-rate",
        str(args.ppo_learning_rate),
        "--ppo-clip",
        str(args.ppo_clip),
        "--vf-coef",
        str(args.vf_coef),
        "--policy-coef",
        str(args.policy_coef),
        "--entropy-coef",
        str(args.entropy_coef),
        "--kl-coef",
        str(args.kl_coef),
    ]
    if args.normalize_advantage:
        cmd.append("--normalize-advantage")
    if args.clip_vloss:
        cmd.append("--clip-vloss")
    return cmd


def main():
    args = parse_args()
    out_root = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    baseline_root = out_root / "baseline"
    iterations_root = out_root / "iterations"
    final_eval_root = out_root / "final_eval"
    for root in (baseline_root, iterations_root, final_eval_root):
        root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MINESTUDIO_DIR", str(Path.home() / ".minestudio"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    current_model_path = args.model_path

    if args.cleanup_stale_pointing_servers:
        cleanup_pointing_servers()
    run_command(build_eval_cmd(args, baseline_root, current_model_path), env)
    baseline_run_dir = latest_subdir(baseline_root)
    if args.cleanup_stale_pointing_servers:
        cleanup_pointing_servers()

    iteration_summaries = []
    for iteration_idx in range(1, int(args.train_iters) + 1):
        iter_dir = iterations_root / f"iter_{iteration_idx:03d}"
        collect_root = iter_dir / "collect"
        update_root = iter_dir / "update"
        eval_root = iter_dir / "eval"
        for root in (collect_root, update_root, eval_root):
            root.mkdir(parents=True, exist_ok=True)

        if args.cleanup_stale_pointing_servers:
            cleanup_pointing_servers()
        run_command(build_collect_cmd(args, collect_root, current_model_path), env)
        collect_run_dir = latest_subdir(collect_root)
        if args.cleanup_stale_pointing_servers:
            cleanup_pointing_servers()

        run_command(build_update_cmd(args, collect_run_dir, update_root, current_model_path), env)
        update_run_dir = latest_subdir(update_root)
        updated_model_path = update_run_dir / "model.pt"

        if args.cleanup_stale_pointing_servers:
            cleanup_pointing_servers()
        run_command(build_eval_cmd(args, eval_root, str(updated_model_path)), env)
        eval_run_dir = latest_subdir(eval_root)
        if args.cleanup_stale_pointing_servers:
            cleanup_pointing_servers()

        iteration_summaries.append(
            {
                "iteration": iteration_idx,
                "input_model_path": str(current_model_path),
                "collect_run_dir": str(collect_run_dir),
                "update_run_dir": str(update_run_dir),
                "eval_run_dir": str(eval_run_dir),
                "updated_model_path": str(updated_model_path),
                "eval_summary": load_summary(eval_run_dir / "summary.json"),
            }
        )
        current_model_path = str(updated_model_path)

    if args.cleanup_stale_pointing_servers:
        cleanup_pointing_servers()
    run_command(build_eval_cmd(args, final_eval_root, current_model_path), env)
    final_eval_run_dir = latest_subdir(final_eval_root)
    if args.cleanup_stale_pointing_servers:
        cleanup_pointing_servers()

    pilot_summary = {
        "config": vars(args),
        "baseline_run_dir": str(baseline_run_dir),
        "baseline_summary": load_summary(baseline_run_dir / "summary.json"),
        "iteration_summaries": iteration_summaries,
        "final_model_path": current_model_path,
        "final_eval_run_dir": str(final_eval_run_dir),
        "final_eval_summary": load_summary(final_eval_run_dir / "summary.json"),
    }
    (out_root / "pilot_summary.json").write_text(
        json.dumps(pilot_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[interaction-ppo-pilot] done")
    print(json.dumps(pilot_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
