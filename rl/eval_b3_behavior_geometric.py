#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

DEFAULT_BASE = (
    REPO
    / "logs/rsl_rl/B1_EE/"
    "2026-08-07_15-33-10_CTBR_250Hz_128_128_updated_URDF_prev_br_penalty_-0.1_br_norm_-0.05/"
    "videos/eval"
)

DEFAULT_CASE = (
    DEFAULT_BASE
    / "b3_aug06_ablation_A_current"
    / "mellinger_replay_case_v2.pt"
)

DEFAULT_RL_TRACE = (
    DEFAULT_BASE
    / "b3_aug06_ablation_A_current"
    / "rl_trace.pt"
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen B3 behavior-geometric baseline "
            "against an existing RL replay."
        )
    )

    parser.add_argument(
        "--goal",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        required=True,
        help="Literal GC/COM step in metres.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--case_path",
        type=Path,
        default=DEFAULT_CASE,
    )

    parser.add_argument(
        "--rl_trace_path",
        type=Path,
        default=DEFAULT_RL_TRACE,
    )

    parser.add_argument(
        "--horizon",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--show_full_log",
        action="store_true",
    )

    args = parser.parse_args()

    for path in (
        args.case_path,
        args.rl_trace_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    goal_text = ",".join(
        f"{v:.9g}"
        for v in args.goal
    )

    env = os.environ.copy()

    # ================================================================
    # Frozen B3 behavior-geometric V1 contract.
    # ================================================================
    env.update(
        {
            "AERIAL_STATIC_GC_GOAL_OFFSET": goal_text,
            "B3_REAL_COMMAND_ZERO_YAW": "1",
            "B3_PREHOVER_S": "2.0",
            "B3_CONTINUE_FROM_PREHOVER": "1",
            "B3_REAL_GOAL_DELIVERY_DELAY_S": "0",
            "B3_GYRO_LPF_ENABLE": "0",
            "B3_DIFF_ACTUATOR_ENABLE": "0",
            "B3_UNIFIED_ACTUATOR_ENABLE": "0",
            "B3_SIM_TRAJ_ENABLE": "0",
            "B3_BEHAVIOR_GEOM_ENABLE": "1",
        }
    )

    command = [
        sys.executable,
        str(
            REPO
            / "rl"
            / "mellinger_rslrl_compare.py"
        ),
        "--task",
        "Isaac-Crazyflie-0DOF-Hover-v0",
        "--case_path",
        str(args.case_path),
        "--rl_trace_path",
        str(args.rl_trace_path),
        "--output_dir",
        str(args.output_dir),
        "--benchmark_horizon_s",
        str(args.horizon),
        "--device",
        args.device,
    ]

    result = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if args.show_full_log:
        print(result.stdout)
    else:
        lines = result.stdout.splitlines()

        wanted = False

        for line in lines:
            if (
                "B3 BEHAVIOR CONTROLLER FIRST STEP"
                in line
                or
                "COMMON APPLES-TO-APPLES BENCHMARK METRICS"
                in line
            ):
                wanted = True
                print()
                print(line)
                continue

            if wanted:
                print(line)

                if (
                    "benchmark manifest"
                    in line
                ):
                    wanted = False

        if result.returncode != 0:
            print()
            print("===== FAILURE TAIL =====")
            print(
                "\n".join(
                    lines[-80:]
                )
            )

    if result.returncode != 0:
        raise SystemExit(
            result.returncode
        )


if __name__ == "__main__":
    main()
