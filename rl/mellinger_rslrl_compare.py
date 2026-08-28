# MELLINGER_RSLRL_REPLAY_HELPER_V1

import argparse
import json
import sys

# ============================================================================
# STATIC_TRANSFER_ORCHESTRATOR_V4
#
# Runs BEFORE Isaac AppLauncher.
#
# Normal replay mode below is unchanged.
# eval_rslrl.py is read but NEVER written.
# ============================================================================
if "--static_transfer" in sys.argv:
    import os as _static_os
    import atexit as _static_atexit
    import subprocess as _static_subprocess
    from pathlib import Path as _StaticPath

    _sp = argparse.ArgumentParser(
        description=(
            "Deterministic static-EE Model-575 vs "
            "frozen-Mellinger benchmark."
        )
    )

    _sp.add_argument("--static_transfer", action="store_true")
    _sp.add_argument(
        "--task",
        default="Isaac-Crazyflie-0DOF-Hover-v0",
    )
    _sp.add_argument(
        "--static_goal_offset",
        type=float,
        nargs=3,
        required=True,
        metavar=("DX", "DY", "DZ"),
    )
    _sp.add_argument(
        "--static_pre_hover_s",
        type=float,
        default=2.0,
    )
    _sp.add_argument(
        "--static_seed",
        type=int,
        default=0,
    )
    _sp.add_argument(
        "--static_experiment_name",
        default="B1_EE",
    )
    _sp.add_argument(
        "--static_load_run",
        default=(
            "2026-08-07_15-33-10_CTBR_250Hz_128_128_"
            "updated_URDF_prev_br_penalty_-0.1_br_norm_-0.05"
        ),
    )
    _sp.add_argument(
        "--static_checkpoint",
        default="model_575.pt",
    )
    _sp.add_argument(
        "--device",
        default="cuda:0",
    )
    _sp.add_argument(
        "--output_dir",
        default=None,
    )

    _sa = _sp.parse_args()

    if _sa.static_pre_hover_s < 0.0:
        _sp.error("--static_pre_hover_s must be >= 0")

    _dx, _dy, _dz = [
        float(v)
        for v in _sa.static_goal_offset
    ]

    _repo = _StaticPath("/home/sumukh/AerialManipulation")
    _eval = _repo / "rl" / "eval_rslrl.py"
    _compare = _repo / "rl" / "mellinger_rslrl_compare.py"

    _run = (
        _repo
        / "logs"
        / "rsl_rl"
        / _sa.static_experiment_name
        / _sa.static_load_run
    )

    _checkpoint = _run / _sa.static_checkpoint

    if not _eval.is_file():
        raise RuntimeError(f"Missing evaluator: {_eval}")

    if not _checkpoint.is_file():
        raise RuntimeError(f"Missing checkpoint: {_checkpoint}")

    def _static_tag(value):
        sign = "p" if value >= 0.0 else "m"
        mag = f"{abs(value):.3f}".replace(".", "p")
        return sign + mag

    _case_name = (
        "static_transfer"
        f"_dx_{_static_tag(_dx)}"
        f"_dy_{_static_tag(_dy)}"
        f"_dz_{_static_tag(_dz)}"
        f"_seed{int(_sa.static_seed)}"
    )

    if _sa.output_dir is None:
        _out = (
            _run
            / "videos"
            / "eval"
            / _case_name
        )
    else:
        _out = (
            _StaticPath(_sa.output_dir)
            .expanduser()
            .resolve()
        )

    if _out.exists():
        raise RuntimeError(
            "Refusing to overwrite existing benchmark directory: "
            f"{_out}"
        )

    _out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # Build a TEMPORARY normal rl.* module.
    #
    # This preserves the exact process/module semantics of:
    #
    #     python -m rl.eval_rslrl
    #
    # which we have already verified is stable with Isaac.
    #
    # eval_rslrl.py itself is NEVER written.
    # ------------------------------------------------------------------------
    _runtime_lines = _eval.read_text().splitlines(keepends=True)

    # ------------------------------------------------------------------------
    # Runtime-only change 1:
    # bypass the REAL_B3 + compare_mellinger guard.
    # ------------------------------------------------------------------------
    _guard_hits = []

    for _i, _line in enumerate(_runtime_lines):
        if _line.strip() == "if args_cli.compare_mellinger:":
            _window = "".join(_runtime_lines[_i:_i + 10])

            if "--real_b3_benchmark is not supported with " in _window:
                _guard_hits.append(_i)

    if len(_guard_hits) > 1:
        raise RuntimeError(
            "Expected at most one REAL_B3 comparison guard, got "
            f"{_guard_hits}"
        )

    # Older eval_rslrl.py revisions explicitly prohibited
    # REAL_B3 + compare_mellinger. Current revisions no longer
    # contain that guard, so zero matches requires no patch.
    if len(_guard_hits) == 1:
        _i = _guard_hits[0]
        _guard_end = None

        for _j in range(
            _i + 1,
            min(_i + 10, len(_runtime_lines)),
        ):
            if _runtime_lines[_j].strip() == ")":
                _guard_end = _j
                break

        if _guard_end is None:
            raise RuntimeError(
                "Could not locate REAL_B3 comparison guard end."
            )

        _indent = _runtime_lines[_i][
            :len(_runtime_lines[_i])
            - len(_runtime_lines[_i].lstrip())
        ]

        _runtime_lines[_i:_guard_end + 1] = [
            _indent + "if args_cli.compare_mellinger:\n",
            _indent + "    pass\n",
        ]

    # ------------------------------------------------------------------------
    # Runtime-only change 2:
    # redirect ALL V2 comparison artifact blocks to this benchmark case.
    # ------------------------------------------------------------------------
    _dir_hits = []

    for _i, _line in enumerate(_runtime_lines):
        if "compare_dir_v2 = os.path.join(" in _line:
            _window = "".join(_runtime_lines[_i:_i + 8])

            if "mellinger_compare_robot_{compare_robot_v2}" in _window:
                _dir_hits.append(_i)

    if not _dir_hits:
        raise RuntimeError(
            "No V2 comparison-directory blocks were found."
        )

    for _i in reversed(_dir_hits):
        _dir_end = None

        for _j in range(
            _i + 1,
            min(_i + 8, len(_runtime_lines)),
        ):
            if _runtime_lines[_j].strip() == ")":
                _dir_end = _j
                break

        if _dir_end is None:
            raise RuntimeError(
                "Could not locate V2 comparison-directory "
                f"block end at line {_i}."
            )

        _indent = _runtime_lines[_i][
            :len(_runtime_lines[_i])
            - len(_runtime_lines[_i].lstrip())
        ]

        _runtime_lines[_i:_dir_end + 1] = [
            _indent
            + "compare_dir_v2 = os.environ.get("
            + "'AERIAL_STATIC_COMPARE_DIR'"
            + ")\n",

            _indent
            + "if not compare_dir_v2:\n",

            _indent
            + "    compare_dir_v2 = os.path.join(\n",

            _indent
            + "        video_folder_path,\n",

            _indent
            + '        f"mellinger_compare_robot_'
            + '{compare_robot_v2}",\n',

            _indent
            + "    )\n",
        ]

    _runtime_source = "".join(_runtime_lines)

    # Syntax-check before creating the temporary runtime module.
    compile(
        _runtime_source,
        str(_eval),
        "exec",
    )

    _runtime_module = (
        f"_static_eval_runtime_{_static_os.getpid()}"
    )

    _runtime_path = (
        _repo
        / "rl"
        / f"{_runtime_module}.py"
    )

    if _runtime_path.exists():
        raise RuntimeError(
            f"Temporary runtime module already exists: {_runtime_path}"
        )

    _runtime_path.write_text(_runtime_source)

    def _cleanup_static_runtime():
        _runtime_path.unlink(missing_ok=True)

    _static_atexit.register(
        _cleanup_static_runtime
    )

    # RL child: exact Model-575 contract already independently verified.
    # ------------------------------------------------------------------------
    _eval_cmd = [
        sys.executable,
        "-m",
        f"rl.{_runtime_module}",

        "--task",
        _sa.task,

        "--num_envs",
        "1",

        "--seed",
        str(int(_sa.static_seed)),

        "--experiment_name",
        _sa.static_experiment_name,

        "--load_run",
        _sa.static_load_run,

        "--checkpoint",
        _sa.static_checkpoint,

        "--follow_robot",
        "0",

            "--benchmark_pre_hover_s",
        str(float(_sa.static_pre_hover_s)),

        "--benchmark_goal_offset",
        str(_dx),
        str(_dy),
        str(_dz),

        "--compare_mellinger",

        "--device",
        _sa.device,

        "env.control_mode=CTBR",
        "env.policy_rate_hz=250",
        "env.decimation=4",
        "env.sim.render_interval=4",

        "env.use_yaw_representation=false",
        "env.use_full_ori_matrix=true",
        "env.use_grav_vector=true",
        "env.use_previous_actions=true",
        "env.action_history_length=11",

        "env.task_body=endeffector",
        "env.goal_body=endeffector",
        "env.reward_task_body=endeffector",
        "env.reward_goal_body=endeffector",
        "env.visualization_body=endeffector",

        "agent.policy.actor_hidden_dims=[128,128]",
        "agent.policy.critic_hidden_dims=[128,128]",
        "agent.policy.activation=tanh",
    ]

    _child_env = _static_os.environ.copy()

    # Run the evaluator directly as its RL child.
    _child_env["AERIAL_COMPARE_RL_CHILD"] = "1"

    # Unique output directory for this transfer case.
    _child_env["AERIAL_STATIC_COMPARE_DIR"] = str(_out)

    # eval_rslrl uses rgb_array / RecordVideo.
    _child_env["HEADLESS"] = "1"
    _child_env["ENABLE_CAMERAS"] = "1"

    _child_env["HYDRA_FULL_ERROR"] = "1"
    _child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    _child_env["PYTHONUNBUFFERED"] = "1"

    print()
    print("=" * 100)
    print("STATIC EE TRANSFER - RL CAPTURE")
    print("=" * 100)
    print("checkpoint         :", _checkpoint)
    print("EE goal offset [m] :", [_dx, _dy, _dz])
    print("RL pre-hover [s]   :", float(_sa.static_pre_hover_s))
    print("RL policy rate     : 250 Hz")
    print("physical plant     : REAL_B3")
    print("output             :", _out)
    print(
        "eval_rslrl.py      : temporary normal rl module; "
        "on-disk file untouched"
    )
    print("=" * 100)
    print()

    try:
        _rl = _static_subprocess.run(
            _eval_cmd,
            cwd=str(_repo),
            env=_child_env,
        )
    finally:
        _cleanup_static_runtime()

    if _rl.returncode != 0:
        raise RuntimeError(
            "Static-transfer RL capture failed with "
            f"return code {_rl.returncode}"
        )

    _case = (
        _out
        / "mellinger_replay_case_v2.pt"
    )

    _trace = (
        _out
        / "rl_trace.pt"
    )

    if not _case.is_file():
        raise RuntimeError(
            f"Missing replay case: {_case}"
        )

    if not _trace.is_file():
        raise RuntimeError(
            f"Missing RL trace: {_trace}"
        )

    print()
    print("=" * 100)
    print("RL STATIC TRANSFER CAPTURE COMPLETE")
    print("=" * 100)
    print("replay case :", _case)
    print("RL trace    :", _trace)
    print("=" * 100)
    print()

    # ------------------------------------------------------------------------
    # Run this same file again in its EXISTING NORMAL replay mode.
    # ------------------------------------------------------------------------
    _compare_cmd = [
        sys.executable,
        str(_compare),

        "--task",
        _sa.task,

        "--case_path",
        str(_case),

        "--rl_trace_path",
        str(_trace),

        "--output_dir",
        str(_out),

        "--benchmark_horizon_s",
        "4.0",

        "--device",
        _sa.device,
    ]

    _compare_env = _static_os.environ.copy()
    _compare_env["HEADLESS"] = "1"
    _compare_env["PYTHONUNBUFFERED"] = "1"

    print("=" * 100)
    print("FROZEN MELLINGER REPLAY")
    print("=" * 100)
    print("benchmark horizon : 4.0 s")
    print("=" * 100)
    print()

    _cmp = _static_subprocess.run(
        _compare_cmd,
        cwd=str(_repo),
        env=_compare_env,
    )

    if _cmp.returncode != 0:
        raise RuntimeError(
            "Frozen-Mellinger replay failed with "
            f"return code {_cmp.returncode}"
        )

    _metrics = (
        _out
        / "benchmark_metrics.json"
    )

    if not _metrics.is_file():
        raise RuntimeError(
            f"Missing benchmark metrics: {_metrics}"
        )

    print()
    print("=" * 100)
    print("STATIC TRANSFER COMPARISON COMPLETE")
    print("=" * 100)
    print("Artifacts :", _out)
    print("Metrics   :", _metrics)
    print("=" * 100)

    raise SystemExit(0)


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Replay an exact realized RL Crazyflie case using the frozen "
        "firmware-style Mellinger controller."
    )
)

parser.add_argument("--task", type=str, required=True)
parser.add_argument("--case_path", type=str, required=True)
parser.add_argument("--rl_trace_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--video", action="store_true", default=False)

parser.add_argument(
    "--benchmark_horizon_s",
    type=float,
    default=4.0,
    help=(
        "Independent physical maneuver horizon for Mellinger replay. "
        "Default: 4.0 s."
    ),
)

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = bool(
    args_cli.enable_cameras or args_cli.video
)
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import math
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------
# Force imports from THIS AerialManipulation repository.
# ---------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)

while _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)

sys.path.insert(0, _REPO_ROOT_STR)

import envs

_envs_path = Path(envs.__file__).resolve()
_expected_envs_root = (_REPO_ROOT / "envs").resolve()

if _expected_envs_root not in _envs_path.parents:
    raise RuntimeError(
        "Wrong envs package imported. "
        f"Expected under {_expected_envs_root}, got {_envs_path}"
    )

from controllers.cf_mellinger_firmware import (
    CrazyflieFirmwareMellinger,
    FIRMWARE_COMMIT,
)

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config


def load_pt(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


CASE = load_pt(args_cli.case_path)
RL_TRACE = load_pt(args_cli.rl_trace_path)

output_dir = Path(args_cli.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


def tensor_cpu(value):
    if value is None:
        return None

    if torch.is_tensor(value):
        return value.detach().cpu()

    return torch.as_tensor(value)


def quaternion_yaw_wxyz(q):
    q = torch.as_tensor(q, dtype=torch.float64).reshape(-1)

    if q.numel() != 4:
        raise RuntimeError(
            f"Expected quaternion with 4 values, got shape={tuple(q.shape)}"
        )

    w, x, y, z = q

    return float(
        torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        ).item()
    )


def quaternion_error(q, q_ref):
    q = q.reshape(-1)
    q_ref = q_ref.reshape(-1)

    return min(
        float(torch.max(torch.abs(q - q_ref)).item()),
        float(torch.max(torch.abs(q + q_ref)).item()),
    )


def rpy_from_quaternion_np(q):
    q = np.asarray(q, dtype=np.float64)

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )

    sin_pitch = 2.0 * (w * y - z * x)

    pitch = np.arcsin(
        np.clip(
            sin_pitch,
            -1.0,
            1.0,
        )
    )

    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )

    return np.stack(
        [roll, pitch, yaw],
        axis=1,
    )



# =====================================================================
# COMMON_BENCHMARK_METRIC_HELPERS_V1
#
# These metrics depend only on realized physical trajectories.
# They do not use controller reward, observations, or action semantics.
# =====================================================================

def _ee_speed_from_position(position, dt):
    position = np.asarray(
        position,
        dtype=np.float64,
    )

    if position.shape[0] < 2:
        return np.zeros(
            position.shape[0],
            dtype=np.float64,
        )

    velocity = np.gradient(
        position,
        float(dt),
        axis=0,
    )

    return np.linalg.norm(
        velocity,
        axis=1,
    )


def _tilt_deg_from_quaternion_wxyz(quaternion):
    quaternion = np.asarray(
        quaternion,
        dtype=np.float64,
    )

    w = quaternion[:, 0]
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    z = quaternion[:, 3]

    norm = np.sqrt(
        w * w
        + x * x
        + y * y
        + z * z
    )

    norm = np.maximum(
        norm,
        1.0e-12,
    )

    x = x / norm
    y = y / norm

    # body +Z dotted with world +Z.
    cos_tilt = (
        1.0
        - 2.0 * (
            x * x
            + y * y
        )
    )

    return np.degrees(
        np.arccos(
            np.clip(
                cos_tilt,
                -1.0,
                1.0,
            )
        )
    )


def _first_settling_time(
    position_error,
    speed,
    dt,
    position_threshold,
    speed_threshold,
    hold_time,
):
    position_error = np.asarray(
        position_error,
        dtype=np.float64,
    )

    speed = np.asarray(
        speed,
        dtype=np.float64,
    )

    good = (
        (position_error < float(position_threshold))
        & (speed < float(speed_threshold))
    )

    hold_samples = max(
        1,
        int(
            np.ceil(
                float(hold_time)
                / float(dt)
            )
        ),
    )

    if good.shape[0] < hold_samples:
        return False, None

    run = 0

    for index, value in enumerate(good):
        if value:
            run += 1
        else:
            run = 0

        if run >= hold_samples:
            first_index = (
                index
                - hold_samples
                + 1
            )

            return (
                True,
                float(
                    first_index
                    * float(dt)
                ),
            )

    return False, None


def _integral(error, dt):
    error = np.asarray(
        error,
        dtype=np.float64,
    )

    if error.shape[0] < 2:
        return 0.0

    if hasattr(np, "trapezoid"):
        return float(
            np.trapezoid(
                error,
                dx=float(dt),
            )
        )

    return float(
        np.trapz(
            error,
            dx=float(dt),
        )
    )


def _max_along_track_overshoot(
    ee_position,
    initial_ee_position,
    goal_position,
):
    ee_position = np.asarray(
        ee_position,
        dtype=np.float64,
    )

    initial_ee_position = np.asarray(
        initial_ee_position,
        dtype=np.float64,
    ).reshape(3)

    goal_position = np.asarray(
        goal_position,
        dtype=np.float64,
    ).reshape(3)

    direction = (
        goal_position
        - initial_ee_position
    )

    distance = float(
        np.linalg.norm(direction)
    )

    if distance < 1.0e-12:
        return 0.0

    unit_direction = (
        direction
        / distance
    )

    # Positive means the EE has passed beyond the goal along the
    # initial straight-line transfer direction.
    beyond_goal = (
        ee_position
        - goal_position[None, :]
    ) @ unit_direction

    return float(
        max(
            0.0,
            float(np.max(beyond_goal)),
        )
    )


def _compute_common_metrics(
    ee_position,
    body_velocity,
    body_quaternion,
    goal_position,
    initial_ee_position,
    dt,
    xy_position_threshold,
    position_3d_threshold,
    speed_threshold,
    hold_time,
):
    ee_position = np.asarray(
        ee_position,
        dtype=np.float64,
    )

    body_velocity = np.asarray(
        body_velocity,
        dtype=np.float64,
    )

    body_quaternion = np.asarray(
        body_quaternion,
        dtype=np.float64,
    )

    goal_position = np.asarray(
        goal_position,
        dtype=np.float64,
    ).reshape(3)

    error_vector = (
        ee_position
        - goal_position[None, :]
    )

    error_xy = np.linalg.norm(
        error_vector[:, 0:2],
        axis=1,
    )

    error_z = np.abs(
        error_vector[:, 2]
    )

    error_3d = np.linalg.norm(
        error_vector,
        axis=1,
    )

    ee_speed = _ee_speed_from_position(
        ee_position,
        dt,
    )

    body_speed = np.linalg.norm(
        body_velocity,
        axis=1,
    )

    tilt_deg = _tilt_deg_from_quaternion_wxyz(
        body_quaternion
    )

    (
        xy_settled,
        xy_settling_time,
    ) = _first_settling_time(
        error_xy,
        ee_speed,
        dt,
        xy_position_threshold,
        speed_threshold,
        hold_time,
    )

    (
        settled_3d,
        settling_time_3d,
    ) = _first_settling_time(
        error_3d,
        ee_speed,
        dt,
        position_3d_threshold,
        speed_threshold,
        hold_time,
    )

    return {
        "samples": int(
            ee_position.shape[0]
        ),
        "duration_s": float(
            max(
                0,
                ee_position.shape[0] - 1,
            )
            * float(dt)
        ),

        "initial_3d_error_m": float(
            error_3d[0]
        ),

        "minimum_3d_error_m": float(
            np.min(error_3d)
        ),
        "minimum_xy_error_m": float(
            np.min(error_xy)
        ),

        "final_3d_error_m": float(
            error_3d[-1]
        ),
        "final_xy_error_m": float(
            error_xy[-1]
        ),
        "final_abs_z_error_m": float(
            error_z[-1]
        ),

        "peak_ee_speed_mps": float(
            np.max(ee_speed)
        ),
        "peak_body_speed_mps": float(
            np.max(body_speed)
        ),
        "peak_tilt_deg": float(
            np.max(tilt_deg)
        ),

        "integrated_3d_error_m_s": _integral(
            error_3d,
            dt,
        ),
        "integrated_xy_error_m_s": _integral(
            error_xy,
            dt,
        ),

        "max_along_track_overshoot_m": (
            _max_along_track_overshoot(
                ee_position,
                initial_ee_position,
                goal_position,
            )
        ),

        "xy_settled_success": bool(
            xy_settled
        ),
        "xy_settling_time_s": (
            xy_settling_time
        ),

        "settled_3d_success": bool(
            settled_3d
        ),
        "settling_3d_time_s": (
            settling_time_3d
        ),
    }


def _json_value(value):
    if value is None:
        return None

    if torch.is_tensor(value):
        value = (
            value
            .detach()
            .cpu()
            .numpy()
        )

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_value(item)
            for item in value
        ]

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    return str(value)


@hydra_task_config(
    args_cli.task,
    "rsl_rl_cfg_entry_point",
)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
    agent_cfg,
):
    meta = CASE["metadata"]
    plant = CASE["plant"]
    initial = CASE["initial_state"]
    goal = CASE["goal"]
    actuator = CASE["actuator"]
    latency = CASE["latency"]
    trajectory = CASE["trajectory"]

    robot_index = int(meta["robot_index"])

    # -----------------------------------------------------------------
    # This comparison currently supports a STATIC position goal.
    # Refuse to silently reinterpret a moving trajectory.
    # -----------------------------------------------------------------
    captured_pos_traj = trajectory.get("pos_traj")

    if torch.is_tensor(captured_pos_traj):
        if (
            captured_pos_traj.ndim >= 1
            and captured_pos_traj.shape[0] >= 2
        ):
            captured_velocity_part = torch.nan_to_num(
                captured_pos_traj[1].to(torch.float64)
            )

            max_captured_goal_speed = float(
                captured_velocity_part.abs().max().item()
            )

            if max_captured_goal_speed > 1.0e-5:
                raise RuntimeError(
                    "The captured RL case has a moving position trajectory. "
                    "This comparison helper intentionally supports only the "
                    "current static-goal experiment."
                )

    captured_yaw_traj = trajectory.get("yaw_traj")

    if torch.is_tensor(captured_yaw_traj):
        if (
            captured_yaw_traj.ndim >= 1
            and captured_yaw_traj.shape[0] >= 2
        ):
            captured_yaw_rate_part = torch.nan_to_num(
                captured_yaw_traj[1].to(torch.float64)
            )

            if float(
                captured_yaw_rate_part.abs().max().item()
            ) > 1.0e-5:
                raise RuntimeError(
                    "The captured RL case has a moving yaw trajectory. "
                    "Static-goal replay was expected."
                )

    # -----------------------------------------------------------------
    # Native Mellinger timing.
    #
    # Isaac physics remains 1000 Hz.
    # Mellinger runs at its native benchmark rate of 500 Hz.
    # -----------------------------------------------------------------
    MELLINGER_RATE_HZ = 500.0

    sim_dt = float(env_cfg.sim.dt)

    required_decimation_float = (
        1.0 / (sim_dt * MELLINGER_RATE_HZ)
    )

    required_decimation = int(
        round(required_decimation_float)
    )

    if abs(
        required_decimation_float
        - required_decimation
    ) > 1.0e-9:
        raise RuntimeError(
            "500 Hz Mellinger cannot be represented exactly with "
            f"sim_dt={sim_dt}"
        )

    rl_rate_hz = float(
        RL_TRACE["policy_rate_hz"]
    )

    rl_steps = int(
        RL_TRACE["steps"]
    )

    rl_trace_duration_s = (
        rl_steps / rl_rate_hz
    )

    physical_duration_s = float(
        args_cli.benchmark_horizon_s
    )

    if physical_duration_s <= 0.0:
        raise RuntimeError(
            "benchmark_horizon_s must be positive."
        )

    mellinger_steps = int(
        round(
            physical_duration_s
            * MELLINGER_RATE_HZ
        )
    )

    expected_rl_steps_for_horizon = int(
        round(
            physical_duration_s
            * rl_rate_hz
        )
    )

    print(
        "benchmark horizon      :",
        physical_duration_s,
        "s",
    )
    print(
        "captured RL duration   :",
        rl_trace_duration_s,
        "s",
    )

    if rl_steps < expected_rl_steps_for_horizon:
        print(
            "[BENCHMARK] RL trace ends before the fixed benchmark "
            "horizon. It will remain an early-ended RL trace; "
            "Mellinger still receives the full independent horizon."
        )

    # -----------------------------------------------------------------
    # Preserve the PHYSICAL command delay.
    #
    # The RL queue index itself cannot be copied because RL CTBR and
    # Mellinger SRT queue entries have different command semantics.
    # -----------------------------------------------------------------
    physical_delay_s = float(
        latency["physical_delay_seconds"]
    )

    mellinger_delay_steps_float = (
        physical_delay_s
        * MELLINGER_RATE_HZ
    )

    mellinger_delay_steps = int(
        round(mellinger_delay_steps_float)
    )

    if abs(
        mellinger_delay_steps_float
        - mellinger_delay_steps
    ) > 1.0e-5:
        raise RuntimeError(
            "Captured physical delay cannot be represented exactly at "
            f"500 Hz: delay={physical_delay_s:.9f}s"
        )

    # -----------------------------------------------------------------
    # Configure a one-robot clone of the same task.
    # -----------------------------------------------------------------
    env_cfg.scene.num_envs = 1

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env_cfg.eval_mode = True
    env_cfg.gc_mode = True
    env_cfg.control_mode = "SRT"

    env_cfg.policy_rate_hz = int(
        MELLINGER_RATE_HZ
    )
    env_cfg.decimation = required_decimation
    env_cfg.sim.render_interval = (
        required_decimation
    )

    for name in (
        "task_body",
        "goal_body",
        "reward_task_body",
        "reward_goal_body",
        "visualization_body",
    ):
        value = meta.get(name)

        if value is not None and hasattr(
            env_cfg,
            name,
        ):
            setattr(
                env_cfg,
                name,
                value,
            )

    if hasattr(
        env_cfg,
        "rotorpy_done",
    ):
        env_cfg.rotorpy_done = bool(
            meta.get(
                "rotorpy_done",
                env_cfg.rotorpy_done,
            )
        )

    # Disable new DR in this second process. We restore the exact
    # realized plant below instead.
    if isinstance(
        getattr(
            env_cfg,
            "dr_dict",
            None,
        ),
        dict,
    ):
        env_cfg.dr_dict = {
            key: 0.0
            for key in env_cfg.dr_dict
        }

    env_cfg.control_latency_steps = (
        mellinger_delay_steps
    )

    # =================================================================
    # FIXED_REAL_B3_MELLINGER_PLANT_V1
    #
    # RL uses its selected realized/randomized plant.
    # Mellinger always uses the fixed physical B3 plant below.
    # Randomized RL dynamics are NOT copied into this process.
    # =================================================================
    B3_PLANT_MASS_KG = 0.046
    B3_PLANT_IXX = 2.4255e-05
    B3_PLANT_IYY = 1.8650e-05
    B3_PLANT_IZZ = 3.9300e-05
    B3_PLANT_ARM_M = 0.050
    B3_PLANT_K_ETA = 0.51033
    B3_PLANT_K_M = 7.8e-10
    B3_PLANT_K_TORQUE = 0.003987
    B3_PLANT_TAU_M = 0.050
    B3_PLANT_KP_ATT = 3264.54
    B3_PLANT_KD_ATT = 361.58
    B3_PLANT_KP_OMEGA = 75.0
    B3_PLANT_KD_OMEGA = 10.0
    B3_PLANT_THRUST_TO_WEIGHT = 3.5

    env_cfg.mass = B3_PLANT_MASS_KG
    env_cfg.Ixx = B3_PLANT_IXX
    env_cfg.Iyy = B3_PLANT_IYY
    env_cfg.Izz = B3_PLANT_IZZ
    env_cfg.arm_length = B3_PLANT_ARM_M
    env_cfg.k_eta = B3_PLANT_K_ETA
    env_cfg.k_m = B3_PLANT_K_M
    env_cfg.k_torque = B3_PLANT_K_TORQUE
    env_cfg.tau_m = B3_PLANT_TAU_M
    env_cfg.kp_att = B3_PLANT_KP_ATT
    env_cfg.kd_att = B3_PLANT_KD_ATT
    env_cfg.kp_omega = B3_PLANT_KP_OMEGA
    env_cfg.kd_omega = B3_PLANT_KD_OMEGA
    env_cfg.thrust_to_weight = B3_PLANT_THRUST_TO_WEIGHT

    # No plant randomization in the Mellinger/B3 process.
    if isinstance(getattr(env_cfg, "dr_dict", None), dict):
        env_cfg.dr_dict = {
            key: 0.0
            for key in env_cfg.dr_dict
        }


    # -----------------------------------------------------------------
    # Recreate the exact static task goal in the new environment's
    # local frame. initialize_trajectories() adds the new env origin.
    # -----------------------------------------------------------------
    old_origin = (
        CASE["env_origin_w"]
        .to(torch.float64)
        .reshape(3)
    )

    desired_pos_original = (
        goal["desired_pos_w"]
        .to(torch.float64)
        .reshape(3)
    )

    desired_ori_original = (
        goal["desired_ori_w"]
        .to(torch.float64)
        .reshape(4)
    )

    goal_local = (
        desired_pos_original
        - old_origin
    )

    desired_yaw = (
        quaternion_yaw_wxyz(
            desired_ori_original
        )
    )

    env_cfg.trajectory_type = "lissaajous"
    env_cfg.trajectory_horizon = 0
    env_cfg.random_shift_trajectory = False

    env_cfg.lissajous_amplitudes = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    env_cfg.lissajous_amplitudes_rand_ranges = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    env_cfg.lissajous_frequencies = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    env_cfg.lissajous_frequencies_rand_ranges = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    env_cfg.lissajous_phases = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    env_cfg.lissajous_phases_rand_ranges = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    env_cfg.lissajous_offsets = [
        float(goal_local[0]),
        float(goal_local[1]),
        float(goal_local[2]),
        desired_yaw,
    ]

    env_cfg.lissajous_offsets_rand_ranges = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    # Reset itself is only a staging operation. The exact captured state
    # is written immediately afterwards.
    env_cfg.init_cfg = "rand"
    env_cfg.init_pos_ranges = [
        0.0,
        0.0,
        0.0,
    ]
    env_cfg.init_lin_vel_ranges = [
        0.0,
        0.0,
        0.0,
    ]
    env_cfg.init_yaw_ranges = [
        0.0,
    ]
    env_cfg.init_ang_vel_ranges = [
        0.0,
        0.0,
        0.0,
    ]

    # Match the normal Crazyflie follow-camera convention.
    env_cfg.viewer.eye = (
        -0.5,
        0.5,
        0.5,
    )
    env_cfg.viewer.resolution = (
        1920,
        1080,
    )
    env_cfg.viewer.lookat = (
        0.0,
        0.0,
        0.0,
    )
    env_cfg.viewer.origin_type = (
        "asset_root"
    )
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.asset_name = "robot"

    envs_gym = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array",
    )

    # Match the physical duration of the original RL video.
    requested_rl_video_steps = int(
        RL_TRACE.get(
            "video_length_rl_steps",
            rl_steps,
        )
    )

    actual_rl_video_steps = min(
        requested_rl_video_steps,
        rl_steps,
    )

    video_duration_s = (
        actual_rl_video_steps
        / rl_rate_hz
    )

    mellinger_video_steps = max(
        1,
        min(
            mellinger_steps,
            int(
                round(
                    video_duration_s
                    * MELLINGER_RATE_HZ
                )
            ),
        ),
    )

    if args_cli.video:
        envs_gym = gym.wrappers.RecordVideo(
            envs_gym,
            video_folder=str(output_dir),
            step_trigger=lambda step: step == 0,
            video_length=mellinger_video_steps,
            name_prefix=(
                f"mellinger_robot_{robot_index}"
            ),
        )

    env = envs_gym.unwrapped
    device = env.device

    obs_dict, info = envs_gym.reset()

    env_ids = torch.tensor(
        [0],
        dtype=torch.long,
        device=device,
    )

    new_origin = (
        env._terrain.env_origins[0]
        .detach()
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    old_origin_device = (
        old_origin.to(
            device=device,
            dtype=torch.float32,
        )
    )

    translation = (
        new_origin
        - old_origin_device
    )

    # =================================================================
    # Fixed B3 dynamics are already instantiated.
    #
    # Deliberately DO NOT restore from the selected RL robot:
    #   mass / inertia
    #   arm length
    #   motor constants
    #   attitude gains
    #   mixer geometry
    #   PhysX mass / inertia
    #
    # We only replay the transfer-onset physical state below.
    # =================================================================

    print()
    print("=" * 100)
    print("MELLINGER FIXED REAL-B3 PHYSICAL PLANT")
    print("=" * 100)
    print("mass       :", B3_PLANT_MASS_KG, "kg")
    print(
        "inertia    :",
        [B3_PLANT_IXX, B3_PLANT_IYY, B3_PLANT_IZZ],
    )
    print("arm length :", B3_PLANT_ARM_M, "m")
    print("k_eta      :", B3_PLANT_K_ETA)
    print("k_m        :", B3_PLANT_K_M)
    print("k_torque   :", B3_PLANT_K_TORQUE)
    print("tau_m      :", B3_PLANT_TAU_M)
    print(
        "PhysX mass :",
        env._robot.root_physx_view.get_masses().detach().cpu(),
    )
    print(
        "PhysX inertia:",
        env._robot.root_physx_view.get_inertias().detach().cpu(),
    )

    # =================================================================
    # B3_EXACT_PHYSX_MASS_V1
    #
    # Centered equivalent rigid-body representation:
    #   body = essentially the full 0.046 kg vehicle mass
    #   arm/helper links = numerical epsilon mass only
    #
    # Total physical vehicle mass remains exactly 0.046 kg.
    # =================================================================
    body_names_exact_b3 = list(env._robot.body_names)

    try:
        body_idx_exact_b3 = body_names_exact_b3.index("body")
    except ValueError as exc:
        raise RuntimeError(
            "Could not find rigid body named 'body'."
        ) from exc

    # B3_CENTERED_EQUIVALENT_RIGID_BODY_V1
    #
    # Represent the known B3 total mass and aggregate inertia as a
    # single equivalent rigid body.  The visual/helper links remain,
    # but their masses are numerical epsilon only.
    #
    # This intentionally removes the unmeasured lateral aggregate-COM
    # shift produced by assigning 1 g to the offset arm link.
    helper_mass_epsilon_exact_b3 = 1.0e-13

    physx_masses_exact_b3 = (
        env._robot.root_physx_view
        .get_masses()
        .clone()
    )

    # Make every non-body link effectively massless.
    physx_masses_exact_b3[0, :] = (
        helper_mass_epsilon_exact_b3
    )

    non_body_mass_exact_b3 = (
        helper_mass_epsilon_exact_b3
        * (len(body_names_exact_b3) - 1)
    )

    desired_body_mass_exact_b3 = (
        B3_PLANT_MASS_KG
        - non_body_mass_exact_b3
    )

    if desired_body_mass_exact_b3 <= 0.0:
        raise RuntimeError(
            "Invalid centered-equivalent B3 body mass: "
            f"{desired_body_mass_exact_b3}"
        )

    physx_masses_exact_b3[
        0,
        body_idx_exact_b3,
    ] = desired_body_mass_exact_b3

    env._robot.root_physx_view.set_masses(
        physx_masses_exact_b3.cpu(),
        env_ids.cpu(),
    )

    # Read back from PhysX. Do not trust the requested value alone.
    verified_masses_exact_b3 = (
        env._robot.root_physx_view
        .get_masses()
        .detach()
        .cpu()
    )

    verified_total_mass_exact_b3 = float(
        verified_masses_exact_b3[0].sum().item()
    )

    print()
    print("=" * 100)
    print("B3 EXACT PHYSX MASS INSTALLATION")
    print("=" * 100)
    print(
        "body index          :",
        body_idx_exact_b3,
    )
    print(
        "body mass           :",
        float(
            verified_masses_exact_b3[
                0,
                body_idx_exact_b3,
            ].item()
        ),
        "kg",
    )
    print(
        "non-body mass       :",
        float(
            (
                verified_masses_exact_b3[0].sum()
                - verified_masses_exact_b3[
                    0,
                    body_idx_exact_b3,
                ]
            ).item()
        ),
        "kg",
    )
    print(
        "total PhysX mass    :",
        verified_total_mass_exact_b3,
        "kg",
    )
    print(
        "target B3 mass      :",
        B3_PLANT_MASS_KG,
        "kg",
    )
    print(
        "mass error          :",
        verified_total_mass_exact_b3
        - B3_PLANT_MASS_KG,
        "kg",
    )

    if abs(
        verified_total_mass_exact_b3
        - B3_PLANT_MASS_KG
    ) > 1.0e-7:
        raise RuntimeError(
            "Failed to install exact B3 PhysX total mass."
        )

    print("[PASS] Exact B3 PhysX total mass installed.")
    print("=" * 100)

    # -----------------------------------------------------------------
    # Geometry dump for exact aggregate-inertia calculation.
    #
    # body_com_pos_w and body_com_quat_w are per-rigid-body COM
    # position/orientation from the initialized PhysX articulation.
    # -----------------------------------------------------------------
    body_com_pos_exact_b3 = (
        env._robot.data.body_com_pos_w[0]
        .detach()
        .cpu()
    )

    body_com_quat_exact_b3 = (
        env._robot.data.body_com_quat_w[0]
        .detach()
        .cpu()
    )

    body_com_pos_b_exact_b3 = (
        env._robot.data.body_com_pos_b[0]
        .detach()
        .cpu()
    )

    body_com_quat_b_exact_b3 = (
        env._robot.data.body_com_quat_b[0]
        .detach()
        .cpu()
    )

    print()
    print("B3 RIGID-BODY COM GEOMETRY AUDIT")
    print("-" * 100)

    for idx_exact_b3, name_exact_b3 in enumerate(
        body_names_exact_b3
    ):
        print(
            f"{idx_exact_b3:2d}  "
            f"{name_exact_b3:20s}  "
            f"mass="
            f"{float(verified_masses_exact_b3[0, idx_exact_b3]):.9f}"
        )
        print(
            "    COM pos world :",
            body_com_pos_exact_b3[
                idx_exact_b3
            ].numpy(),
        )
        print(
            "    COM quat world:",
            body_com_quat_exact_b3[
                idx_exact_b3
            ].numpy(),
        )
        print(
            "    COM pos link  :",
            body_com_pos_b_exact_b3[
                idx_exact_b3
            ].numpy(),
        )
        print(
            "    COM quat link :",
            body_com_quat_b_exact_b3[
                idx_exact_b3
            ].numpy(),
        )

    print("-" * 100)

    # =================================================================
    # B3_EXACT_AGGREGATE_INERTIA_V1
    #
    # Match the TOTAL fixed vehicle inertia about the aggregate COM:
    #
    #   Ixx = 2.4255e-05 kg m^2
    #   Iyy = 1.8650e-05 kg m^2
    #   Izz = 3.9300e-05 kg m^2
    #
    # The arm remains a separate 1 g fixed rigid body. Therefore the
    # main-body inertia is solved so that:
    #
    #   intrinsic inertias
    #   + all parallel-axis terms
    #   = measured whole-B3 inertia.
    # =================================================================

    def _quat_wxyz_to_rotmat_b3(q):
        q = q.to(dtype=torch.float64)
        q = q / torch.linalg.norm(q)

        w, x, y, z = q

        return torch.stack([
            torch.stack([
                1.0 - 2.0 * (y*y + z*z),
                2.0 * (x*y - z*w),
                2.0 * (x*z + y*w),
            ]),
            torch.stack([
                2.0 * (x*y + z*w),
                1.0 - 2.0 * (x*x + z*z),
                2.0 * (y*z - x*w),
            ]),
            torch.stack([
                2.0 * (x*z - y*w),
                2.0 * (y*z + x*w),
                1.0 - 2.0 * (x*x + y*y),
            ]),
        ])

    body_names_inertia_b3 = list(
        env._robot.body_names
    )

    body_idx_inertia_b3 = (
        body_names_inertia_b3.index("body")
    )

    masses_inertia_b3 = (
        env._robot.root_physx_view
        .get_masses()[0]
        .detach()
        .to(
            device=device,
            dtype=torch.float64,
        )
    )

    inertias_inertia_b3 = (
        env._robot.root_physx_view
        .get_inertias()[0]
        .detach()
        .to(
            device=device,
            dtype=torch.float64,
        )
        .reshape(-1, 3, 3)
    )

    com_pos_w_inertia_b3 = (
        env._robot.data.body_com_pos_w[0]
        .detach()
        .to(
            device=device,
            dtype=torch.float64,
        )
    )

    body_quat_w_inertia_b3 = (
        env._robot.data.body_quat_w[0]
        .detach()
        .to(
            device=device,
            dtype=torch.float64,
        )
    )

    total_mass_inertia_b3 = (
        masses_inertia_b3.sum()
    )

    aggregate_com_w_inertia_b3 = (
        (
            masses_inertia_b3[:, None]
            * com_pos_w_inertia_b3
        ).sum(dim=0)
        / total_mass_inertia_b3
    )

    R_ref_inertia_b3 = (
        _quat_wxyz_to_rotmat_b3(
            body_quat_w_inertia_b3[
                body_idx_inertia_b3
            ]
        )
    )

    target_inertia_ref_b3 = torch.diag(
        torch.tensor(
            [
                B3_PLANT_IXX,
                B3_PLANT_IYY,
                B3_PLANT_IZZ,
            ],
            dtype=torch.float64,
            device=device,
        )
    )

    # Express target vehicle inertia in world coordinates.
    target_inertia_w_b3 = (
        R_ref_inertia_b3
        @ target_inertia_ref_b3
        @ R_ref_inertia_b3.T
    )

    parallel_sum_w_b3 = torch.zeros(
        (3, 3),
        dtype=torch.float64,
        device=device,
    )

    other_intrinsic_sum_w_b3 = torch.zeros(
        (3, 3),
        dtype=torch.float64,
        device=device,
    )

    eye3_b3 = torch.eye(
        3,
        dtype=torch.float64,
        device=device,
    )

    print()
    print("=" * 100)
    print("B3 AGGREGATE-INERTIA SOLVE")
    print("=" * 100)
    print(
        "aggregate COM world :",
        aggregate_com_w_inertia_b3
        .detach()
        .cpu()
        .numpy(),
    )

    for idx_inertia_b3 in range(
        len(body_names_inertia_b3)
    ):
        mass_i_b3 = masses_inertia_b3[
            idx_inertia_b3
        ]

        d_i_b3 = (
            com_pos_w_inertia_b3[
                idx_inertia_b3
            ]
            - aggregate_com_w_inertia_b3
        )

        parallel_i_b3 = (
            mass_i_b3
            * (
                torch.dot(d_i_b3, d_i_b3)
                * eye3_b3
                - torch.outer(d_i_b3, d_i_b3)
            )
        )

        parallel_sum_w_b3 += parallel_i_b3

        if idx_inertia_b3 != body_idx_inertia_b3:
            R_i_b3 = _quat_wxyz_to_rotmat_b3(
                body_quat_w_inertia_b3[
                    idx_inertia_b3
                ]
            )

            I_i_w_b3 = (
                R_i_b3
                @ inertias_inertia_b3[
                    idx_inertia_b3
                ]
                @ R_i_b3.T
            )

            other_intrinsic_sum_w_b3 += (
                I_i_w_b3
            )

        print(
            f"{idx_inertia_b3:2d} "
            f"{body_names_inertia_b3[idx_inertia_b3]:20s} "
            f"m={float(mass_i_b3):.9f} "
            f"d={d_i_b3.detach().cpu().numpy()}"
        )

    # The body's own parallel-axis contribution is already included
    # in parallel_sum_w_b3. Solve only for its intrinsic inertia.
    desired_body_inertia_w_b3 = (
        target_inertia_w_b3
        - parallel_sum_w_b3
        - other_intrinsic_sum_w_b3
    )

    desired_body_inertia_local_b3 = (
        R_ref_inertia_b3.T
        @ desired_body_inertia_w_b3
        @ R_ref_inertia_b3
    )

    # Numerical symmetry cleanup.
    desired_body_inertia_local_b3 = (
        0.5
        * (
            desired_body_inertia_local_b3
            + desired_body_inertia_local_b3.T
        )
    )

    body_inertia_eigs_b3 = torch.linalg.eigvalsh(
        desired_body_inertia_local_b3
    )

    print()
    print("target aggregate inertia:")
    print(
        target_inertia_ref_b3
        .detach()
        .cpu()
        .numpy()
    )

    print()
    print("total parallel-axis contribution:")
    print(
        (
            R_ref_inertia_b3.T
            @ parallel_sum_w_b3
            @ R_ref_inertia_b3
        )
        .detach()
        .cpu()
        .numpy()
    )

    print()
    print("required main-body intrinsic inertia:")
    print(
        desired_body_inertia_local_b3
        .detach()
        .cpu()
        .numpy()
    )

    print(
        "required-body eigenvalues:",
        body_inertia_eigs_b3
        .detach()
        .cpu()
        .numpy(),
    )

    if bool(
        torch.any(
            body_inertia_eigs_b3 <= 0.0
        ).item()
    ):
        raise RuntimeError(
            "Solved B3 body inertia is not positive definite."
        )

    # -------------------------------------------------------------
    # B3_PRINCIPAL_AXIS_INERTIA_ENCODING_V1
    #
    # PhysX stores rigid-body inertia as principal moments plus the
    # orientation of the principal-inertia / COM frame. Therefore,
    # diagonalize the solved dense body tensor and encode:
    #
    #   eigenvalues  -> diagonal PhysX inertia
    #   eigenvectors -> COM/principal-axis orientation
    # -------------------------------------------------------------

    principal_values_raw_b3, principal_axes_raw_b3 = (
        torch.linalg.eigh(
            desired_body_inertia_local_b3
        )
    )

    # -------------------------------------------------------------
    # Reorder the principal axes to remain as close as possible to
    # the original body x/y/z axes.
    #
    # The eigenvalues are distinct here, so map each one to the
    # closest desired x/y/z vehicle moment.
    # -------------------------------------------------------------
    target_diag_b3 = torch.diag(
        target_inertia_ref_b3
    )

    available_indices_b3 = [0, 1, 2]
    ordered_indices_b3 = []

    for target_axis_b3 in range(3):
        best_index_b3 = min(
            available_indices_b3,
            key=lambda idx: abs(
                float(
                    principal_values_raw_b3[idx].item()
                    - target_diag_b3[target_axis_b3].item()
                )
            ),
        )

        ordered_indices_b3.append(
            best_index_b3
        )
        available_indices_b3.remove(
            best_index_b3
        )

    principal_values_b3 = (
        principal_values_raw_b3[
            ordered_indices_b3
        ]
    )

    principal_axes_b3 = (
        principal_axes_raw_b3[
            :,
            ordered_indices_b3
        ].clone()
    )

    # Pick eigenvector signs that keep the frame close to the original
    # body frame.
    for axis_b3 in range(3):
        if float(
            principal_axes_b3[
                axis_b3,
                axis_b3,
            ].item()
        ) < 0.0:
            principal_axes_b3[
                :,
                axis_b3,
            ] *= -1.0

    # Eigenvectors can form a reflection. PhysX needs a proper rotation.
    if float(
        torch.linalg.det(
            principal_axes_b3
        ).item()
    ) < 0.0:
        principal_axes_b3[:, 2] *= -1.0

    principal_reconstruction_b3 = (
        principal_axes_b3
        @ torch.diag(
            principal_values_b3
        )
        @ principal_axes_b3.T
    )

    principal_reconstruction_error_b3 = float(
        torch.max(
            torch.abs(
                principal_reconstruction_b3
                - desired_body_inertia_local_b3
            )
        ).item()
    )

    if principal_reconstruction_error_b3 > 1.0e-12:
        raise RuntimeError(
            "Principal-axis decomposition failed: "
            f"error={principal_reconstruction_error_b3:.9e}"
        )

    # -------------------------------------------------------------
    # Rotation-matrix -> quaternion WXYZ.
    # Only one matrix is converted, so an explicit branch is clearer
    # and avoids introducing another dependency.
    # -------------------------------------------------------------
    def _rotmat_to_quat_wxyz_b3(R):
        R = R.to(dtype=torch.float64)

        m00 = R[0, 0]
        m01 = R[0, 1]
        m02 = R[0, 2]
        m10 = R[1, 0]
        m11 = R[1, 1]
        m12 = R[1, 2]
        m20 = R[2, 0]
        m21 = R[2, 1]
        m22 = R[2, 2]

        trace = m00 + m11 + m22

        if float(trace.item()) > 0.0:
            S = torch.sqrt(
                trace + 1.0
            ) * 2.0

            qw = 0.25 * S
            qx = (m21 - m12) / S
            qy = (m02 - m20) / S
            qz = (m10 - m01) / S

        elif (
            float(m00.item()) > float(m11.item())
            and float(m00.item()) > float(m22.item())
        ):
            S = torch.sqrt(
                1.0 + m00 - m11 - m22
            ) * 2.0

            qw = (m21 - m12) / S
            qx = 0.25 * S
            qy = (m01 + m10) / S
            qz = (m02 + m20) / S

        elif float(m11.item()) > float(m22.item()):
            S = torch.sqrt(
                1.0 + m11 - m00 - m22
            ) * 2.0

            qw = (m02 - m20) / S
            qx = (m01 + m10) / S
            qy = 0.25 * S
            qz = (m12 + m21) / S

        else:
            S = torch.sqrt(
                1.0 + m22 - m00 - m11
            ) * 2.0

            qw = (m10 - m01) / S
            qx = (m02 + m20) / S
            qy = (m12 + m21) / S
            qz = 0.25 * S

        q = torch.stack(
            [qw, qx, qy, qz]
        )

        return q / torch.linalg.norm(q)

    principal_quat_wxyz_b3 = (
        _rotmat_to_quat_wxyz_b3(
            principal_axes_b3
        )
    )

    # -------------------------------------------------------------
    # root_physx_view.get_coms()/set_coms() uses the raw PhysX tensor
    # representation:
    #
    #   [px, py, pz, qx, qy, qz, qw]
    #
    # Isaac Lab converts this to WXYZ when exposing body_com_quat_b.
    # -------------------------------------------------------------
    principal_quat_xyzw_b3 = torch.stack([
        principal_quat_wxyz_b3[1],
        principal_quat_wxyz_b3[2],
        principal_quat_wxyz_b3[3],
        principal_quat_wxyz_b3[0],
    ])

    current_coms_b3 = (
        env._robot.root_physx_view
        .get_coms()
        .clone()
    )

    # Keep the existing COM POSITION exactly unchanged.
    # Change only the orientation of the body principal-inertia frame.
    current_coms_b3[
        0,
        body_idx_inertia_b3,
        3:7,
    ] = principal_quat_xyzw_b3.to(
        dtype=current_coms_b3.dtype,
        device=current_coms_b3.device,
    )

    env._robot.root_physx_view.set_coms(
        current_coms_b3.cpu(),
        env_ids.cpu(),
    )

    # Install only the principal moments on the PhysX inertia diagonal.
    current_physx_inertias_b3 = (
        env._robot.root_physx_view
        .get_inertias()
        .clone()
    )

    principal_matrix_b3 = torch.diag(
        principal_values_b3
    )

    current_physx_inertias_b3[
        0,
        body_idx_inertia_b3,
        :,
    ] = (
        principal_matrix_b3
        .reshape(9)
        .to(
            dtype=current_physx_inertias_b3.dtype,
            device=current_physx_inertias_b3.device,
        )
    )

    env._robot.root_physx_view.set_inertias(
        current_physx_inertias_b3.cpu(),
        env_ids.cpu(),
    )

    print()
    print("B3 PHYSX PRINCIPAL-AXIS ENCODING")
    print("-" * 100)
    print(
        "principal moments:",
        principal_values_b3
        .detach()
        .cpu()
        .numpy(),
    )
    print(
        "principal axes relative to body:"
    )
    print(
        principal_axes_b3
        .detach()
        .cpu()
        .numpy()
    )
    print(
        "principal quaternion WXYZ:",
        principal_quat_wxyz_b3
        .detach()
        .cpu()
        .numpy(),
    )
    print(
        "decomposition error:",
        principal_reconstruction_error_b3,
    )
    print("-" * 100)

    # -------------------------------------------------------------
    # Read back and independently recompute aggregate vehicle inertia.
    # -------------------------------------------------------------
    verified_inertias_b3 = (
        env._robot.root_physx_view
        .get_inertias()[0]
        .detach()
        .to(
            device=device,
            dtype=torch.float64,
        )
        .reshape(-1, 3, 3)
    )

    print()
    print("B3 PHYSX INERTIA READBACK")
    print("-" * 100)
    print("main-body tensor returned by get_inertias():")
    print(
        verified_inertias_b3[
            body_idx_inertia_b3
        ]
        .detach()
        .cpu()
        .numpy()
    )
    print("desired main-body tensor:")
    print(
        desired_body_inertia_local_b3
        .detach()
        .cpu()
        .numpy()
    )
    print(
        "main-body readback max error:",
        float(
            torch.max(
                torch.abs(
                    verified_inertias_b3[
                        body_idx_inertia_b3
                    ]
                    - desired_body_inertia_local_b3
                )
            ).item()
        ),
    )
    print("-" * 100)

    verified_aggregate_w_b3 = torch.zeros(
        (3, 3),
        dtype=torch.float64,
        device=device,
    )

    verified_coms_raw_b3 = (
        env._robot.root_physx_view
        .get_coms()[0]
        .detach()
        .to(
            device=device,
            dtype=torch.float64,
        )
    )

    for idx_inertia_b3 in range(
        len(body_names_inertia_b3)
    ):
        # get_inertias() already returns the full inertia tensor
        # about the COM expressed in the rigid body's ACTOR/LINK frame.
        #
        # Do NOT apply the PhysX principal-axis/COM quaternion again here.
        # Doing so double-rotates the tensor.
        R_link_w_b3 = _quat_wxyz_to_rotmat_b3(
            body_quat_w_inertia_b3[
                idx_inertia_b3
            ]
        )

        I_i_w_b3 = (
            R_link_w_b3
            @ verified_inertias_b3[
                idx_inertia_b3
            ]
            @ R_link_w_b3.T
        )

        d_i_b3 = (
            com_pos_w_inertia_b3[
                idx_inertia_b3
            ]
            - aggregate_com_w_inertia_b3
        )

        parallel_i_b3 = (
            masses_inertia_b3[
                idx_inertia_b3
            ]
            * (
                torch.dot(d_i_b3, d_i_b3)
                * eye3_b3
                - torch.outer(d_i_b3, d_i_b3)
            )
        )

        verified_aggregate_w_b3 += (
            I_i_w_b3
            + parallel_i_b3
        )

    verified_aggregate_ref_b3 = (
        R_ref_inertia_b3.T
        @ verified_aggregate_w_b3
        @ R_ref_inertia_b3
    )

    aggregate_inertia_error_b3 = (
        verified_aggregate_ref_b3
        - target_inertia_ref_b3
    )

    max_inertia_error_b3 = float(
        torch.max(
            torch.abs(
                aggregate_inertia_error_b3
            )
        ).item()
    )

    print()
    print("verified aggregate inertia:")
    print(
        verified_aggregate_ref_b3
        .detach()
        .cpu()
        .numpy()
    )

    print()
    print("aggregate inertia error:")
    print(
        aggregate_inertia_error_b3
        .detach()
        .cpu()
        .numpy()
    )

    print(
        "max |inertia error| :",
        max_inertia_error_b3,
    )

    if max_inertia_error_b3 > 1.0e-9:
        raise RuntimeError(
            "Failed exact B3 aggregate inertia verification: "
            f"max error={max_inertia_error_b3:.9e}"
        )

    print(
        "[PASS] Exact aggregate B3 PhysX inertia installed."
    )
    print("=" * 100)

    # =================================================================
    # B3 PHYSX BODY AUDIT
    # =================================================================
    body_names_b3 = list(
        getattr(env._robot, "body_names", [])
    )

    physx_masses_b3 = (
        env._robot.root_physx_view
        .get_masses()[0]
        .detach()
        .cpu()
    )

    physx_inertias_b3 = (
        env._robot.root_physx_view
        .get_inertias()[0]
        .detach()
        .cpu()
    )

    total_physx_mass_b3 = float(
        physx_masses_b3.sum().item()
    )

    print()
    print("B3 PHYSX BODY AUDIT")

    # B3_RAW_PHYSX_DUMP_V1
    raw_physx_masses_b3 = (
        env._robot.root_physx_view
        .get_masses()
        .detach()
        .cpu()
    )

    raw_physx_inertias_b3 = (
        env._robot.root_physx_view
        .get_inertias()
        .detach()
        .cpu()
    )

    print("AUDIT SOURCE FILE :", __file__)
    print("robot body_names  :", list(getattr(env._robot, "body_names", [])))
    print("raw masses shape  :", tuple(raw_physx_masses_b3.shape))
    print("raw masses numel  :", raw_physx_masses_b3.numel())
    print("raw masses tensor :")
    print(raw_physx_masses_b3)
    print("raw inertia shape :", tuple(raw_physx_inertias_b3.shape))
    print("raw inertia numel :", raw_physx_inertias_b3.numel())
    print("raw inertia tensor:")
    print(raw_physx_inertias_b3)

    print(
        "RAW TOTAL MASS    :",
        float(raw_physx_masses_b3.sum().item()),
        "kg",
    )
    print(
        "TARGET B3 MASS    :",
        B3_PLANT_MASS_KG,
        "kg",
    )
    print("-" * 100)
    print("body_names        :", body_names_b3)
    print("mass tensor shape :", tuple(physx_masses_b3.shape))
    print("inertia shape     :", tuple(physx_inertias_b3.shape))

    for body_idx in range(physx_masses_b3.shape[0]):
        if body_idx < len(body_names_b3):
            body_name = body_names_b3[body_idx]
        else:
            body_name = f"rigid_body_{body_idx}"

        print(
            f"{body_idx:2d}  "
            f"{body_name:35s}  "
            f"mass={float(physx_masses_b3[body_idx]):.9f} kg"
        )
        print(
            "    inertia=",
            physx_inertias_b3[body_idx].numpy(),
        )

    print("-" * 100)
    print(
        "PhysX total mass :",
        f"{total_physx_mass_b3:.9f}",
        "kg",
    )
    print(
        "Expected B3 mass :",
        f"{B3_PLANT_MASS_KG:.9f}",
        "kg",
    )
    print(
        "Total mass error :",
        f"{total_physx_mass_b3 - B3_PLANT_MASS_KG:+.9f}",
        "kg",
    )
    print("-" * 100)

    print("=" * 100)

    # =================================================================
    # Restore exact physical initial state.
    # =================================================================
    root_state = (
        initial["root_state_w"]
        .to(
            device=device,
            dtype=torch.float32,
        )
        .clone()
        .reshape(13)
    )

    root_state[0:3] += translation

    env._robot.write_root_pose_to_sim(
        root_state[0:7].view(1, 7),
        env_ids=env_ids,
    )

    env._robot.write_root_velocity_to_sim(
        root_state[7:13].view(1, 6),
        env_ids=env_ids,
    )

    env.episode_length_buf[0] = 0

    # =================================================================
    # Exact physical motor state + exact physical latency duration.
    #
    # We intentionally DO NOT transplant RL's CTBR queue contents into
    # Mellinger's SRT queue because they are different control spaces.
    #
    # Instead the new queue is initially filled with the SRT command
    # corresponding to the exact captured physical motor state. Thus
    # the vehicle begins with the same motors and remains continuous
    # while the first Mellinger command traverses the same physical
    # latency.
    # =================================================================
    captured_motor_state = (
        actuator["motor_speeds"]
        .to(
            device=device,
            dtype=torch.float32,
        )
        .reshape(4)
    )

    if (
        float(captured_motor_state.min()) < -1.0e-4
        or float(captured_motor_state.max()) > 1.0001
    ):
        raise RuntimeError(
            "Captured motor state is not normalized [0,1], "
            "so the expected Crazyflie SRT mapping is invalid."
        )

    env._motor_speeds[0].copy_(
        captured_motor_state
    )

    env._motor_speeds_des[0].copy_(
        captured_motor_state
    )

    initial_srt_action = (
        2.0 * captured_motor_state
        - 1.0
    )

    queue_length = int(
        env._action_queue.shape[0]
    )

    expected_queue_length = (
        mellinger_delay_steps + 1
    )

    if queue_length != expected_queue_length:
        raise RuntimeError(
            "Unexpected Mellinger action queue length: "
            f"expected={expected_queue_length}, "
            f"actual={queue_length}"
        )

    env._action_queue[:, 0, :] = (
        initial_srt_action
        .view(1, 4)
        .repeat(
            queue_length,
            1,
        )
    )

    env._actions[0].copy_(
        initial_srt_action
    )

    if hasattr(
        env,
        "_previous_action",
    ):
        env._previous_action[0].copy_(
            initial_srt_action
        )

    if hasattr(
        env,
        "_action_history",
    ):
        env._action_history[0] = (
            initial_srt_action
            .view(1, 4)
            .repeat(
                env._action_history.shape[1],
                1,
            )
        )

    # queue[-1] is newest. With queue length delay+1, index 0 gives the
    # requested physical command age.
    env._control_latency_steps[0] = 0

    realized_delay_steps = (
        queue_length - 1
    )

    realized_delay_s = (
        realized_delay_steps
        / MELLINGER_RATE_HZ
    )

    if abs(
        realized_delay_s
        - physical_delay_s
    ) > 1.0e-6:
        raise RuntimeError(
            "Physical latency replay mismatch: "
            f"RL={physical_delay_s:.9f}s, "
            f"Mellinger={realized_delay_s:.9f}s"
        )

    # -----------------------------------------------------------------
    # Invalidate Isaac lazy buffers after direct root writes.
    # -----------------------------------------------------------------
    robot_data = env._robot.data

    for buffer_name in (
        "_body_com_vel_w",
        "_body_link_vel_w",
        "_body_state_w",
        "_body_link_state_w",
        "_body_com_state_w",
    ):
        buffer = getattr(
            robot_data,
            buffer_name,
            None,
        )

        if buffer is not None:
            buffer.timestamp = -1.0

    obs_dict = env._get_observations()

    if obs_dict.get("gc") is None:
        raise RuntimeError(
            "Mellinger replay environment did not provide GC observation."
        )

    # =================================================================
    # Validate that the new process really represents the captured case.
    # =================================================================
    body_pos, body_quat, body_vel, body_ang = (
        env.get_frame_state_from_task(
            "body"
        )
    )

    expected_body_pos = (
        initial["body_pos_w"]
        .to(
            device=device,
            dtype=torch.float32,
        )
        .reshape(3)
        + translation
    )

    expected_body_quat = (
        initial["body_quat_w"]
        .to(
            device=device,
            dtype=torch.float32,
        )
        .reshape(4)
    )

    expected_body_vel = (
        initial["body_lin_vel_w"]
        .to(
            device=device,
            dtype=torch.float32,
        )
        .reshape(3)
    )

    expected_body_ang = (
        initial["body_ang_vel_w"]
        .to(
            device=device,
            dtype=torch.float32,
        )
        .reshape(3)
    )

    pos_error = float(
        torch.max(
            torch.abs(
                body_pos[0]
                - expected_body_pos
            )
        ).item()
    )

    quat_error = quaternion_error(
        body_quat[0],
        expected_body_quat,
    )

    vel_error = float(
        torch.max(
            torch.abs(
                body_vel[0]
                - expected_body_vel
            )
        ).item()
    )

    ang_error = float(
        torch.max(
            torch.abs(
                body_ang[0]
                - expected_body_ang
            )
        ).item()
    )

    gc_goal = (
        obs_dict["gc"][0, 13:16]
    )

    expected_gc_goal = (
        goal["mellinger_goal_pos_w"]
        .to(
            device=device,
            dtype=torch.float32,
        )
        .reshape(3)
        + translation
    )

    gc_goal_error = float(
        torch.max(
            torch.abs(
                gc_goal
                - expected_gc_goal
            )
        ).item()
    )

    desired_goal_error = float(
        torch.max(
            torch.abs(
                env._desired_pos_w[0]
                - (
                    desired_pos_original
                    .to(
                        device=device,
                        dtype=torch.float32,
                    )
                    + translation
                )
            )
        ).item()
    )

    print()
    print("=" * 100)
    print("EXACT RL -> MELLINGER REPLAY VALIDATION")
    print("=" * 100)
    print("original RL robot index :", robot_index)
    print("RL policy rate          :", rl_rate_hz, "Hz")
    print("Mellinger rate          :", MELLINGER_RATE_HZ, "Hz")
    print("physical duration       :", physical_duration_s, "s")
    print("RL physical latency     :", physical_delay_s, "s")
    print("Mellinger latency       :", realized_delay_s, "s")
    print("initial body pos error  :", f"{pos_error:.9e}")
    print("initial body quat error :", f"{quat_error:.9e}")
    print("initial body vel error  :", f"{vel_error:.9e}")
    print("initial body ang error  :", f"{ang_error:.9e}")
    print("GC goal error           :", f"{gc_goal_error:.9e}")
    print("task goal error         :", f"{desired_goal_error:.9e}")
    print("=" * 100)

    for name, value, tolerance in (
        ("body position", pos_error, 1.0e-5),
        ("body quaternion", quat_error, 1.0e-5),
        ("body velocity", vel_error, 1.0e-5),
        ("body angular velocity", ang_error, 1.0e-5),
        ("Mellinger GC goal", gc_goal_error, 1.0e-5),
        ("task goal", desired_goal_error, 1.0e-5),
    ):
        if value > tolerance:
            raise RuntimeError(
                f"Replay validation failed for {name}: "
                f"error={value:.9e}"
            )

    # =================================================================
    # Frozen B3-faithful firmware-style Mellinger controller.
    #
    # IMPORTANT:
    # The PHYSICAL plant above is the fixed real-B3 plant.
    # The controller calibration below is also fixed B3.
    # No randomized RL dynamics are provided to Mellinger.
    # =================================================================
    B3_CONTROLLER_MASS_KG = 0.046
    B3_CONTROLLER_K_ETA = 0.51033
    B3_CONTROLLER_MASS_THRUST = 132000.0
    B3_CONTROLLER_IDLE_THRUST = 10000.0
    B3_CONTROLLER_KD_OMEGA_RP = 200.0

    controller = CrazyflieFirmwareMellinger(
        mass_kg=B3_CONTROLLER_MASS_KG,
        mass_thrust=B3_CONTROLLER_MASS_THRUST,
    )

    # Explicit deployment calibration.
    controller.controller.kd_omega_rp = (
        B3_CONTROLLER_KD_OMEGA_RP
    )

    print(
        "Mellinger calibration  : ACTUAL BITCRAZE FIRMWARE SIL"
    )
    print(
        "controller mass        :",
        B3_CONTROLLER_MASS_KG,
        "kg",
    )
    print(
        "controller massThrust  :",
        B3_CONTROLLER_MASS_THRUST,
    )
    print(
        "controller idleThrust  :",
        B3_CONTROLLER_IDLE_THRUST,
    )
    print(
        "controller kd_omega_rp :",
        B3_CONTROLLER_KD_OMEGA_RP,
    )
    print(
        "firmware commit         :",
        FIRMWARE_COMMIT,
    )

    # =================================================================
    # B3_SYSID_ACTUATOR_MODEL_V1
    #
    # Comparator-only physical actuator model identified from real B3.
    #
    # Firmware output:
    #     motor.m1...m4 -> normalized command u_des in [0,1]
    #
    # Motor dynamics:
    #     tau_m = 0.050 s
    #
    # Per-motor thrust:
    #     F(u) = 0.187453678*u + 0.126414663*u^2   [N]
    #
    # The existing f_to_TM matrix is retained for B3 rotor geometry
    # and reaction torque. RL/environment source remains unchanged.
    # =================================================================
    from types import MethodType

    B3_SYSID_THRUST_LINEAR_N = 0.187453678
    B3_SYSID_THRUST_QUADRATIC_N = 0.126414663
    B3_SYSID_TAU_M_S = 0.050

    if env.cfg.control_mode != "SRT":
        raise RuntimeError(
            "B3 sysid actuator override requires SRT mode; "
            f"got {env.cfg.control_mode!r}."
        )

    if bool(env.cfg.skip_motor_dynamics):
        raise RuntimeError(
            "B3 sysid actuator requires motor dynamics enabled."
        )

    installed_tau_m = float(
        env._tau_m.reshape(-1)[0]
        .detach()
        .cpu()
        .item()
    )

    if abs(installed_tau_m - B3_SYSID_TAU_M_S) > 1.0e-6:
        raise RuntimeError(
            "Unexpected B3 motor time constant: "
            f"{installed_tau_m} s"
        )

    # =================================================================
    # B3_DIFFERENTIAL_ACTUATOR_ABLATION_V1
    #
    # EXPERIMENTAL / NON-DEFAULT physical-model ablation.
    #
    # Baseline remains EXACTLY the existing model when:
    #
    #     B3_DIFF_ACTUATOR_ENABLE=0
    #
    # Experimental split:
    #
    #   firmware motor command
    #       -> collective component
    #            existing tau_m = 50 ms
    #
    #       -> zero-mean differential component
    #            configurable delay
    #            configurable tau_diff
    #
    # Roll/pitch moment gains are applied AFTER the existing measured
    # per-motor thrust curve and existing f_to_TM mapping.
    #
    # Therefore:
    #   - collective thrust is never multiplied by the fitted alpha;
    #   - the measured B3 thrust law remains unchanged;
    #   - yaw moment is currently left unchanged;
    #   - this is an identification ablation, not a new default model.
    # =================================================================
    import os

    B3_DIFF_ACTUATOR_ENABLE = (
        os.environ.get(
            "B3_DIFF_ACTUATOR_ENABLE",
            "0",
        ).strip() == "1"
    )

    B3_DIFF_TAU_M_S = float(
        os.environ.get(
            "B3_DIFF_TAU_M_S",
            str(B3_SYSID_TAU_M_S),
        )
    )

    B3_DIFF_DELAY_S = float(
        os.environ.get(
            "B3_DIFF_DELAY_S",
            "0.0",
        )
    )

    B3_DIFF_ROLL_GAIN = float(
        os.environ.get(
            "B3_DIFF_ROLL_GAIN",
            "1.0",
        )
    )

    B3_DIFF_PITCH_GAIN = float(
        os.environ.get(
            "B3_DIFF_PITCH_GAIN",
            "1.0",
        )
    )

    # B3_UNIFIED_ACTUATOR_SYSID_V1
    #
    # Comparator-only actuator model matching the offline physical-ID model:
    #
    #   capped firmware PWM (all four motors)
    #       -> common pure delay
    #       -> common first-order lag
    #       -> measured nonlinear thrust law
    #       -> geometry/wrench map
    #       -> shared roll/pitch moment gain
    #
    # This is deliberately separate from B3_DIFFERENTIAL_ACTUATOR_ABLATION_V1.
    B3_UNIFIED_ACTUATOR_ENABLE = (
        os.environ.get(
            "B3_UNIFIED_ACTUATOR_ENABLE",
            "0",
        ).strip().lower()
        in {"1", "true", "yes", "on"}
    )

    B3_UNIFIED_TAU_M_S = float(
        os.environ.get(
            "B3_UNIFIED_TAU_M_S",
            "0.010",
        )
    )

    B3_UNIFIED_DELAY_S = float(
        os.environ.get(
            "B3_UNIFIED_DELAY_S",
            "0.0",
        )
    )

    B3_UNIFIED_RP_GAIN = float(
        os.environ.get(
            "B3_UNIFIED_RP_GAIN",
            "1.0",
        )
    )

    if (
        B3_UNIFIED_ACTUATOR_ENABLE
        and B3_DIFF_ACTUATOR_ENABLE
    ):
        raise RuntimeError(
            "B3_UNIFIED_ACTUATOR_ENABLE and "
            "B3_DIFF_ACTUATOR_ENABLE are mutually exclusive."
        )

    # B3_UNIFIED_SECOND_POLE_V1
    B3_UNIFIED_TAU_M2_S = float(
        os.environ.get(
            "B3_UNIFIED_TAU_M2_S",
            "0.0",
        )
    )

    if B3_UNIFIED_TAU_M2_S < 0.0:
        raise ValueError(
            "B3_UNIFIED_TAU_M2_S must be >= 0."
        )

    if B3_UNIFIED_TAU_M_S <= 0.0:
        raise RuntimeError(
            "B3_UNIFIED_TAU_M_S must be positive."
        )

    if B3_UNIFIED_DELAY_S < 0.0:
        raise RuntimeError(
            "B3_UNIFIED_DELAY_S must be non-negative."
        )

    if B3_UNIFIED_RP_GAIN < 0.0:
        raise RuntimeError(
            "B3_UNIFIED_RP_GAIN must be non-negative."
        )

    if B3_DIFF_TAU_M_S <= 0.0:
        raise RuntimeError(
            "B3_DIFF_TAU_M_S must be positive."
        )

    if B3_DIFF_DELAY_S < 0.0:
        raise RuntimeError(
            "B3_DIFF_DELAY_S must be non-negative."
        )

    if B3_DIFF_ROLL_GAIN < 0.0:
        raise RuntimeError(
            "B3_DIFF_ROLL_GAIN must be non-negative."
        )

    if B3_DIFF_PITCH_GAIN < 0.0:
        raise RuntimeError(
            "B3_DIFF_PITCH_GAIN must be non-negative."
        )

    _b3_physics_dt_s = float(env.physics_dt)

    _b3_unified_delay_steps = int(
        round(
            B3_UNIFIED_DELAY_S
            / _b3_physics_dt_s
        )
    )

    _b3_unified_realized_delay_s = (
        _b3_unified_delay_steps
        * _b3_physics_dt_s
    )

    _b3_diff_delay_steps = int(
        round(
            B3_DIFF_DELAY_S
            / _b3_physics_dt_s
        )
    )

    _b3_diff_realized_delay_s = (
        _b3_diff_delay_steps
        * _b3_physics_dt_s
    )

    env._b3_diff_actuator_enabled = (
        B3_DIFF_ACTUATOR_ENABLE
    )

    env._b3_diff_tau_m_s = (
        B3_DIFF_TAU_M_S
    )

    env._b3_diff_delay_s = (
        B3_DIFF_DELAY_S
    )

    env._b3_diff_realized_delay_s = (
        _b3_diff_realized_delay_s
    )

    env._b3_diff_delay_steps = (
        _b3_diff_delay_steps
    )

    env._b3_diff_roll_gain = (
        B3_DIFF_ROLL_GAIN
    )

    env._b3_diff_pitch_gain = (
        B3_DIFF_PITCH_GAIN
    )

    # B3_UNIFIED_ACTUATOR_SYSID_V1
    env._b3_unified_actuator_enabled = (
        B3_UNIFIED_ACTUATOR_ENABLE
    )

    env._b3_unified_tau_m_s = (
        B3_UNIFIED_TAU_M_S
    )

    env._b3_unified_tau_m2_s = (
        B3_UNIFIED_TAU_M2_S
    )

    env._b3_unified_delay_s = (
        B3_UNIFIED_DELAY_S
    )

    env._b3_unified_realized_delay_s = (
        _b3_unified_realized_delay_s
    )

    env._b3_unified_delay_steps = (
        _b3_unified_delay_steps
    )

    env._b3_unified_rp_gain = (
        B3_UNIFIED_RP_GAIN
    )

    # The unified state represents the physical normalized motor state
    # for all four motors directly.
    env._b3_unified_motor_state = (
        env._motor_speeds
        .detach()
        .clone()
    )

    _b3_initial_unified_des = (
        env._motor_speeds_des
        .detach()
        .clone()
    )

    if _b3_unified_delay_steps > 0:
        env._b3_unified_delay_buffer = (
            _b3_initial_unified_des
            .unsqueeze(0)
            .repeat(
                _b3_unified_delay_steps,
                1,
                1,
            )
            .clone()
        )
    else:
        env._b3_unified_delay_buffer = None

    env._b3_unified_delay_index = 0

    # Initialize the split state from the actuator state that already
    # exists at installation time.
    _b3_initial_collective = (
        env._motor_speeds
        .mean(
            dim=-1,
            keepdim=True,
        )
        .detach()
        .clone()
    )

    _b3_initial_differential = (
        env._motor_speeds
        - _b3_initial_collective
    ).detach().clone()

    env._b3_collective_motor_state = (
        _b3_initial_collective
    )

    env._b3_diff_motor_state = (
        _b3_initial_differential
    )

    _b3_initial_des_collective = (
        env._motor_speeds_des
        .mean(
            dim=-1,
            keepdim=True,
        )
    )

    _b3_initial_des_diff = (
        env._motor_speeds_des
        - _b3_initial_des_collective
    ).detach().clone()

    if _b3_diff_delay_steps > 0:
        env._b3_diff_delay_buffer = (
            _b3_initial_des_diff
            .unsqueeze(0)
            .repeat(
                _b3_diff_delay_steps,
                1,
                1,
            )
            .clone()
        )
    else:
        env._b3_diff_delay_buffer = None

    env._b3_diff_delay_index = 0

    print()
    print("=" * 100)
    print("B3 DIFFERENTIAL ACTUATOR ABLATION")
    print("=" * 100)
    print(
        "enabled                 :",
        B3_DIFF_ACTUATOR_ENABLE,
    )
    print(
        "collective tau          :",
        B3_SYSID_TAU_M_S,
        "s",
    )
    print(
        "differential tau        :",
        B3_DIFF_TAU_M_S,
        "s",
    )
    print(
        "requested diff delay    :",
        B3_DIFF_DELAY_S,
        "s",
    )
    print(
        "realized diff delay     :",
        _b3_diff_realized_delay_s,
        "s",
        f"({_b3_diff_delay_steps} physics steps)",
    )
    print(
        "roll moment gain        :",
        B3_DIFF_ROLL_GAIN,
    )
    print(
        "pitch moment gain       :",
        B3_DIFF_PITCH_GAIN,
    )
    print(
        "yaw moment gain         : 1.0 (unchanged)"
    )
    print("=" * 100)

    print()
    print("=" * 100)
    print("B3 UNIFIED ACTUATOR SYSTEM-ID MODEL")
    print("=" * 100)
    print(
        "enabled                 :",
        B3_UNIFIED_ACTUATOR_ENABLE,
    )
    print(
        "common motor tau        :",
        B3_UNIFIED_TAU_M_S,
        "s",
    )
    print(
        "requested common delay  :",
        B3_UNIFIED_DELAY_S,
        "s",
    )
    print(
        "realized common delay   :",
        _b3_unified_realized_delay_s,
        "s",
        f"({_b3_unified_delay_steps} physics steps)",
    )
    print(
        "shared RP moment gain   :",
        B3_UNIFIED_RP_GAIN,
    )
    print(
        "yaw moment gain         : 1.0 (unchanged)"
    )
    print("=" * 100)


    def _apply_action_b3_sysid(self):
        # =============================================================
        # B3_BEHAVIOR_GEOMETRIC_DIRECT_WRENCH_V2
        #
        # Behavior-geometric mode directly commands physical body
        # wrench. PWM, motor lag, thrust-curve and SYSID machinery are
        # completely bypassed.
        # =============================================================
        _behavior_wrench = getattr(
            self,
            "_b3_behavior_direct_wrench",
            None,
        )

        if _behavior_wrench is not None:
            wrench = _behavior_wrench.to(
                device=self._thrust.device,
                dtype=self._thrust.dtype,
            )

            if wrench.ndim == 1:
                wrench = wrench.unsqueeze(0)

            if (
                wrench.ndim != 2
                or wrench.shape[-1] != 4
            ):
                raise RuntimeError(
                    "B3 behavior wrench must have shape "
                    "(num_envs, 4); got "
                    f"{tuple(wrench.shape)}"
                )

            self._thrust.zero_()
            self._moment.zero_()

            self._thrust[:, 0, 2] = wrench[:, 0]
            self._moment[:, 0, :] = wrench[:, 1:]

            self._robot.set_external_force_and_torque(
                self._thrust,
                self._moment,
                body_ids=self._body_id,
            )

            # This is the only actuator-side quantity that physically
            # exists in behavior-geometric mode.
            self._b3_sysid_last_wrench = (
                wrench.detach().clone()
            )

            return

        # Legacy non-behavior path.
        self.pd_loop_counter += 1

        unified_des_delayed = None

        # -------------------------------------------------------------
        # B3_UNIFIED_ACTUATOR_SYSID_V1
        #
        # This branch intentionally matches the offline identification
        # model literally.  No collective/differential decomposition.
        # -------------------------------------------------------------
        if self._b3_unified_actuator_enabled:
            u_des = self._motor_speeds_des

            if self._b3_unified_delay_steps > 0:
                idx = self._b3_unified_delay_index

                unified_des_delayed = (
                    self._b3_unified_delay_buffer[
                        idx
                    ]
                    .detach()
                    .clone()
                )

                self._b3_unified_delay_buffer[
                    idx
                ].copy_(u_des)

                self._b3_unified_delay_index = (
                    idx + 1
                ) % self._b3_unified_delay_steps
            else:
                unified_des_delayed = u_des

            alpha_unified = math.exp(
                -float(self.physics_dt)
                / self._b3_unified_tau_m_s
            )

            self._b3_unified_motor_state = (
                alpha_unified
                * self._b3_unified_motor_state
                + (1.0 - alpha_unified)
                * unified_des_delayed
            )

            self._b3_unified_motor_state = (
                self._b3_unified_motor_state.clamp(
                    self.cfg.motor_speed_min,
                    self.cfg.motor_speed_max,
                )
            )

            # B3_UNIFIED_SECOND_POLE_V1
            #
            # self._motor_speeds is the output/state of stage 2 from
            # the previous physics tick.  This avoids introducing a
            # separate actuator state that would need special reset
            # handling.
            if self._b3_unified_tau_m2_s > 0.0:
                alpha_unified_2 = math.exp(
                    -float(self.physics_dt)
                    / self._b3_unified_tau_m2_s
                )

                u = (
                    alpha_unified_2
                    * self._motor_speeds
                    + (1.0 - alpha_unified_2)
                    * self._b3_unified_motor_state
                )

                u = u.clamp(
                    self.cfg.motor_speed_min,
                    self.cfg.motor_speed_max,
                )
            else:
                # tau2 == 0 preserves the original unified actuator.
                u = self._b3_unified_motor_state

            self._motor_speeds = u

            diff_des_now = None
            diff_des_delayed = None

        # -------------------------------------------------------------
        # BASELINE PATH.
        #
        # Keep the pre-existing actuator equations literally intact
        # whenever the experimental split is disabled.
        # -------------------------------------------------------------
        elif not self._b3_diff_actuator_enabled:

            alpha = torch.exp(
                -self.physics_dt / self._tau_m
            ).unsqueeze(-1)

            self._motor_speeds = (
                alpha * self._motor_speeds
                + (1.0 - alpha)
                * self._motor_speeds_des
            )

            self._motor_speeds = (
                self._motor_speeds.clamp(
                    self.cfg.motor_speed_min,
                    self.cfg.motor_speed_max,
                )
            )

            u = self._motor_speeds

            diff_des_now = None
            diff_des_delayed = None

        # -------------------------------------------------------------
        # EXPERIMENTAL SPLIT PATH.
        # -------------------------------------------------------------
        else:
            u_des = self._motor_speeds_des

            collective_des = (
                u_des.mean(
                    dim=-1,
                    keepdim=True,
                )
            )

            diff_des_now = (
                u_des
                - collective_des
            )

            # Effective differential-channel delay.
            if self._b3_diff_delay_steps > 0:
                idx = self._b3_diff_delay_index

                diff_des_delayed = (
                    self._b3_diff_delay_buffer[
                        idx
                    ]
                    .detach()
                    .clone()
                )

                self._b3_diff_delay_buffer[
                    idx
                ].copy_(
                    diff_des_now
                )

                self._b3_diff_delay_index = (
                    idx + 1
                ) % self._b3_diff_delay_steps

            else:
                diff_des_delayed = (
                    diff_des_now
                )

            # Existing 50 ms collective dynamics.
            alpha_collective = torch.exp(
                -self.physics_dt / self._tau_m
            ).unsqueeze(-1)

            self._b3_collective_motor_state = (
                alpha_collective
                * self._b3_collective_motor_state
                + (1.0 - alpha_collective)
                * collective_des
            )

            # Identified differential dynamics.
            alpha_diff = math.exp(
                -float(self.physics_dt)
                / self._b3_diff_tau_m_s
            )

            self._b3_diff_motor_state = (
                alpha_diff
                * self._b3_diff_motor_state
                + (1.0 - alpha_diff)
                * diff_des_delayed
            )

            # Numerically enforce zero collective contribution from the
            # differential actuator state.
            self._b3_diff_motor_state = (
                self._b3_diff_motor_state
                - self._b3_diff_motor_state.mean(
                    dim=-1,
                    keepdim=True,
                )
            )

            u = (
                self._b3_collective_motor_state
                + self._b3_diff_motor_state
            )

            u = u.clamp(
                self.cfg.motor_speed_min,
                self.cfg.motor_speed_max,
            )

            self._motor_speeds = u

        # -------------------------------------------------------------
        # SAME measured B3 static thrust law.
        # -------------------------------------------------------------
        motor_forces = (
            B3_SYSID_THRUST_LINEAR_N * u
            + B3_SYSID_THRUST_QUADRATIC_N
            * u.square()
        )

        # SAME B3 geometry / power-to-wrench mapping.
        wrench_raw = torch.bmm(
            self.f_to_TM,
            motor_forces.unsqueeze(2),
        ).squeeze(2)

        # =============================================================
        # B3_BEHAVIOR_GEOMETRIC_DIRECT_WRENCH_V1
        #
        # Optional behavior-controller authority.
        #
        # When present, this replaces ONLY the physical wrench.
        # The existing Isaac force/moment plumbing below is reused
        # unchanged, so no new body/world force convention is created.
        # =============================================================
        _behavior_wrench = getattr(
            self,
            "_b3_behavior_direct_wrench",
            None,
        )

        if _behavior_wrench is not None:
            wrench = _behavior_wrench.to(
                device=wrench_raw.device,
                dtype=wrench_raw.dtype,
            )

            if wrench.ndim == 1:
                wrench = wrench.unsqueeze(0)

            if (
                wrench.ndim != 2
                or wrench.shape[-1] != 4
            ):
                raise RuntimeError(
                    "B3 behavior wrench must have shape "
                    "(num_envs, 4); got "
                    f"{tuple(wrench.shape)}"
                )

        else:
            wrench = wrench_raw

        # -------------------------------------------------------------
        # The fitted alpha values correspond to effective physical
        # roll/pitch torque authority.
        #
        # DO NOT touch total thrust (wrench[:,0]).
        # DO NOT touch yaw yet; it was not identified by this fit.
        # -------------------------------------------------------------
        if (
            _behavior_wrench is None
            and self._b3_unified_actuator_enabled
        ):
            wrench = wrench_raw.clone()

            wrench[:, 1] = (
                self._b3_unified_rp_gain
                * wrench_raw[:, 1]
            )

            wrench[:, 2] = (
                self._b3_unified_rp_gain
                * wrench_raw[:, 2]
            )

        elif (
            _behavior_wrench is None
            and self._b3_diff_actuator_enabled
        ):
            wrench = wrench_raw.clone()

            wrench[:, 1] = (
                self._b3_diff_roll_gain
                * wrench_raw[:, 1]
            )

            wrench[:, 2] = (
                self._b3_diff_pitch_gain
                * wrench_raw[:, 2]
            )

        self._thrust[:, 0, 2] = wrench[:, 0]
        self._moment[:, 0, :] = wrench[:, 1:]

        # Diagnostics.
        self._b3_sysid_last_u_des = (
            self._motor_speeds_des
            .detach()
            .clone()
        )

        self._b3_sysid_last_u = (
            u.detach().clone()
        )

        self._b3_sysid_last_motor_forces = (
            motor_forces.detach().clone()
        )

        self._b3_sysid_last_wrench_raw = (
            wrench_raw.detach().clone()
        )

        self._b3_sysid_last_wrench = (
            wrench.detach().clone()
        )

        if unified_des_delayed is not None:
            self._b3_sysid_last_unified_u_delayed = (
                unified_des_delayed
                .detach()
                .clone()
            )

        if diff_des_now is not None:
            self._b3_sysid_last_diff_u_des = (
                diff_des_now
                .detach()
                .clone()
            )

            self._b3_sysid_last_diff_u_delayed = (
                diff_des_delayed
                .detach()
                .clone()
            )

        self._robot.set_external_force_and_torque(
            self._thrust,
            self._moment,
            body_ids=self._body_id,
        )


    env._apply_action = MethodType(
        _apply_action_b3_sysid,
        env,
    )

    hover_force_per_motor = (
        B3_PLANT_MASS_KG * 9.81 / 4.0
    )

    discriminant = (
        B3_SYSID_THRUST_LINEAR_N ** 2
        + 4.0
        * B3_SYSID_THRUST_QUADRATIC_N
        * hover_force_per_motor
    )

    b3_sysid_hover_u = (
        -B3_SYSID_THRUST_LINEAR_N
        + discriminant ** 0.5
    ) / (
        2.0 * B3_SYSID_THRUST_QUADRATIC_N
    )

    print()
    print("=" * 100)
    print("B3 REAL SYSTEM-ID ACTUATOR INSTALLED")
    print("=" * 100)
    print(
        "thrust law : F = "
        f"{B3_SYSID_THRUST_LINEAR_N}*u + "
        f"{B3_SYSID_THRUST_QUADRATIC_N}*u^2 N"
    )
    print(
        "tau_m      :",
        installed_tau_m,
        "s",
    )
    print(
        "hover F/motor:",
        hover_force_per_motor,
        "N",
    )
    print(
        "predicted hover u:",
        b3_sysid_hover_u,
    )
    print(
        "f_to_TM:",
        env.f_to_TM[0].detach().cpu(),
    )
    print("=" * 100)

    # =================================================================
    # B3_DEPLOYMENT_PREHOVER_V1
    #
    # Optional deployment-faithful controller/actuator warm start.
    #
    # IMPORTANT:
    #   - Kinematic benchmark state is restored exactly afterward.
    #   - Mellinger keeps its own controller internal state.
    #   - Mellinger keeps its own B3 actuator state.
    #   - RL randomized motor state is NOT transplanted into B3.
    #
    # Enable with:
    #   B3_DEPLOYMENT_PREHOVER=1
    # =================================================================
    import os

    b3_deployment_prehover = (
        os.environ.get(
            "B3_DEPLOYMENT_PREHOVER",
            "1",
        ).strip() == "1"
    )

    # Optional legacy Aug-06 measured-state overrides.
    # Define these independently of deployment pre-hover so the clean
    # behavior-geometric path can bypass firmware initialization entirely.
    _b3_real_pregoal_error_text = None
    _b3_real_rpy_text = None
    _b3_real_lin_vel_text = None
    _b3_real_ang_vel_text = None

    if b3_deployment_prehover:
        B3_PREHOVER_S = float(
            os.environ.get(
                "B3_PREHOVER_S",
                "2.0",
            )
        )

        b3_prehover_steps = int(
            round(
                B3_PREHOVER_S
                * MELLINGER_RATE_HZ
            )
        )

        if b3_prehover_steps <= 0:
            raise RuntimeError(
                "B3_PREHOVER_S must be positive."
            )

        # -------------------------------------------------------------
        # Initialize the B3 actuator at its own physical hover state.
        #
        # Do NOT use the randomized RL robot's captured motor state:
        # those normalized values correspond to a different actuator
        # and mass.
        # -------------------------------------------------------------
        b3_hover_motor_state = torch.full(
            (4,),
            float(b3_sysid_hover_u),
            device=device,
            dtype=torch.float32,
        )

        env._motor_speeds[0].copy_(
            b3_hover_motor_state
        )

        env._motor_speeds_des[0].copy_(
            b3_hover_motor_state
        )

        # B3_UNIFIED_ACTUATOR_SYSID_V1:
        # initialize the full four-motor actuator consistently at hover.
        if env._b3_unified_actuator_enabled:
            env._b3_unified_motor_state[
                0
            ].fill_(
                float(b3_sysid_hover_u)
            )

            if (
                env._b3_unified_delay_buffer
                is not None
            ):
                env._b3_unified_delay_buffer[
                    :,
                    0,
                    :,
                ].fill_(
                    float(b3_sysid_hover_u)
                )

            env._b3_unified_delay_index = 0

        # B3_DIFFERENTIAL_ACTUATOR_ABLATION_V1:
        # initialize split dynamics consistently at hover.
        if env._b3_diff_actuator_enabled:
            env._b3_collective_motor_state[
                0,
                0,
            ] = float(
                b3_sysid_hover_u
            )

            env._b3_diff_motor_state[
                0
            ].zero_()

            if (
                env._b3_diff_delay_buffer
                is not None
            ):
                env._b3_diff_delay_buffer[
                    :,
                    0,
                    :,
                ].zero_()

            env._b3_diff_delay_index = 0

        b3_hover_srt_action = (
            2.0 * b3_hover_motor_state
            - 1.0
        )

        env._actions[0].copy_(
            b3_hover_srt_action
        )

        env._action_queue[:, 0, :] = (
            b3_hover_srt_action
            .view(1, 4)
            .repeat(
                env._action_queue.shape[0],
                1,
            )
        )

        if hasattr(
            env,
            "_previous_action",
        ):
            env._previous_action[0].copy_(
                b3_hover_srt_action
            )

        # -------------------------------------------------------------
        # Pre-hover reference:
        #
        # Hold the INITIAL BODY location and initial yaw.
        # Mellinger/B3 is allowed to establish its own actual
        # equilibrium relative to that command.
        # -------------------------------------------------------------
        b3_prehover_goal = (
            expected_body_pos
            .detach()
            .clone()
        )

        q_w = float(
            expected_body_quat[0].item()
        )
        q_x = float(
            expected_body_quat[1].item()
        )
        q_y = float(
            expected_body_quat[2].item()
        )
        q_z = float(
            expected_body_quat[3].item()
        )

        b3_prehover_yaw = math.atan2(
            2.0 * (
                q_w * q_z
                + q_x * q_y
            ),
            1.0 - 2.0 * (
                q_y * q_y
                + q_z * q_z
            ),
        )

        # =============================================================
        # B3_AUG06_REAL_COMMAND_YAW_V1
        #
        # The Aug-06 PoseStamped commands use identity orientation for
        # every point-to-point goal.  In this real-flight validation
        # mode, the firmware therefore receives psi_des = 0 before and
        # after the position setpoint step.
        # =============================================================
        b3_real_command_zero_yaw = (
            os.environ.get(
                "B3_REAL_COMMAND_ZERO_YAW",
                "0",
            ).strip().lower()
            in ("1", "true", "yes", "on")
        )

        if b3_real_command_zero_yaw:
            b3_prehover_yaw = 0.0

        print()
        print("=" * 100)
        print(
            "B3 DEPLOYMENT-FAITHFUL PRE-HOVER"
        )
        print("=" * 100)
        print(
            "duration              :",
            B3_PREHOVER_S,
            "s",
        )
        print(
            "pre-hover body goal   :",
            b3_prehover_goal.detach().cpu(),
        )
        print(
            "pre-hover yaw         :",
            b3_prehover_yaw,
            "rad",
        )
        print(
            "initial B3 hover u     :",
            float(b3_sysid_hover_u),
        )
        print(
            "RL captured motor u    :",
            captured_motor_state.detach().cpu(),
        )

        prehover_checkpoints = set(
            int(
                round(
                    frac
                    * max(
                        b3_prehover_steps - 1,
                        1,
                    )
                )
            )
            for frac in (
                0.0,
                0.25,
                0.50,
                0.75,
                1.0,
            )
        )

        last_prehover_fw = None

        for pre_step in range(
            b3_prehover_steps
        ):
            warm_obs = env._get_observations()
            warm_gc = warm_obs["gc"].clone()

            warm_gc[0, 13:16] = (
                b3_prehover_goal
            )
            warm_gc[0, 16] = (
                b3_prehover_yaw
            )

            warm_action, warm_fw = (
                controller.get_action_from_gc(
                    warm_gc,
                    device=device,
                )
            )

            last_prehover_fw = warm_fw

            (
                _warm_obs_dict,
                _warm_reward,
                _warm_term,
                _warm_trunc,
                _warm_info,
            ) = env.step(
                warm_action
            )

            # Warm-up is not part of benchmark episode accounting.
            env.episode_length_buf[0] = 0

            if (
                pre_step
                in prehover_checkpoints
            ):
                (
                    warm_body_pos,
                    _,
                    warm_body_vel,
                    _,
                ) = (
                    env.get_frame_state_from_task(
                        "body"
                    )
                )

                warm_z_error = float(
                    b3_prehover_goal[2]
                    - warm_body_pos[0, 2]
                )

                i_error_z = getattr(
                    controller.controller,
                    "i_error_z",
                    float("nan"),
                )

                print(
                    f"[prehover "
                    f"{(pre_step + 1) / MELLINGER_RATE_HZ:6.3f}s] "
                    f"z={float(warm_body_pos[0,2]):+.5f} "
                    f"ez={warm_z_error:+.5f} "
                    f"vz={float(warm_body_vel[0,2]):+.5f} "
                    f"i_z={float(i_error_z):+.5f} "
                    f"thrust={float(warm_fw['thrust']):.2f} "
                    f"motors={warm_fw['motor_pwm']}"
                )

        # Save B3-specific hidden/actuator state resulting from
        # the physically simulated pre-hover.
        warmed_motor_state = (
            env._motor_speeds[0]
            .detach()
            .clone()
        )

        (
            prehover_final_body_pos,
            _,
            prehover_final_body_vel,
            _,
        ) = env.get_frame_state_from_task(
            "body"
        )

        prehover_final_i_z = float(
            getattr(
                controller.controller,
                "i_error_z",
                float("nan"),
            )
        )

        print("-" * 100)
        print(
            "pre-hover final z      :",
            float(
                prehover_final_body_pos[
                    0,
                    2,
                ]
            ),
        )
        print(
            "pre-hover z offset     :",
            float(
                prehover_final_body_pos[
                    0,
                    2,
                ]
                - b3_prehover_goal[2]
            ),
            "m",
        )
        print(
            "pre-hover final vz     :",
            float(
                prehover_final_body_vel[
                    0,
                    2,
                ]
            ),
            "m/s",
        )
        print(
            "pre-hover final i_z    :",
            prehover_final_i_z,
        )
        print(
            "pre-hover motor state  :",
            warmed_motor_state.detach().cpu(),
        )

        if last_prehover_fw is not None:
            print(
                "pre-hover cmd thrust   :",
                last_prehover_fw[
                    "thrust"
                ],
            )
            print(
                "pre-hover motor PWM    :",
                last_prehover_fw[
                    "motor_pwm"
                ],
            )

        # B3_CONTINUE_FROM_PREHOVER_V1
        # Preserve the genuine physical state reached at the end of
        # deployment pre-hover.  We still perform the exact-state
        # restoration below as a diagnostic, but deployment-faithful
        # evaluation can switch back to this state before rollout.
        prehover_root_state = (
            env._robot.data.root_state_w[0]
            .detach()
            .clone()
        )

        (
            _b3_prehover_body_pos_for_mapping,
            _,
            _,
            _,
        ) = env.get_frame_state_from_task(
            "body"
        )

        b3_prehover_root_minus_body_pos = (
            prehover_root_state[0:3]
            - _b3_prehover_body_pos_for_mapping[0]
        ).detach().clone()

        # -------------------------------------------------------------
        # Restore EXACT benchmark kinematics.
        #
        # Controller state and B3 actuator state are intentionally
        # retained.  This gives both controllers the same physical
        # handoff pose/velocity while allowing controller-specific
        # hidden state.
        # -------------------------------------------------------------
        env._robot.write_root_pose_to_sim(
            root_state[0:7].view(1, 7),
            env_ids=env_ids,
        )

        env._robot.write_root_velocity_to_sim(
            root_state[7:13].view(1, 6),
            env_ids=env_ids,
        )

        env._motor_speeds[0].copy_(
            warmed_motor_state
        )

        env._motor_speeds_des[0].copy_(
            warmed_motor_state
        )

        warmed_srt_action = (
            2.0 * warmed_motor_state
            - 1.0
        )

        env._actions[0].copy_(
            warmed_srt_action
        )

        env._action_queue[:, 0, :] = (
            warmed_srt_action
            .view(1, 4)
            .repeat(
                env._action_queue.shape[0],
                1,
            )
        )

        if hasattr(
            env,
            "_previous_action",
        ):
            env._previous_action[0].copy_(
                warmed_srt_action
            )

        if hasattr(
            env,
            "_action_history",
        ):
            env._action_history[0] = (
                warmed_srt_action
                .view(1, 4)
                .repeat(
                    env._action_history.shape[1],
                    1,
                )
            )

        env.episode_length_buf[0] = 0

        # Invalidate Isaac state caches after direct root write.
        robot_data = env._robot.data

        for buffer_name in (
            "_body_com_vel_w",
            "_body_link_vel_w",
            "_body_state_w",
            "_body_link_state_w",
            "_body_com_state_w",
        ):
            buffer = getattr(
                robot_data,
                buffer_name,
                None,
            )

            if buffer is not None:
                buffer.timestamp = -1.0

        obs_dict = env._get_observations()

        # -------------------------------------------------------------
        # Revalidate exact benchmark kinematics AFTER warm-up.
        # -------------------------------------------------------------
        (
            restored_body_pos,
            restored_body_quat,
            restored_body_vel,
            restored_body_ang,
        ) = env.get_frame_state_from_task(
            "body"
        )

        warm_restore_pos_error = float(
            torch.max(
                torch.abs(
                    restored_body_pos[0]
                    - expected_body_pos
                )
            ).item()
        )

        warm_restore_quat_error = (
            quaternion_error(
                restored_body_quat[0],
                expected_body_quat,
            )
        )

        warm_restore_vel_error = float(
            torch.max(
                torch.abs(
                    restored_body_vel[0]
                    - expected_body_vel
                )
            ).item()
        )

        warm_restore_ang_error = float(
            torch.max(
                torch.abs(
                    restored_body_ang[0]
                    - expected_body_ang
                )
            ).item()
        )

        print("-" * 100)
        print(
            "RESTORED BENCHMARK STATE"
        )
        print(
            "position error         :",
            f"{warm_restore_pos_error:.9e}",
        )
        print(
            "quaternion error       :",
            f"{warm_restore_quat_error:.9e}",
        )
        print(
            "linear velocity error  :",
            f"{warm_restore_vel_error:.9e}",
        )
        print(
            "angular velocity error :",
            f"{warm_restore_ang_error:.9e}",
        )
        print(
            "retained Mellinger i_z :",
            float(
                getattr(
                    controller.controller,
                    "i_error_z",
                    float("nan"),
                )
            ),
        )
        print(
            "retained B3 motor state:",
            env._motor_speeds[
                0
            ].detach().cpu(),
        )
        print("=" * 100)

        if max(
            warm_restore_pos_error,
            warm_restore_quat_error,
            warm_restore_vel_error,
            warm_restore_ang_error,
        ) > 1.0e-5:
            raise RuntimeError(
                "B3 pre-hover failed to restore "
                "the exact benchmark kinematic state."
            )

        # -------------------------------------------------------------
        # Optional deployment-faithful task handoff.
        #
        # The exact-state restoration above remains useful as a
        # validation diagnostic.  For the actual deployment-faithful
        # experiment, return to the genuine state reached naturally by
        # the B3 after pre-hover and continue without a physical reset.
        # -------------------------------------------------------------
        b3_continue_from_prehover = (
            os.environ.get(
                "B3_CONTINUE_FROM_PREHOVER",
                "1",
            ).strip() == "1"
        )

        # =============================================================
        # B3_AUG06_REAL_PREGOAL_STATE_V1
        #
        # Prehover above is used ONLY to warm hidden controller and
        # actuator state.  For real-flight validation, install the
        # measured Aug-06 pre-goal kinematic state before rollout.
        #
        # Controller-position convention:
        #
        #   e_pre = held_setpoint - actual_position
        #
        # therefore:
        #
        #   actual_position = held_setpoint - e_pre
        #
        # The actual world position may be translated arbitrarily; only
        # the controller-relative error and orientation/velocity matter.
        # =============================================================
        _b3_real_pregoal_error_text = os.environ.get(
            "B3_REAL_PREGOAL_GC_ERROR"
        )

        _b3_real_rpy_text = os.environ.get(
            "B3_REAL_INITIAL_RPY_DEG"
        )

        _b3_real_lin_vel_text = os.environ.get(
            "B3_REAL_INITIAL_LIN_VEL_BODY"
        )

        _b3_real_ang_vel_text = os.environ.get(
            "B3_REAL_INITIAL_ANG_VEL_BODY"
        )

        _b3_real_state_fields = (
            _b3_real_pregoal_error_text,
            _b3_real_rpy_text,
            _b3_real_lin_vel_text,
            _b3_real_ang_vel_text,
        )

        if any(v is not None for v in _b3_real_state_fields):
            if not all(v is not None for v in _b3_real_state_fields):
                raise RuntimeError(
                    "B3 Aug-06 real pre-goal state requires all four: "
                    "B3_REAL_PREGOAL_GC_ERROR, "
                    "B3_REAL_INITIAL_RPY_DEG, "
                    "B3_REAL_INITIAL_LIN_VEL_BODY, "
                    "B3_REAL_INITIAL_ANG_VEL_BODY."
                )

            def _b3_parse_real_vec3(text, name):
                vals = [
                    float(v)
                    for v in text.split(",")
                ]

                if len(vals) != 3:
                    raise RuntimeError(
                        f"{name} must contain exactly three "
                        "comma-separated values."
                    )

                return torch.tensor(
                    vals,
                    device=prehover_root_state.device,
                    dtype=prehover_root_state.dtype,
                )

            b3_real_pregoal_gc_error = _b3_parse_real_vec3(
                _b3_real_pregoal_error_text,
                "B3_REAL_PREGOAL_GC_ERROR",
            )

            b3_real_initial_rpy_deg = _b3_parse_real_vec3(
                _b3_real_rpy_text,
                "B3_REAL_INITIAL_RPY_DEG",
            )

            b3_real_initial_lin_vel_body = _b3_parse_real_vec3(
                _b3_real_lin_vel_text,
                "B3_REAL_INITIAL_LIN_VEL_BODY",
            )

            b3_real_initial_ang_vel_body = _b3_parse_real_vec3(
                _b3_real_ang_vel_text,
                "B3_REAL_INITIAL_ANG_VEL_BODY",
            )

            # Exact body position needed to reproduce measured controller
            # error against the held old setpoint.
            b3_real_initial_body_pos = (
                b3_prehover_goal
                - b3_real_pregoal_gc_error
            )

            # Convert measured roll/pitch/yaw to Isaac WXYZ quaternion.
            _roll = math.radians(
                float(b3_real_initial_rpy_deg[0].item())
            )
            _pitch = math.radians(
                float(b3_real_initial_rpy_deg[1].item())
            )
            _yaw = math.radians(
                float(b3_real_initial_rpy_deg[2].item())
            )

            _cr = math.cos(0.5 * _roll)
            _sr = math.sin(0.5 * _roll)
            _cp = math.cos(0.5 * _pitch)
            _sp = math.sin(0.5 * _pitch)
            _cy = math.cos(0.5 * _yaw)
            _sy = math.sin(0.5 * _yaw)

            b3_real_initial_quat_wxyz = torch.tensor(
                [
                    _cr * _cp * _cy + _sr * _sp * _sy,
                    _sr * _cp * _cy - _cr * _sp * _sy,
                    _cr * _sp * _cy + _sr * _cp * _sy,
                    _cr * _cp * _sy - _sr * _sp * _cy,
                ],
                device=prehover_root_state.device,
                dtype=prehover_root_state.dtype,
            )

            # Rotation body -> world. ROS Odometry twist convention is
            # child/body-frame; Isaac root velocities are world-frame.
            _qw = b3_real_initial_quat_wxyz[0]
            _qx = b3_real_initial_quat_wxyz[1]
            _qy = b3_real_initial_quat_wxyz[2]
            _qz = b3_real_initial_quat_wxyz[3]

            b3_real_R_wb = torch.stack(
                (
                    torch.stack(
                        (
                            1.0 - 2.0 * (_qy * _qy + _qz * _qz),
                            2.0 * (_qx * _qy - _qw * _qz),
                            2.0 * (_qx * _qz + _qw * _qy),
                        )
                    ),
                    torch.stack(
                        (
                            2.0 * (_qx * _qy + _qw * _qz),
                            1.0 - 2.0 * (_qx * _qx + _qz * _qz),
                            2.0 * (_qy * _qz - _qw * _qx),
                        )
                    ),
                    torch.stack(
                        (
                            2.0 * (_qx * _qz - _qw * _qy),
                            2.0 * (_qy * _qz + _qw * _qx),
                            1.0 - 2.0 * (_qx * _qx + _qy * _qy),
                        )
                    ),
                )
            )

            b3_real_initial_lin_vel_world = (
                b3_real_R_wb
                @ b3_real_initial_lin_vel_body
            )

            b3_real_initial_ang_vel_world = (
                b3_real_R_wb
                @ b3_real_initial_ang_vel_body
            )

            # Keep root/body positional relationship measured directly
            # from the current articulation.
            prehover_root_state[0:3].copy_(
                b3_real_initial_body_pos
                + b3_prehover_root_minus_body_pos
            )

            prehover_root_state[3:7].copy_(
                b3_real_initial_quat_wxyz
            )

            prehover_root_state[7:10].copy_(
                b3_real_initial_lin_vel_world
            )

            prehover_root_state[10:13].copy_(
                b3_real_initial_ang_vel_world
            )

            print()
            print("=" * 100)
            print(
                "B3 AUG-06 REAL PRE-GOAL HANDOFF STATE"
            )
            print("=" * 100)
            print(
                "held GC setpoint       :",
                b3_prehover_goal.detach().cpu(),
            )
            print(
                "real pre-goal GC error :",
                b3_real_pregoal_gc_error.detach().cpu(),
            )
            print(
                "target actual body pos :",
                b3_real_initial_body_pos.detach().cpu(),
            )
            print(
                "real initial RPY deg   :",
                b3_real_initial_rpy_deg.detach().cpu(),
            )
            print(
                "real lin vel body      :",
                b3_real_initial_lin_vel_body.detach().cpu(),
            )
            print(
                "real ang vel body      :",
                b3_real_initial_ang_vel_body.detach().cpu(),
            )
            print(
                "installed root quat WXYZ:",
                b3_real_initial_quat_wxyz.detach().cpu(),
            )
            print("=" * 100)

        if b3_continue_from_prehover:
            env._robot.write_root_pose_to_sim(
                prehover_root_state[0:7].view(1, 7),
                env_ids=env_ids,
            )

            env._robot.write_root_velocity_to_sim(
                prehover_root_state[7:13].view(1, 6),
                env_ids=env_ids,
            )

            env._motor_speeds[0].copy_(
                warmed_motor_state
            )
            env._motor_speeds_des[0].copy_(
                warmed_motor_state
            )

            deployment_srt_action = (
                2.0 * warmed_motor_state
                - 1.0
            )

            env._actions[0].copy_(
                deployment_srt_action
            )

            env._action_queue[:, 0, :] = (
                deployment_srt_action
                .view(1, 4)
                .repeat(
                    env._action_queue.shape[0],
                    1,
                )
            )

            if hasattr(env, "_previous_action"):
                env._previous_action[0].copy_(
                    deployment_srt_action
                )

            if hasattr(env, "_action_history"):
                env._action_history[0] = (
                    deployment_srt_action
                    .view(1, 4)
                    .repeat(
                        env._action_history.shape[1],
                        1,
                    )
                )

            env.episode_length_buf[0] = 0

            robot_data = env._robot.data

            for buffer_name in (
                "_body_com_vel_w",
                "_body_link_vel_w",
                "_body_state_w",
                "_body_link_state_w",
                "_body_com_state_w",
            ):
                buffer = getattr(
                    robot_data,
                    buffer_name,
                    None,
                )

                if buffer is not None:
                    buffer.timestamp = -1.0

            # =============================================================
            # DEPLOYMENT_RELATIVE_EE_TRANSFER_V1_PATCH
            #
            # RL and Mellinger receive the same physical EE displacement,
            # but each starts from its own deployment operating state.
            #
            # RL displacement:
            #   delta_p = RL_goal_EE - RL_initial_EE
            #
            # Mellinger goal:
            #   post_prehover_EE + delta_p
            # =============================================================
            deployment_ee_frame = (
                "endeffector"
                if bool(
                    getattr(
                        env.cfg,
                        "has_end_effector",
                        False,
                    )
                )
                else "body"
            )

            (
                deployment_ee_pos,
                deployment_ee_quat,
                _deployment_ee_vel,
                _deployment_ee_ang,
            ) = env.get_frame_state_from_task(
                deployment_ee_frame
            )

            # B3_REAL_GOAL_DELIVERY_DELAY_V2
            #
            # b3_prehover_goal is the authoritative BODY/COM setpoint
            # Mellinger was receiving during deployment pre-hover.
            #
            # The frozen relative-goal machinery below operates on an
            # authoritative EE goal, so convert the commanded pre-hover
            # body goal into the corresponding EE goal using the ACTUAL
            # body->EE displacement at task handoff.
            #
            # Do NOT use env._desired_pos_w here: the environment goal
            # machinery may already have regenerated stale task-goal
            # components into that buffer.
            #
            # Do NOT use the current vehicle position as the goal either:
            # doing so would erase the genuine pre-hover tracking error
            # against which the warmed Mellinger integrator was accumulated.
            (
                b3_hold_body_pos,
                _b3_hold_body_quat,
                _b3_hold_body_vel,
                _b3_hold_body_ang,
            ) = env.get_frame_state_from_task(
                "body"
            )

            b3_pre_task_goal_env = (
                b3_prehover_goal
                + (
                    deployment_ee_pos[0]
                    - b3_hold_body_pos[0]
                )
            ).detach().clone()

            # STATIC_TRANSFER_LITERAL_EE_DELTA_V1
            #
            # Normal replay keeps the original deployment-relative
            # transfer semantics. Static system-ID replay may provide
            # an exact physical EE displacement through the environment.
            _static_goal_offset_text = os.environ.get(
                "AERIAL_STATIC_GOAL_OFFSET"
            )

            # REAL_B3_COMMAND_SETPOINT_REPLAY_V1
            #
            # For validation against a real Mellinger point-to-point
            # experiment, replay the SETPOINT CHANGE the real firmware
            # received, rather than reconstructing a target from the
            # vehicle's realized position.
            #
            # Example Aug-06 goal:
            #   [0, 4, 0.8] -> [1, 6, 0.8]
            # gives controller-setpoint delta:
            #   [+1, +2, 0]
            _static_gc_goal_offset_text = os.environ.get(
                "AERIAL_STATIC_GC_GOAL_OFFSET"
            )

            if (
                _static_goal_offset_text
                and _static_gc_goal_offset_text
            ):
                raise RuntimeError(
                    "Set only one of AERIAL_STATIC_GOAL_OFFSET "
                    "or AERIAL_STATIC_GC_GOAL_OFFSET."
                )

            b3_static_gc_goal_offset = None

            if _static_gc_goal_offset_text:
                _gc_values = [
                    float(v)
                    for v in
                    _static_gc_goal_offset_text.split(",")
                ]

                if len(_gc_values) != 3:
                    raise RuntimeError(
                        "AERIAL_STATIC_GC_GOAL_OFFSET must contain "
                        "exactly three comma-separated values."
                    )

                b3_static_gc_goal_offset = torch.tensor(
                    _gc_values,
                    device=b3_prehover_goal.device,
                    dtype=b3_prehover_goal.dtype,
                )

                print(
                    "[STATIC GC COMMAND] literal controller-setpoint delta:",
                    b3_static_gc_goal_offset.detach().cpu(),
                )

            if _static_goal_offset_text:
                _values = [
                    float(v)
                    for v in _static_goal_offset_text.split(",")
                ]

                if len(_values) != 3:
                    raise RuntimeError(
                        "AERIAL_STATIC_GOAL_OFFSET must contain "
                        "exactly three comma-separated values."
                    )

                rl_ee_displacement = torch.tensor(
                    _values,
                    device=device,
                    dtype=torch.float32,
                )

                print(
                    "[STATIC TRANSFER] literal EE delta:",
                    rl_ee_displacement.detach().cpu(),
                )

            else:
                rl_ee_displacement = (
                    desired_pos_original
                    .to(
                        device=device,
                        dtype=torch.float32,
                    )
                    .reshape(3)
                    -
                    initial["ee_pos_w"]
                    .to(
                        device=device,
                        dtype=torch.float32,
                    )
                    .reshape(3)
                )

            mellinger_task_goal_env = (
                deployment_ee_pos[0]
                + rl_ee_displacement
            )

            # _desired_pos_w is the authoritative EE goal.
            # get_goal_state_from_task("COM") converts this to the
            # corresponding Mellinger COM goal when GC is generated.
            # Install the new static EE target in BOTH representations.
            #
            # quadrotor_env.py refreshes:
            #   _desired_pos_w <- _desired_pos_traj_w[:, 0]
            #
            # Therefore changing only _desired_pos_w is temporary and
            # would be overwritten before/while generating GC.
            env._desired_pos_w[0].copy_(
                mellinger_task_goal_env
            )

            env._desired_pos_traj_w[0].copy_(
                mellinger_task_goal_env
                .view(1, 3)
                .expand_as(
                    env._desired_pos_traj_w[0]
                )
            )

            # =============================================================
            # B3_RELATIVE_GOAL_PERSISTENCE_V1
            #
            # _get_observations() calls update_goal_state(), which normally
            # regenerates the original trajectory and would overwrite the
            # deployment-relative target on every control cycle.
            #
            # This comparator experiment is a STATIC EE transfer, so freeze
            # the post-prehover relative target for the Mellinger rollout.
            # The baseline environment source remains unchanged.
            # =============================================================
            from types import MethodType as _B3MethodType

            b3_final_relative_goal_pos = (
                mellinger_task_goal_env
                .detach()
                .clone()
            )

            # Mutable target used by the frozen update_goal_state() hook.
            # It normally equals the final task goal, but during the
            # measured real command-delivery latency it remains at the
            # pre-task setpoint.
            b3_active_relative_goal_pos = (
                b3_final_relative_goal_pos
                .detach()
                .clone()
            )

            # Final task orientation, preserved exactly as before.
            b3_final_relative_goal_ori = (
                env._desired_ori_w[0]
                .detach()
                .clone()
            )

            b3_final_relative_goal_ori_traj = (
                env._desired_ori_traj_w[0]
                .detach()
                .clone()
            )

            # B3_REAL_GOAL_DELIVERY_DELAY_V3
            #
            # Pre-hover was commanded using warm_gc[0,16] =
            # b3_prehover_yaw.  Build the equivalent pure-yaw quaternion
            # in Isaac's WXYZ convention so the 82 ms hold continues the
            # SAME commanded orientation rather than immediately applying
            # the final task orientation.
            b3_prehover_goal_ori = torch.tensor(
                [
                    math.cos(0.5 * b3_prehover_yaw),
                    0.0,
                    0.0,
                    math.sin(0.5 * b3_prehover_yaw),
                ],
                device=env._desired_ori_w.device,
                dtype=env._desired_ori_w.dtype,
            )

            b3_prehover_goal_ori_traj = (
                b3_prehover_goal_ori
                .view(1, 4)
                .expand_as(
                    b3_final_relative_goal_ori_traj
                )
                .clone()
            )

            # Mutable orientation targets.  Normally these contain the
            # final task orientation.  During a real-goal delivery hold
            # they temporarily contain the exact pre-hover orientation.
            b3_active_relative_goal_ori = (
                b3_final_relative_goal_ori
                .detach()
                .clone()
            )

            b3_active_relative_goal_ori_traj = (
                b3_final_relative_goal_ori_traj
                .detach()
                .clone()
            )

            def _b3_static_relative_update_goal_state(
                self,
                *args,
                **kwargs,
            ):
                self._desired_pos_w[0].copy_(
                    b3_active_relative_goal_pos
                )

                self._desired_pos_traj_w[0].copy_(
                    b3_active_relative_goal_pos
                    .view(1, 3)
                    .expand_as(
                        self._desired_pos_traj_w[0]
                    )
                )

                self._desired_ori_w[0].copy_(
                    b3_active_relative_goal_ori
                )

                self._desired_ori_traj_w[0].copy_(
                    b3_active_relative_goal_ori_traj
                )

            env.update_goal_state = _B3MethodType(
                _b3_static_relative_update_goal_state,
                env,
            )

            # Install once immediately.
            env.update_goal_state()

            # =============================================================
            # REAL_B3_COMMAND_SETPOINT_REPLAY_V1
            #
            # b3_prehover_goal is literally the vector written to
            # warm_gc[0,13:16] throughout physical pre-hover.
            #
            # Therefore a real commanded setpoint step d_gc must produce:
            #
            #   final_gc = b3_prehover_goal + d_gc
            #
            # NOT:
            #
            #   current_realized_vehicle_position + displacement
            #
            # Solve the corresponding EE goal through the environment's
            # own EE -> COM transformation so no frame offsets are guessed.
            # =============================================================
            if b3_static_gc_goal_offset is not None:

                # =====================================================
                # B3_AUG06_REAL_FINAL_YAW_V1
                #
                # Real Aug-06 PoseStamped goals all use identity
                # orientation.  Internal orientation convention is WXYZ.
                # =====================================================
                if b3_real_command_zero_yaw:
                    b3_final_relative_goal_ori.zero_()
                    b3_final_relative_goal_ori[..., 0] = 1.0

                    b3_final_relative_goal_ori_traj.copy_(
                        b3_final_relative_goal_ori
                    )

                    b3_active_relative_goal_ori.copy_(
                        b3_final_relative_goal_ori
                    )

                    b3_active_relative_goal_ori_traj.copy_(
                        b3_final_relative_goal_ori_traj
                    )

                    print(
                        "[REAL B3 COMMAND] final orientation WXYZ:",
                        b3_final_relative_goal_ori.detach().cpu(),
                    )

                b3_real_command_final_gc_goal = (
                    b3_prehover_goal
                    + b3_static_gc_goal_offset
                ).detach().clone()

                b3_real_command_ee_correction = torch.zeros_like(
                    b3_final_relative_goal_pos
                )

                for _gc_solve_iter in range(6):
                    (
                        _gc_goal_now,
                        _,
                    ) = env.get_goal_state_from_task(
                        "COM"
                    )

                    _gc_goal_residual = (
                        b3_real_command_final_gc_goal
                        - _gc_goal_now[0]
                    )

                    if float(
                        torch.norm(
                            _gc_goal_residual
                        ).item()
                    ) <= 1.0e-7:
                        break

                    b3_real_command_ee_correction.add_(
                        _gc_goal_residual
                    )

                    b3_final_relative_goal_pos.add_(
                        _gc_goal_residual
                    )

                    b3_active_relative_goal_pos.copy_(
                        b3_final_relative_goal_pos
                    )

                    env.update_goal_state()

                env.update_goal_state()

                (
                    _gc_goal_check,
                    _,
                ) = env.get_goal_state_from_task(
                    "COM"
                )

                b3_real_command_gc_error = float(
                    torch.norm(
                        _gc_goal_check[0]
                        - b3_real_command_final_gc_goal
                    ).item()
                )

                if b3_real_command_gc_error > 1.0e-4:
                    raise RuntimeError(
                        "REAL_B3_COMMAND_SETPOINT_REPLAY_V1: "
                        f"final GC goal mismatch="
                        f"{b3_real_command_gc_error:.6e} m, "
                        f"actual="
                        f"{_gc_goal_check[0].detach().cpu().tolist()}, "
                        f"target="
                        f"{b3_real_command_final_gc_goal.detach().cpu().tolist()}"
                    )

                # Keep all downstream EE-goal logging/metrics consistent
                # with the corrected real-command-faithful target.
                mellinger_task_goal_env.copy_(
                    b3_final_relative_goal_pos
                )

                rl_ee_displacement = (
                    mellinger_task_goal_env
                    - deployment_ee_pos[0]
                ).detach().clone()

                print()
                print("=" * 100)
                print(
                    "REAL B3 COMMAND-SETPOINT REPLAY"
                )
                print("=" * 100)
                print(
                    "pre-hover GC setpoint :",
                    b3_prehover_goal.detach().cpu(),
                )
                print(
                    "commanded GC delta    :",
                    b3_static_gc_goal_offset.detach().cpu(),
                )
                print(
                    "final GC setpoint     :",
                    b3_real_command_final_gc_goal.detach().cpu(),
                )
                print(
                    "solved final EE goal  :",
                    b3_final_relative_goal_pos.detach().cpu(),
                )
                print(
                    "EE correction applied :",
                    b3_real_command_ee_correction.detach().cpu(),
                )
                print(
                    "final GC error        :",
                    f"{b3_real_command_gc_error:.9e}",
                    "m",
                )
                print("=" * 100)

            expected_relative_com_goal, _ = (
                env.get_goal_state_from_task(
                    "COM"
                )
            )

            print(
                "[RELGOAL] desired EE goal:",
                env._desired_pos_w[0]
                .detach()
                .cpu(),
            )
            print(
                "[RELGOAL] expected COM goal:",
                expected_relative_com_goal[0]
                .detach()
                .cpu(),
            )

            # This calls the newly installed frozen update_goal_state().
            obs_dict = env._get_observations()

            actual_relative_gc_goal = (
                obs_dict["gc"][0, 13:16]
            )

            relative_gc_goal_error = float(
                torch.max(
                    torch.abs(
                        actual_relative_gc_goal
                        - expected_relative_com_goal[0]
                    )
                ).item()
            )

            print(
                "[RELGOAL] GC COM goal:",
                actual_relative_gc_goal
                .detach()
                .cpu(),
            )
            print(
                "[RELGOAL] GC goal error:",
                f"{relative_gc_goal_error:.9e}",
            )

            if relative_gc_goal_error > 1.0e-5:
                raise RuntimeError(
                    "Deployment-relative Mellinger goal did not "
                    "persist into the GC observation: "
                    f"error={relative_gc_goal_error:.9e}"
                )

            # Save both Mellinger start and target in the ORIGINAL
            # world frame so metrics/provenance are directly comparable
            # with the saved RL trace.
            mellinger_initial_ee_original = (
                deployment_ee_pos[0]
                - translation
            ).detach().cpu()

            mellinger_task_goal_original = (
                mellinger_task_goal_env
                - translation
            ).detach().cpu()

            (
                deployment_body_pos,
                deployment_body_quat,
                deployment_body_vel,
                deployment_body_ang,
            ) = env.get_frame_state_from_task(
                "body"
            )

            print()
            print("=" * 100)
            print(
                "B3 DEPLOYMENT-FAITHFUL TASK HANDOFF"
            )
            print("=" * 100)
            print(
                "body position         :",
                deployment_body_pos[0].detach().cpu(),
            )
            print(
                "body velocity         :",
                deployment_body_vel[0].detach().cpu(),
            )
            print(
                "body angular velocity :",
                deployment_body_ang[0].detach().cpu(),
            )
            print(
                "Mellinger i_z         :",
                float(
                    getattr(
                        controller.controller,
                        "i_error_z",
                        float("nan"),
                    )
                ),
            )
            print(
                "B3 motor state        :",
                env._motor_speeds[0].detach().cpu(),
            )
            print(
                "initial EE position   :",
                deployment_ee_pos[0]
                .detach()
                .cpu(),
            )
            print(
                "commanded EE delta    :",
                rl_ee_displacement
                .detach()
                .cpu(),
            )
            print(
                "Mellinger EE goal     :",
                mellinger_task_goal_env
                .detach()
                .cpu(),
            )
            print(
                "task GC COM goal      :",
                obs_dict["gc"][0, 13:16]
                .detach()
                .cpu(),
            )
            print("=" * 100)

    # =================================================================
    # B3_REAL_GOAL_DELIVERY_DELAY_V1
    #
    # Real Aug-6 bag:
    #   median goal -> body-rate onset     = 96.3 ms
    # Corrected gain=0.15 simulator:
    #   goal -> body-rate onset            = 16.0 ms
    #
    # Missing command-path timing ~= 80-84 ms.
    # Use 82 ms (41 controller ticks at 500 Hz).
    #
    # This is NOT trajectory shaping.  The controller continues to see
    # the previous point target during the delay and then receives the
    # complete new point target instantaneously.
    # =================================================================
    b3_real_goal_delivery_delay_s = float(
        os.environ.get(
            "B3_REAL_GOAL_DELIVERY_DELAY_S",
            "0.0",
        )
    )

    if b3_real_goal_delivery_delay_s < 0.0:
        raise RuntimeError(
            "B3_REAL_GOAL_DELIVERY_DELAY_S must be >= 0."
        )

    b3_real_goal_delivery_delay_steps = int(
        round(
            b3_real_goal_delivery_delay_s
            * MELLINGER_RATE_HZ
        )
    )

    if (
        b3_real_goal_delivery_delay_steps > 0
        and "b3_active_relative_goal_pos" in locals()
    ):
        b3_active_relative_goal_pos.copy_(
            b3_pre_task_goal_env
        )

        b3_active_relative_goal_ori.copy_(
            b3_prehover_goal_ori
        )

        b3_active_relative_goal_ori_traj.copy_(
            b3_prehover_goal_ori_traj
        )

        env.update_goal_state()
        obs_dict = env._get_observations()

        # =============================================================
        # B3_REAL_GOAL_DELIVERY_DELAY_V4
        #
        # During physical pre-hover the firmware saw EXACTLY:
        #
        #   warm_gc[0,13:16] = b3_prehover_goal
        #   warm_gc[0,16]    = b3_prehover_yaw
        #
        # Therefore reproduce those exact GC values during the delay.
        # Let the environment perform its own EE -> COM transform and
        # iteratively apply the resulting residual to the held EE goal.
        # =============================================================
        _prehover_gc_position_correction = torch.zeros_like(
            b3_active_relative_goal_pos
        )

        for _b3_gc_fix_iter in range(4):
            (
                _preliminary_hold_com_goal,
                _,
            ) = env.get_goal_state_from_task(
                "COM"
            )

            _b3_gc_residual = (
                b3_prehover_goal
                - _preliminary_hold_com_goal[0]
            )

            if float(
                torch.norm(
                    _b3_gc_residual
                ).item()
            ) <= 1.0e-7:
                break

            _prehover_gc_position_correction.add_(
                _b3_gc_residual
            )

            b3_active_relative_goal_pos.add_(
                _b3_gc_residual
            )

            env.update_goal_state()

        # Keep the provenance variable equal to the actual corrected hold.
        b3_pre_task_goal_env.copy_(
            b3_active_relative_goal_pos
        )

        # Final refresh AFTER correction.
        env.update_goal_state()
        obs_dict = env._get_observations()

        # -------------------------------------------------------------
        # B3_REAL_GOAL_DELIVERY_DELAY_V4 validity gates.
        #
        # A bad hold must abort immediately instead of generating
        # another uninterpretable 4-second trajectory.
        # -------------------------------------------------------------
        _held_ee = (
            b3_active_relative_goal_pos
            .detach()
            .clone()
        )

        # EE representation must contain exactly the held EE goal.
        _held_ee_err = float(
            torch.norm(
                env._desired_pos_w[0]
                - _held_ee
            ).item()
        )

        if _held_ee_err > 1.0e-4:
            raise RuntimeError(
                "B3_REAL_GOAL_DELIVERY_DELAY_V2: "
                f"EE hold mismatch={_held_ee_err:.6e} m"
            )

        # Convert the held EE goal through the environment's own
        # authoritative EE -> COM goal transform.
        (
            _expected_held_com_goal,
            _,
        ) = env.get_goal_state_from_task(
            "COM"
        )

        _held_gc_goal = (
            obs_dict["gc"][0, 13:16]
            .detach()
            .clone()
        )

        _held_gc_yaw = float(
            obs_dict["gc"][0, 16].item()
        )

        _held_gc_yaw_err = abs(
            math.atan2(
                math.sin(
                    _held_gc_yaw
                    - b3_prehover_yaw
                ),
                math.cos(
                    _held_gc_yaw
                    - b3_prehover_yaw
                ),
            )
        )

        if _held_gc_yaw_err > 1.0e-4:
            raise RuntimeError(
                "B3_REAL_GOAL_DELIVERY_DELAY_V3: "
                f"held GC yaw mismatch="
                f"{_held_gc_yaw_err:.6e} rad, "
                f"gc_yaw={_held_gc_yaw:.9f}, "
                f"prehover_yaw={b3_prehover_yaw:.9f}"
            )

        _held_gc_err = float(
            torch.norm(
                _held_gc_goal
                - _expected_held_com_goal[0]
            ).item()
        )

        if _held_gc_err > 1.0e-4:
            raise RuntimeError(
                "B3_REAL_GOAL_DELIVERY_DELAY_V2: "
                f"GC COM hold mismatch={_held_gc_err:.6e} m, "
                f"gc={_held_gc_goal.detach().cpu().tolist()}, "
                f"expected="
                f"{_expected_held_com_goal[0].detach().cpu().tolist()}"
            )

        # Exact controller-command continuity gate.
        #
        # b3_prehover_goal is not being interpreted geometrically here:
        # it is literally the vector written to warm_gc[0,13:16] during
        # every pre-hover controller call.
        _prehover_gc_position_err = float(
            torch.norm(
                _expected_held_com_goal[0]
                - b3_prehover_goal
            ).item()
        )

        if _prehover_gc_position_err > 1.0e-4:
            raise RuntimeError(
                "B3_REAL_GOAL_DELIVERY_DELAY_V4: "
                f"held GC position differs from actual pre-hover "
                f"controller command by "
                f"{_prehover_gc_position_err:.6e} m, "
                f"held_gc="
                f"{_expected_held_com_goal[0].detach().cpu().tolist()}, "
                f"prehover_gc="
                f"{b3_prehover_goal.detach().cpu().tolist()}"
            )

        # This gate would have caught the previous contaminated hold.
        _held_final_sep = float(
            torch.norm(
                _held_ee
                - b3_final_relative_goal_pos
            ).item()
        )

        if _held_final_sep < 1.0:
            raise RuntimeError(
                "B3_REAL_GOAL_DELIVERY_DELAY_V2: "
                f"held EE goal is only {_held_final_sep:.3f} m "
                "from final task goal; held pre-task goal is contaminated."
            )

        print()
        print("=" * 100)
        print("B3 REAL GOAL DELIVERY DELAY")
        print("=" * 100)
        print(
            "requested delay      :",
            b3_real_goal_delivery_delay_s,
            "s",
        )
        print(
            "realized delay       :",
            b3_real_goal_delivery_delay_steps
            / MELLINGER_RATE_HZ,
            "s",
        )
        print(
            "delay controller ticks:",
            b3_real_goal_delivery_delay_steps,
        )
        print(
            "held EE goal         :",
            _held_ee.detach().cpu(),
        )
        print(
            "held GC COM goal     :",
            _held_gc_goal.detach().cpu(),
        )
        print(
            "expected held COM    :",
            _expected_held_com_goal[0]
            .detach()
            .cpu(),
        )
        print(
            "pre-hover BODY goal  :",
            b3_prehover_goal
            .detach()
            .cpu(),
        )
        print(
            "held EE error        :",
            f"{_held_ee_err:.9e}",
        )
        print(
            "held GC error        :",
            f"{_held_gc_err:.9e}",
        )
        print(
            "held GC yaw          :",
            _held_gc_yaw,
            "rad",
        )
        print(
            "pre-hover yaw        :",
            b3_prehover_yaw,
            "rad",
        )
        print(
            "held yaw error       :",
            f"{_held_gc_yaw_err:.9e}",
            "rad",
        )
        print(
            "GC correction applied:",
            _prehover_gc_position_correction
            .detach()
            .cpu(),
        )
        print(
            "prehover GC pos error:",
            f"{_prehover_gc_position_err:.9e}",
            "m",
        )
        print(
            "held/final separation:",
            _held_final_sep,
            "m",
        )
        print(
            "final EE goal        :",
            b3_final_relative_goal_pos
            .detach()
            .cpu(),
        )
        print("=" * 100)

    # =================================================================
    # B3_SESSION_C_REACTION_STATE_V1
    #
    # When replaying an experimentally measured controller-reaction
    # state with zero artificial goal-delivery delay, initialize the
    # Mellinger hidden rate state consistently with the measured gyro.
    #
    # Otherwise the firmware sees an artificial instantaneous jump from
    # the pre-hover omega (~0) to the installed real omega and its
    # 2-ms finite-difference angular-acceleration term is wrong.
    # =================================================================
    if (
        _b3_real_ang_vel_text is not None
        and b3_real_goal_delivery_delay_steps == 0
    ):
        controller.controller.prev_omega_roll = float(
            b3_real_initial_ang_vel_body[0].item()
        )

        # Firmware convention:
        # stateAttitudeRatePitch = -radians(sensors->gyro.y)
        controller.controller.prev_omega_pitch = -float(
            b3_real_initial_ang_vel_body[1].item()
        )

        controller.controller.prev_setpoint_omega_roll = 0.0
        controller.controller.prev_setpoint_omega_pitch = 0.0

        _real_ix = os.environ.get("B3_REAL_MELLINGER_I_X")
        _real_iy = os.environ.get("B3_REAL_MELLINGER_I_Y")
        _real_iz = os.environ.get("B3_REAL_MELLINGER_I_Z")

        if _real_ix is not None:
            controller.controller.i_error_x = float(_real_ix)

        if _real_iy is not None:
            controller.controller.i_error_y = float(_real_iy)

        if _real_iz is not None:
            controller.controller.i_error_z = float(_real_iz)

        print()
        print("=" * 100)
        print("B3 SESSION-C REACTION-STATE CONTROLLER MEMORY")
        print("=" * 100)
        print(
            "prev omega roll :",
            float(controller.controller.prev_omega_roll),
        )
        print(
            "prev omega pitch:",
            float(controller.controller.prev_omega_pitch),
        )
        print(
            "i_error_xyz     :",
            float(controller.controller.i_error_x),
            float(controller.controller.i_error_y),
            float(controller.controller.i_error_z),
        )
        print("=" * 100)

    # =================================================================
    # B3_BEHAVIOR_GEOMETRIC_CONTROLLER_V1
    # =================================================================
    b3_behavior_geom_enabled = (
        os.environ.get(
            "B3_BEHAVIOR_GEOM_ENABLE",
            "0",
        ).strip() == "1"
    )

    b3_behavior_controller = None

    if b3_behavior_geom_enabled:
        from controllers.b3_behavior_geometric_controller import (
            B3BehaviorGeometricController,
        )

        if B3_UNIFIED_ACTUATOR_ENABLE:
            raise RuntimeError(
                "Behavior geometric mode must not be combined "
                "with B3_UNIFIED_ACTUATOR_ENABLE."
            )

        if B3_DIFF_ACTUATOR_ENABLE:
            raise RuntimeError(
                "Behavior geometric mode must not be combined "
                "with B3_DIFF_ACTUATOR_ENABLE."
            )

        b3_behavior_controller = (
            B3BehaviorGeometricController(
                device=device,
            )
        )

        # Explicitly clear any stale authority before first rollout.
        env._b3_behavior_direct_wrench = None

        print()
        print("=" * 100)
        print("B3 BEHAVIOR GEOMETRIC MODE ACTIVE")
        print("=" * 100)
        print(
            "goal source     : gc[0,13:16] "
            "(authoritative GC/COM goal)"
        )
        print(
            "position law    : real B3 Mellinger gains"
        )
        print(
            "attitude law    : clean SI SO(3)"
        )
        print(
            "plant authority : direct body wrench"
        )
        print(
            "PWM/motor lag   : bypassed"
        )
        print(
            "firmware SIL    : diagnostic only"
        )
        print("=" * 100)

    # =================================================================
    # Roll out Mellinger.
    # =================================================================
    body_pos_log = []
    body_quat_log = []
    body_vel_log = []
    body_ang_log = []

    ee_pos_log = []
    ee_quat_log = []

    task_goal_log = []
    gc_goal_log = []
    action_log = []

    # B3_FIRMWARE_CHAIN_TRACE_V1
    firmware_thrust_log = []
    firmware_roll_log = []
    firmware_pitch_log = []
    firmware_yaw_log = []
    firmware_motor_pwm_log = []
    firmware_motor_normalized_isaac_log = []

    b3_u_des_log = []
    b3_u_after_lag_log = []
    b3_motor_forces_log = []
    b3_applied_wrench_log = []
    motor_log = []

    terminated = False
    truncated = False

    printed_firmware_first_step = False
    printed_b3_actuator_first_step = False
    for step in range(
        mellinger_steps
    ):
        if (
            b3_real_goal_delivery_delay_steps > 0
            and step == b3_real_goal_delivery_delay_steps
            and "b3_active_relative_goal_pos" in locals()
        ):
            b3_active_relative_goal_pos.copy_(
                b3_final_relative_goal_pos
            )

            b3_active_relative_goal_ori.copy_(
                b3_final_relative_goal_ori
            )

            b3_active_relative_goal_ori_traj.copy_(
                b3_final_relative_goal_ori_traj
            )

            env.update_goal_state()
            obs_dict = env._get_observations()

            print()
            print("=" * 100)
            print("B3 REAL GOAL DELIVERED")
            print("=" * 100)
            print(
                "rollout time:",
                step / MELLINGER_RATE_HZ,
                "s",
            )
            print(
                "delivered EE goal:",
                b3_active_relative_goal_pos
                .detach()
                .cpu(),
            )
            print("=" * 100)

        gc = obs_dict["gc"]

        # During the real-goal delivery hold, make sure neither the EE
        # goal nor its GC COM representation is silently regenerated.
        if (
            b3_real_goal_delivery_delay_steps > 0
            and step < b3_real_goal_delivery_delay_steps
            and step % 8 == 0
            and "_held_ee" in locals()
        ):
            _ee_drift = float(
                torch.norm(
                    env._desired_pos_w[0]
                    - _held_ee
                ).item()
            )

            _gc_drift = float(
                torch.norm(
                    gc[0, 13:16]
                    - _expected_held_com_goal[0]
                ).item()
            )

            _gc_yaw_now = float(
                gc[0, 16].item()
            )

            _gc_yaw_drift = abs(
                math.atan2(
                    math.sin(
                        _gc_yaw_now
                        - b3_prehover_yaw
                    ),
                    math.cos(
                        _gc_yaw_now
                        - b3_prehover_yaw
                    ),
                )
            )

            if (
                max(
                    _ee_drift,
                    _gc_drift,
                    _gc_yaw_drift,
                )
                > 1.0e-4
            ):
                raise RuntimeError(
                    "B3_REAL_GOAL_DELIVERY_DELAY_V3: "
                    f"goal drift during delay at step={step}: "
                    f"ee={_ee_drift:.6e} m, "
                    f"gc={_gc_drift:.6e} m, "
                    f"yaw={_gc_yaw_drift:.6e} rad"
                )

        body_pos, body_quat, body_vel, body_ang = (
            env.get_frame_state_from_task(
                "body"
            )
        )

        if bool(
            getattr(
                env.cfg,
                "has_end_effector",
                False,
            )
        ):
            (
                ee_pos,
                ee_quat,
                _,
                _,
            ) = env.get_frame_state_from_task(
                "endeffector"
            )
        else:
            ee_pos = body_pos
            ee_quat = body_quat

        body_pos_log.append(
            body_pos[0].detach().cpu()
        )
        body_quat_log.append(
            body_quat[0].detach().cpu()
        )
        body_vel_log.append(
            body_vel[0].detach().cpu()
        )
        body_ang_log.append(
            body_ang[0].detach().cpu()
        )

        ee_pos_log.append(
            ee_pos[0].detach().cpu()
        )
        ee_quat_log.append(
            ee_quat[0].detach().cpu()
        )

        task_goal_log.append(
            env._desired_pos_w[0]
            .detach()
            .cpu()
        )

        gc_goal_log.append(
            gc[0, 13:16]
            .detach()
            .cpu()
        )

        # Behavior-geometric control has direct physical authority.
        # The environment still requires a four-dimensional action,
        # but its motor interpretation is irrelevant in this mode.
        if b3_behavior_geom_enabled:
            action = torch.zeros_like(
                env._motor_speeds_des
            )
        else:
            action, _ = (
                controller.get_action_from_gc(
                    gc.clone(),
                    device=device,
                )
            )

        # =============================================================
        # B3_BEHAVIOR_GEOMETRIC_WRENCH_STEP_V1
        # =============================================================
        if b3_behavior_geom_enabled:
            _behavior_ang_vel_w = (
                env._robot.data.root_ang_vel_w[0]
            )

            _behavior_out = (
                b3_behavior_controller.step(
                    pos_w=body_pos[0],
                    quat_wxyz=body_quat[0],
                    lin_vel_w=body_vel[0],
                    ang_vel_w=_behavior_ang_vel_w,
                    # This is the authoritative COM/GC controller goal.
                    goal_pos_w=gc[0, 13:16],
                    goal_yaw_rad=float(
                        gc[0, 16].item()
                    ),
                )
            )

            env._b3_behavior_direct_wrench = (
                _behavior_out[
                    "wrench_body"
                ]
                .view(1, 4)
            )

            if step == 0:
                print()
                print("=" * 100)
                print(
                    "B3 BEHAVIOR CONTROLLER FIRST STEP"
                )
                print("=" * 100)
                print(
                    "body pos W     :",
                    body_pos[0].detach().cpu(),
                )
                print(
                    "GC goal W      :",
                    gc[0, 13:16].detach().cpu(),
                )
                print(
                    "position error :",
                    _behavior_out[
                        "position_error_w"
                    ].detach().cpu(),
                )
                print(
                    "target force W :",
                    _behavior_out[
                        "target_force_w"
                    ].detach().cpu(),
                    "N",
                )
                print(
                    "desired tilt   :",
                    float(
                        torch.rad2deg(
                            _behavior_out[
                                "desired_tilt_rad"
                            ]
                        ).cpu()
                    ),
                    "deg",
                )
                print(
                    "current tilt   :",
                    float(
                        torch.rad2deg(
                            _behavior_out[
                                "current_tilt_rad"
                            ]
                        ).cpu()
                    ),
                    "deg",
                )
                print(
                    "thrust         :",
                    float(
                        _behavior_out[
                            "thrust_N"
                        ].cpu()
                    ),
                    "N",
                )
                print(
                    "torque         :",
                    _behavior_out[
                        "torque_Nm"
                    ].detach().cpu(),
                    "Nm",
                )
                print(
                    "omega body     :",
                    _behavior_out[
                        "omega_b"
                    ].detach().cpu(),
                    "rad/s",
                )
                print(
                    "eR             :",
                    _behavior_out[
                        "eR"
                    ].detach().cpu(),
                )
                print("=" * 100)

        else:
            env._b3_behavior_direct_wrench = None

        action_log.append(
            action[0].detach().cpu()
        )

        obs_dict, reward, term, trunc, info = (
            envs_gym.step(
                action
            )
        )

        if (
            not b3_behavior_geom_enabled
            and not printed_b3_actuator_first_step
        ):
            printed_b3_actuator_first_step = True

            print()
            print("=" * 100)
            print("B3 SYSID ACTUATOR FIRST APPLIED STEP")
            print("=" * 100)
            print(
                "u_des:",
                env._b3_sysid_last_u_des[0]
                .detach()
                .cpu(),
            )
            print(
                "u after lag:",
                env._b3_sysid_last_u[0]
                .detach()
                .cpu(),
            )
            print(
                "motor forces [N]:",
                env._b3_sysid_last_motor_forces[0]
                .detach()
                .cpu(),
            )
            print(
                "applied wrench [N,Nm]:",
                env._b3_sysid_last_wrench[0]
                .detach()
                .cpu(),
            )
            print("=" * 100)

        # Physical actuation trace.
        #
        # Behavior-geometric mode has no PWM command, lagged motor
        # command, or per-motor force state.  Its actuator output is
        # directly the body wrench applied to PhysX.
        if b3_behavior_geom_enabled:
            b3_applied_wrench_log.append(
                env._b3_sysid_last_wrench[0]
                .detach()
                .cpu()
                .clone()
            )
        else:
            b3_u_des_log.append(
                env._b3_sysid_last_u_des[0]
                .detach()
                .cpu()
                .clone()
            )
            b3_u_after_lag_log.append(
                env._b3_sysid_last_u[0]
                .detach()
                .cpu()
                .clone()
            )
            b3_motor_forces_log.append(
                env._b3_sysid_last_motor_forces[0]
                .detach()
                .cpu()
                .clone()
            )
            b3_applied_wrench_log.append(
                env._b3_sysid_last_wrench[0]
                .detach()
                .cpu()
                .clone()
            )

        motor_log.append(
            env._motor_speeds[0]
            .detach()
            .cpu()
        )

        terminated = bool(
            term[0].item()
        )
        truncated = bool(
            trunc[0].item()
        )

        if terminated or truncated:
            print(
                "[Mellinger replay] first rollout ended at "
                f"step={step + 1}, "
                f"t={(step + 1) / MELLINGER_RATE_HZ:.4f}s, "
                f"terminated={terminated}, "
                f"truncated={truncated}"
            )
            break

    actual_mellinger_steps = len(
        body_pos_log
    )

    if actual_mellinger_steps == 0:
        raise RuntimeError(
            "Mellinger replay produced zero states."
        )

    body_pos_tensor = torch.stack(
        body_pos_log
    )
    body_quat_tensor = torch.stack(
        body_quat_log
    )
    body_vel_tensor = torch.stack(
        body_vel_log
    )
    body_ang_tensor = torch.stack(
        body_ang_log
    )

    ee_pos_tensor = torch.stack(
        ee_pos_log
    )
    ee_quat_tensor = torch.stack(
        ee_quat_log
    )

    task_goal_tensor = torch.stack(
        task_goal_log
    )
    gc_goal_tensor = torch.stack(
        gc_goal_log
    )
    action_tensor = torch.stack(
        action_log
    )
    motor_tensor = torch.stack(
        motor_log
    )

    # Translate replay positions back into the ORIGINAL RL world's
    # coordinates so the overlaid position plots line up exactly with
    # the existing RL graph.
    replay_to_original_translation = (
        old_origin
        - new_origin.detach().cpu().to(
            torch.float64
        )
    ).to(torch.float32)

    body_pos_original_world = (
        body_pos_tensor
        + replay_to_original_translation
    )

    ee_pos_original_world = (
        ee_pos_tensor
        + replay_to_original_translation
    )

    task_goal_original_world = (
        task_goal_tensor
        + replay_to_original_translation
    )

    gc_goal_original_world = (
        gc_goal_tensor
        + replay_to_original_translation
    )

    mellinger_trace_path = (
        output_dir
        / "mellinger_trace.pt"
    )

    torch.save(
        {
            "metadata": {
                "source_rl_robot_index": (
                    robot_index
                ),
                "controller": (
                    "bitcraze_controller_mellinger_power_distribution_sil"
                ),
                "mellinger_rate_hz": (
                    MELLINGER_RATE_HZ
                ),
                "rl_rate_hz": (
                    rl_rate_hz
                ),
                "requested_physical_duration_s": (
                    physical_duration_s
                ),
                "actual_steps": (
                    actual_mellinger_steps
                ),
                "terminated": (
                    terminated
                ),
                "truncated": (
                    truncated
                ),
                "physical_latency_seconds": (
                    realized_delay_s
                ),
                "queue_initialization": (
                    "SRT hold command corresponding to "
                    "captured physical motor state; RL CTBR "
                    "queue contents intentionally not transplanted"
                ),
                "b3_controller_mass_kg": (
                    B3_CONTROLLER_MASS_KG
                ),
                "b3_controller_k_eta": (
                    B3_CONTROLLER_K_ETA
                ),
                "b3_controller_mass_thrust": (
                    B3_CONTROLLER_MASS_THRUST
                ),
                "b3_controller_idle_thrust": (
                    B3_CONTROLLER_IDLE_THRUST
                ),
                "b3_controller_kd_omega_rp": (
                    B3_CONTROLLER_KD_OMEGA_RP
                ),
            },
            "body_position_replay_world": (
                body_pos_tensor
            ),
            "body_position_original_rl_world": (
                body_pos_original_world
            ),
            "body_quaternion_wxyz": (
                body_quat_tensor
            ),
            "body_linear_velocity": (
                body_vel_tensor
            ),
            "body_angular_velocity_radps": (
                body_ang_tensor
            ),
            "ee_position_replay_world": (
                ee_pos_tensor
            ),
            "ee_position_original_rl_world": (
                ee_pos_original_world
            ),
            "ee_quaternion_wxyz": (
                ee_quat_tensor
            ),
            "task_goal_original_rl_world": (
                task_goal_original_world
            ),
            "gc_goal_original_rl_world": (
                gc_goal_original_world
            ),
            "actions_srt": (
                action_tensor
            ),
            "actual_motor_state": (
                motor_tensor
            ),
            # B3_FIRMWARE_CHAIN_TRACE_V1
            "firmware_control_thrust": torch.tensor(
                firmware_thrust_log,
                dtype=torch.float32,
            ),
            "firmware_control_roll": torch.tensor(
                firmware_roll_log,
                dtype=torch.float32,
            ),
            "firmware_control_pitch": torch.tensor(
                firmware_pitch_log,
                dtype=torch.float32,
            ),
            "firmware_control_yaw": torch.tensor(
                firmware_yaw_log,
                dtype=torch.float32,
            ),
            "firmware_motor_pwm": (torch.stack(firmware_motor_pwm_log, dim=0) if firmware_motor_pwm_log else torch.empty((0, 4), dtype=torch.float32)),
            "firmware_motor_normalized_isaac": (torch.stack(firmware_motor_normalized_isaac_log, dim=0) if firmware_motor_normalized_isaac_log else torch.empty((0, 4), dtype=torch.float32)),
            "b3_actuator_u_des": (
                torch.stack(
                    b3_u_des_log,
                    dim=0,
                )
                if b3_u_des_log
                else torch.empty(
                    (0, 4),
                    dtype=torch.float32,
                )
            ),
            "b3_actuator_u_after_lag": (
                torch.stack(
                    b3_u_after_lag_log,
                    dim=0,
                )
                if b3_u_after_lag_log
                else torch.empty(
                    (0, 4),
                    dtype=torch.float32,
                )
            ),
            "b3_motor_forces_N": (
                torch.stack(
                    b3_motor_forces_log,
                    dim=0,
                )
                if b3_motor_forces_log
                else torch.empty(
                    (0, 4),
                    dtype=torch.float32,
                )
            ),
            "b3_applied_wrench": (
                torch.stack(
                    b3_applied_wrench_log,
                    dim=0,
                )
                if b3_applied_wrench_log
                else torch.empty(
                    (0, 4),
                    dtype=torch.float32,
                )
            ),
        },
        mellinger_trace_path,
    )

    # Close first so RecordVideo flushes the MP4 before we report files.
    envs_gym.close()

    # =================================================================
    # RL + Mellinger 4 x 3 state overlay.
    # =================================================================
    rl_full_state = (
        RL_TRACE["full_state"]
        .detach()
        .cpu()
        .to(torch.float32)
    )

    rl_steps_available = min(
        int(RL_TRACE["steps"]),
        int(rl_full_state.shape[0]),
        int(
            round(
                physical_duration_s
                * rl_rate_hz
            )
        ),
    )

    rl_full_state = (
        rl_full_state[
            :rl_steps_available
        ]
    )

    if rl_full_state.shape[1] < 16:
        raise RuntimeError(
            "RL full_state is too short for the expected "
            "Crazyflie state layout."
        )

    rl_body_pos = (
        rl_full_state[:, 0:3]
        .numpy()
    )
    rl_body_quat = (
        rl_full_state[:, 3:7]
        .numpy()
    )
    rl_body_vel = (
        rl_full_state[:, 7:10]
        .numpy()
    )
    rl_body_ang = (
        rl_full_state[:, 10:13]
        .numpy()
    )
    rl_ee_pos = (
        rl_full_state[:, 13:16]
        .numpy()
    )

    m_body_pos = (
        body_pos_original_world
        .numpy()
    )
    m_body_quat = (
        body_quat_tensor.numpy()
    )
    m_body_vel = (
        body_vel_tensor.numpy()
    )
    m_body_ang = (
        body_ang_tensor.numpy()
    )
    m_ee_pos = (
        ee_pos_original_world
        .numpy()
    )

    rl_rpy = rpy_from_quaternion_np(
        rl_body_quat
    )
    m_rpy = rpy_from_quaternion_np(
        m_body_quat
    )

    rl_time = (
        np.arange(
            rl_steps_available,
            dtype=np.float64,
        )
        / rl_rate_hz
    )

    m_time = (
        np.arange(
            actual_mellinger_steps,
            dtype=np.float64,
        )
        / MELLINGER_RATE_HZ
    )

    task_goal_original_np = (
        desired_pos_original.numpy()
    )

    # =================================================================
    # COMMON_BENCHMARK_METRICS_V1
    #
    # Frozen controller-independent evaluation contract.
    # =================================================================
    XY_SETTLE_POSITION_THRESHOLD_M = 0.15
    SETTLE_3D_POSITION_THRESHOLD_M = 0.25
    SETTLE_SPEED_THRESHOLD_MPS = 0.20
    SETTLE_HOLD_TIME_S = 0.50

    initial_ee_original_np = (
        initial["ee_pos_w"]
        .detach()
        .cpu()
        .numpy()
        .reshape(3)
    )


    deployment_relative_goal_active = (
        "mellinger_task_goal_original"
        in locals()
    )

    commanded_ee_displacement_np = (
        task_goal_original_np
        - initial_ee_original_np
    )

    if deployment_relative_goal_active:
        mellinger_task_goal_original_np = (
            mellinger_task_goal_original
            .numpy()
            .reshape(3)
        )
        mellinger_initial_ee_original_np = (
            mellinger_initial_ee_original
            .numpy()
            .reshape(3)
        )
    else:
        mellinger_task_goal_original_np = (
            task_goal_original_np.copy()
        )
        mellinger_initial_ee_original_np = (
            initial_ee_original_np.copy()
        )

    benchmark_contract_version = (
        "DEPLOYMENT_RELATIVE_EE_TRANSFER_V1"
        if deployment_relative_goal_active
        else "STATIC_EE_TRANSFER_V1"
    )

    rl_metrics = _compute_common_metrics(
        ee_position=rl_ee_pos,
        body_velocity=rl_body_vel,
        body_quaternion=rl_body_quat,
        goal_position=task_goal_original_np,
        initial_ee_position=initial_ee_original_np,
        dt=1.0 / rl_rate_hz,
        xy_position_threshold=(
            XY_SETTLE_POSITION_THRESHOLD_M
        ),
        position_3d_threshold=(
            SETTLE_3D_POSITION_THRESHOLD_M
        ),
        speed_threshold=(
            SETTLE_SPEED_THRESHOLD_MPS
        ),
        hold_time=(
            SETTLE_HOLD_TIME_S
        ),
    )

    mellinger_metrics = _compute_common_metrics(
        ee_position=m_ee_pos,
        body_velocity=m_body_vel,
        body_quaternion=m_body_quat,
        goal_position=mellinger_task_goal_original_np,
        initial_ee_position=mellinger_initial_ee_original_np,
        dt=1.0 / MELLINGER_RATE_HZ,
        xy_position_threshold=(
            XY_SETTLE_POSITION_THRESHOLD_M
        ),
        position_3d_threshold=(
            SETTLE_3D_POSITION_THRESHOLD_M
        ),
        speed_threshold=(
            SETTLE_SPEED_THRESHOLD_MPS
        ),
        hold_time=(
            SETTLE_HOLD_TIME_S
        ),
    )

    rl_expected_steps = int(
        round(
            physical_duration_s
            * rl_rate_hz
        )
    )

    rl_trace_early_end = bool(
        int(RL_TRACE["steps"])
        < rl_expected_steps
    )

    mellinger_early_end = bool(
        actual_mellinger_steps
        < mellinger_steps
    )

    benchmark_manifest = {
        "benchmark_contract_version": (
            benchmark_contract_version
        ),

        "source": {
            "case_path": str(
                args_cli.case_path
            ),
            "rl_trace_path": str(
                args_cli.rl_trace_path
            ),
            "robot_index": int(
                robot_index
            ),
        },

        "timing": {
            "benchmark_horizon_s": float(
                physical_duration_s
            ),
            "physics_rate_hz": float(
                1.0 / sim_dt
            ),
            "rl_rate_hz": float(
                rl_rate_hz
            ),
            "mellinger_rate_hz": float(
                MELLINGER_RATE_HZ
            ),
            "physical_latency_s": float(
                physical_delay_s
            ),
        },

        "settling_definition": {
            "xy_position_threshold_m": (
                XY_SETTLE_POSITION_THRESHOLD_M
            ),
            "position_3d_threshold_m": (
                SETTLE_3D_POSITION_THRESHOLD_M
            ),
            "ee_speed_threshold_mps": (
                SETTLE_SPEED_THRESHOLD_MPS
            ),
            "continuous_hold_time_s": (
                SETTLE_HOLD_TIME_S
            ),
        },

        "rl_physical_trial": {
            "initial_body_pos_w": _json_value(
                initial.get("body_pos_w")
            ),
            "initial_body_quat_wxyz": _json_value(
                initial.get("body_quat_w")
                if initial.get("body_quat_w") is not None
                else initial.get("body_quat_wxyz")
            ),
            "initial_body_lin_vel_w": _json_value(
                initial.get("body_lin_vel_w")
            ),
            "initial_body_ang_vel_w": _json_value(
                initial.get("body_ang_vel_w")
                if initial.get("body_ang_vel_w") is not None
                else initial.get("body_ang_vel_w_radps")
            ),
            "initial_ee_pos_w": _json_value(
                initial.get("ee_pos_w")
            ),

            "authoritative_ee_goal_pos_w": _json_value(
                desired_pos_original
            ),
            "mellinger_com_goal_pos_w": _json_value(
                goal.get("mellinger_goal_pos_w")
            ),

            "robot_mass": _json_value(
                plant.get(
                    "robot_mass",
                    plant.get("robot_mass_kg"),
                )
            ),
            "robot_inertia": _json_value(
                plant.get("robot_inertia")
            ),
            "physx_inertias": _json_value(
                plant.get("physx_inertias")
            ),
            "k_eta": _json_value(
                plant.get(
                    "k_eta",
                    plant.get("k_eta"),
                )
            ),
            "k_torque": _json_value(
                plant.get("k_torque")
            ),
            "tau_m": _json_value(
                plant.get(
                    "tau_m",
                    plant.get("tau_m_s"),
                )
            ),
        },

        "task_transfer": {
            "type": (
                "same_relative_ee_displacement"
                if deployment_relative_goal_active
                else "same_absolute_ee_goal"
            ),
            "commanded_ee_displacement_m": _json_value(
                commanded_ee_displacement_np
            ),
            "rl_initial_ee_pos_w": _json_value(
                initial_ee_original_np
            ),
            "rl_ee_goal_pos_w": _json_value(
                task_goal_original_np
            ),
            "mellinger_initial_ee_pos_w": _json_value(
                mellinger_initial_ee_original_np
            ),
            "mellinger_ee_goal_pos_w": _json_value(
                mellinger_task_goal_original_np
            ),
        },

        "mellinger_physical_trial": {
            "plant_role": "fixed_real_b3",
            "total_mass_kg": 0.046,
            "aggregate_inertia_kg_m2": [
                [2.4255e-05, 0.0, 0.0],
                [0.0, 1.8650e-05, 0.0],
                [0.0, 0.0, 3.9300e-05],
            ],
            "arm_length_m": 0.05,
            "motor_time_constant_s": float(
                B3_SYSID_TAU_M_S
            ),
            "differential_actuator_ablation": {
                "enabled": bool(
                    B3_DIFF_ACTUATOR_ENABLE
                ),
                "collective_tau_s": float(
                    B3_SYSID_TAU_M_S
                ),
                "differential_tau_s": float(
                    B3_DIFF_TAU_M_S
                ),
                "requested_delay_s": float(
                    B3_DIFF_DELAY_S
                ),
                "realized_delay_s": float(
                    _b3_diff_realized_delay_s
                ),
                "roll_moment_gain": float(
                    B3_DIFF_ROLL_GAIN
                ),
                "pitch_moment_gain": float(
                    B3_DIFF_PITCH_GAIN
                ),
                "yaw_moment_gain": 1.0,
                "status": (
                    "EXPERIMENTAL_NONDEFAULT"
                    if B3_DIFF_ACTUATOR_ENABLE
                    else "DISABLED_BASELINE"
                ),
            },
            "unified_actuator_sysid_model": {
                "enabled": bool(
                    B3_UNIFIED_ACTUATOR_ENABLE
                ),
                "common_tau_s": float(
                    B3_UNIFIED_TAU_M_S
                ),
                "requested_delay_s": float(
                    B3_UNIFIED_DELAY_S
                ),
                "realized_delay_s": float(
                    _b3_unified_realized_delay_s
                ),
                "shared_rp_moment_gain": float(
                    B3_UNIFIED_RP_GAIN
                ),
                "yaw_moment_gain": 1.0,
                "structure": (
                    "full_four_motor_delay_then_lag"
                ),
                "status": (
                    "EXPERIMENTAL_NONDEFAULT"
                    if B3_UNIFIED_ACTUATOR_ENABLE
                    else "DISABLED_BASELINE"
                ),
            },
            "reaction_torque_coefficient": 0.003987,
            "motor_thrust_model": {
                "input": "normalized_firmware_motor_command_u",
                "equation": "F_N = linear*u + quadratic*u^2",
                "linear_N": float(
                    B3_SYSID_THRUST_LINEAR_N
                ),
                "quadratic_N": float(
                    B3_SYSID_THRUST_QUADRATIC_N
                ),
            },
            "firmware_commit": str(
                FIRMWARE_COMMIT
            ),
            "prehover_duration_s": (
                float(B3_PREHOVER_S)
                if deployment_relative_goal_active
                else None
            ),
        },

        "mellinger_controller_calibration": {
            "name": "B3-faithful",
            "mass_kg": float(
                B3_CONTROLLER_MASS_KG
            ),
            "k_eta_provenance": float(
                B3_CONTROLLER_K_ETA
            ),
            "mass_thrust": float(
                B3_CONTROLLER_MASS_THRUST
            ),
            "idle_thrust": float(
                B3_CONTROLLER_IDLE_THRUST
            ),
            "kd_omega_rp": float(
                B3_CONTROLLER_KD_OMEGA_RP
            ),
            "motor_command_scale": 1.0,
        },

        "replay_validation": {
            "initial_body_pos_error": float(
                pos_error
            ),
            "initial_body_quat_error": float(
                quat_error
            ),
            "initial_body_vel_error": float(
                vel_error
            ),
            "initial_body_ang_error": float(
                ang_error
            ),
            "gc_goal_error": float(
                gc_goal_error
            ),
            "task_goal_error": float(
                desired_goal_error
            ),
        },

        "outcome": {
            "rl_trace_early_end": (
                rl_trace_early_end
            ),
            "mellinger_early_end": (
                mellinger_early_end
            ),
            "mellinger_terminated": bool(
                terminated
            ),
            "mellinger_truncated": bool(
                truncated
            ),
        },

        "metrics": {
            "rl": rl_metrics,
            "mellinger": mellinger_metrics,
        },
    }

    benchmark_metrics_path = (
        output_dir
        / "benchmark_metrics.json"
    )

    with benchmark_metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            benchmark_manifest,
            file,
            indent=2,
            sort_keys=True,
        )

    print()
    print("=" * 100)
    print("COMMON APPLES-TO-APPLES BENCHMARK METRICS")
    print("=" * 100)

    for controller_name, metrics in (
        ("RL", rl_metrics),
        ("Mellinger", mellinger_metrics),
    ):
        print(
            f"{controller_name:10s} "
            f"min3D={metrics['minimum_3d_error_m']:.4f} m  "
            f"final3D={metrics['final_3d_error_m']:.4f} m  "
            f"finalXY={metrics['final_xy_error_m']:.4f} m  "
            f"finalZ={metrics['final_abs_z_error_m']:.4f} m"
        )
        print(
            f"{'':10s} "
            f"peakEEv={metrics['peak_ee_speed_mps']:.4f} m/s  "
            f"peakTilt={metrics['peak_tilt_deg']:.2f} deg  "
            f"XYsettled={metrics['xy_settled_success']}  "
            f"tXY={metrics['xy_settling_time_s']}  "
            f"3Dsettled={metrics['settled_3d_success']}  "
            f"t3D={metrics['settling_3d_time_s']}"
        )

    print(
        "benchmark manifest :",
        benchmark_metrics_path,
    )
    print("=" * 100)

    fig = plt.figure(
        figsize=(15, 14)
    )

    position_names = [
        "X",
        "Y",
        "Z",
    ]

    # Row 1: positions.
    for axis in range(3):
        plt.subplot(
            4,
            3,
            axis + 1,
        )

        plt.plot(
            rl_time,
            rl_body_pos[:, axis],
            label="RL Quad",
        )
        plt.plot(
            rl_time,
            rl_ee_pos[:, axis],
            "--",
            label="RL EE",
        )
        plt.plot(
            m_time,
            m_body_pos[:, axis],
            label="Mellinger Quad",
        )
        plt.plot(
            m_time,
            m_ee_pos[:, axis],
            "--",
            label="Mellinger EE",
        )
        plt.plot(
            rl_time,
            np.full(
                rl_time.shape,
                task_goal_original_np[axis],
            ),
            ":",
            label="RL Goal",
        )
        plt.plot(
            m_time,
            np.full(
                m_time.shape,
                mellinger_task_goal_original_np[axis],
            ),
            ":",
            label="Mellinger Goal",
        )

        plt.title(
            f"Position {position_names[axis]}"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend(loc="best")

    # Row 2: linear velocity.
    for axis in range(3):
        plt.subplot(
            4,
            3,
            4 + axis,
        )

        plt.plot(
            rl_time,
            rl_body_vel[:, axis],
            label=f"RL Quad Vel {position_names[axis]}",
        )
        plt.plot(
            m_time,
            m_body_vel[:, axis],
            label=f"Mellinger Quad Vel {position_names[axis]}",
        )

        plt.title(
            f"Linear Velocity {position_names[axis]}"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.legend(loc="best")

    # Row 3: Euler attitude.
    attitude_names = [
        "Roll",
        "Pitch",
        "Yaw",
    ]

    for axis in range(3):
        plt.subplot(
            4,
            3,
            7 + axis,
        )

        plt.plot(
            rl_time,
            rl_rpy[:, axis],
            label=f"RL Quad {attitude_names[axis]}",
        )
        plt.plot(
            m_time,
            m_rpy[:, axis],
            label=f"Mellinger Quad {attitude_names[axis]}",
        )

        plt.title(
            attitude_names[axis]
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (rad)")
        plt.legend(loc="best")

    # Row 4: angular velocity.
    for axis in range(3):
        plt.subplot(
            4,
            3,
            10 + axis,
        )

        plt.plot(
            rl_time,
            rl_body_ang[:, axis],
            label=f"RL Quad Ang Vel {position_names[axis]}",
        )
        plt.plot(
            m_time,
            m_body_ang[:, axis],
            label=f"Mellinger Quad Ang Vel {position_names[axis]}",
        )

        plt.title(
            f"Angular Velocity {position_names[axis]}"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Angular velocity (rad/s)")
        plt.legend(loc="best")

    fig.suptitle(
        f"RL vs Frozen Mellinger — Robot {robot_index}",
        fontsize=14,
    )

    plt.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.97,
        ]
    )

    overlay_path = (
        output_dir
        / f"rl_vs_mellinger_robot_{robot_index}.png"
    )

    plt.savefig(
        overlay_path,
        dpi=150,
    )

    plt.close(fig)

    # =================================================================
    # MELLINGER_COMPARE_XY_DISTANCE_PLOT_V1
    #
    # Apples-to-apples horizontal distance of each END EFFECTOR to the
    # same authoritative EE goal.
    #
    #     d_xy = sqrt((x_ee - x_goal)^2 + (y_ee - y_goal)^2)
    #
    # Do not use body position here: the task and RL policy are defined
    # in the EE frame.
    # =================================================================
    rl_xy_distance_to_goal = np.linalg.norm(
        rl_ee_pos[:, 0:2]
        - task_goal_original_np[None, 0:2],
        axis=1,
    )

    m_xy_distance_to_goal = np.linalg.norm(
        m_ee_pos[:, 0:2]
        - mellinger_task_goal_original_np[None, 0:2],
        axis=1,
    )

    xy_fig = plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        rl_time,
        rl_xy_distance_to_goal,
        label="RL EE",
    )

    plt.plot(
        m_time,
        m_xy_distance_to_goal,
        label="Mellinger EE",
    )

    plt.axhline(
        0.0,
        linestyle=":",
        label="Goal",
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "XY distance to EE goal (m)"
    )

    plt.title(
        f"RL vs Frozen Mellinger — "
        f"XY Distance to Goal — Robot {robot_index}"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend(
        loc="best"
    )

    plt.tight_layout()

    xy_distance_path = (
        output_dir
        / (
            f"rl_vs_mellinger_robot_"
            f"{robot_index}_xy_distance_to_goal.png"
        )
    )

    plt.savefig(
        xy_distance_path,
        dpi=150,
    )

    plt.close(
        xy_fig
    )

    mp4_files = sorted(
        output_dir.glob(
            f"mellinger_robot_{robot_index}*.mp4"
        )
    )

    print()
    print("=" * 100)
    print("RL vs MELLINGER COMPARISON COMPLETE")
    print("=" * 100)
    print("comparison directory :", output_dir)
    print("Mellinger trace      :", mellinger_trace_path)
    print("overlay graph        :", overlay_path)
    print("XY distance graph    :", xy_distance_path)

    if args_cli.video:
        if mp4_files:
            for path in mp4_files:
                print(
                    "Mellinger video      :",
                    path,
                )
        else:
            print(
                "Mellinger video      : "
                "RecordVideo requested; no MP4 was found after close."
            )
    else:
        print(
            "Mellinger video      : disabled "
            "(original RL run did not use --video)"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
    simulation_app.close()
