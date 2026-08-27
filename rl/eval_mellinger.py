import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Evaluate firmware-style Mellinger controller in Isaac."
)
parser.add_argument("--task", type=str, default="Isaac-Crazyflie-0DOF-Hover-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=2500)
parser.add_argument("--print_every", type=int, default=50)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record the Mellinger evaluation as an MP4.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=1500,
    help="Number of environment steps to record.",
)
parser.add_argument(
    "--zero_latency",
    action="store_true",
    help="Force the environment action latency to zero for controller validation.",
)
parser.add_argument(
    "--hold_initial_pose",
    action="store_true",
    help="Make Mellinger hold the initial position and initial yaw.",
)
parser.add_argument(
    "--sign_probe",
    action="store_true",
    help="Run static Mellinger -> Isaac physical torque sign tests and stop.",
)
parser.add_argument(
    "--kd_omega_rp",
    type=float,
    default=200.0,
    help="Override Mellinger roll/pitch angular-acceleration D gain.",
)

parser.add_argument(
    "--desired_yaw_deg",
    type=float,
    default=0.0,
    help=(
        "Mellinger desired yaw setpoint in degrees. "
        "Applied only when the benchmark step begins; "
        "pre-hover remains at the initial yaw."
    ),
)

parser.add_argument(
    "--swap_xy_inertia",
    action="store_true",
    help=(
        "Diagnostic control: swap real-B3 Ixx and Iyy "
        "in the PhysX plant only."
    ),
)
parser.add_argument(
    "--goal_offset",
    type=float,
    nargs=3,
    metavar=("DX", "DY", "DZ"),
    default=None,
    help=(
        "Command a fixed XYZ goal relative to the initial vehicle "
        "position. Example: --goal_offset 0.5 0 0"
    ),
)

parser.add_argument(
    "--goal_offset_range",
    type=float,
    nargs=3,
    default=None,
    metavar=("RX", "RY", "RZ"),
    help=(
        "Sample one fixed per-environment goal offset with "
        "dx~U(-RX,RX), dy~U(-RY,RY), dz~U(-RZ,RZ). "
        "Mutually exclusive with --goal_offset and "
        "--hold_initial_pose."
    ),
)

parser.add_argument(
    "--pre_hover_s",
    type=float,
    default=0.0,
    help=(
        "Run Mellinger on the initial hover target for this many "
        "seconds before applying --goal_offset/--goal_offset_range. "
        "The controller, motor state, and vehicle are NOT reset at "
        "the goal switch."
    ),
)

# RL_TRANSFER_CASES_REAL_B3_V2
parser.add_argument(
    "--rl_position_cases",
    type=str,
    default=None,
    help=(
        "Load an RL eval_full_states.pt tensor and reproduce only "
        "each original EE displacement: "
        "delta = RL_EE_goal - RL_EE_start. "
        "All environments remain identical fixed real-B3 plants "
        "at the normal safe benchmark hover."
    ),
)

parser.add_argument(
    "--validate_goal_sync",
    action="store_true",
    help=(
        "Assert every controller step that the environment and GC "
        "still expose the exact benchmark target."
    ),
)

parser.add_argument(
    "--follow_robot",
    type=int,
    default=-1,
    help="Robot index for follow-camera video/plots. Use 0 for a single-env run.",
)
parser.add_argument(
    "--calibration_mode",
    type=str,
    choices=("same_plant", "nominal", "real_b3"),
default="nominal",
    help=(
        "same_plant: oracle diagnostic using each randomized plant; "
        "nominal: fixed 0.04 kg / k_eta 0.338 Mellinger calibration; ""real_b3: fixed physically-calibrated B3 plant with deployed ""firmware Mellinger calibration."
    ),
)

# REAL_B3_EVAL_THRUST_CURVE_V1
parser.add_argument(
    "--real_b3_thrust_curve",
    action="store_true",
    default=False,
    help=(
        "Evaluator-local real-B3 motor thrust model identified from "
        "sysidB100: F_i = 0.187453678*u_i + "
        "0.126414663*u_i^2. "
        "Requires --calibration_mode real_b3. "
        "Does not modify quadrotor_env.py."
    ),
)

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if (
    args_cli.real_b3_thrust_curve
    and args_cli.calibration_mode != "real_b3"
):
    parser.error(
        "--real_b3_thrust_curve requires "
        "--calibration_mode real_b3"
    )

args_cli.enable_cameras = args_cli.video
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import math
import os
import csv
import json
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

# ------------------------------------------------------------------
# Force local AerialManipulation imports.
#
# When this file is launched as:
#     python rl/eval_mellinger.py
# Python places .../AerialManipulation/rl on sys.path. Another repo
# may therefore win generic imports such as `import envs`.
# ------------------------------------------------------------------
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)

# Make this repository authoritative for local package imports.
while _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)

sys.path.insert(0, _REPO_ROOT_STR)

import envs

_envs_path = Path(envs.__file__).resolve()
_expected_envs_root = (_REPO_ROOT / "envs").resolve()

if _expected_envs_root not in _envs_path.parents:
    raise RuntimeError(
        "Wrong `envs` package imported. "
        f"Expected under {_expected_envs_root}, "
        f"but imported {_envs_path}"
    )

print(
    "[Evaluator] Local env package:",
    _envs_path,
)

from controllers.mellinger_firmware_controller import (
    MellingerFirmwareController,
)

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
    agent_cfg,
):
    # ------------------------------------------------------------
    # Environment configuration
    # ------------------------------------------------------------
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # We need the geometric-controller observation.
    env_cfg.gc_mode = True
    env_cfg.eval_mode = True

    # Mellinger will directly generate four normalized motor commands.
    env_cfg.control_mode = "SRT"

    # Same body/goal convention used by the existing Crazyflie baseline.
    env_cfg.task_body = "body"
    env_cfg.goal_body = "body"
    env_cfg.reward_task_body = "body"
    env_cfg.reward_goal_body = "body"

    # REAL_B3_FIXED_PLANT_V1
    # ------------------------------------------------------------
    # Opt-in physically calibrated B3 competence benchmark.
    # Normal nominal/same_plant behavior remains untouched.
    # ------------------------------------------------------------
    REAL_B3_MASS_KG = 0.046
    REAL_B3_IXX = 2.4255e-5
    REAL_B3_IYY = 1.8650e-5
    REAL_B3_IZZ = 3.9300e-5

    REAL_B3_ARM_LENGTH_M = 0.050
    REAL_B3_K_ETA = 0.51033
    REAL_B3_K_M = 7.8e-10
    REAL_B3_K_TORQUE = 3.987e-3
    REAL_B3_TAU_M_S = 0.050

    REAL_B3_MASS_THRUST = 132000.0
    REAL_B3_IDLE_THRUST = 10000.0

    if args_cli.calibration_mode == "real_b3":
        env_cfg.mass = REAL_B3_MASS_KG
        env_cfg.Ixx = REAL_B3_IXX
        env_cfg.Iyy = REAL_B3_IYY
        env_cfg.Izz = REAL_B3_IZZ

        env_cfg.arm_length = REAL_B3_ARM_LENGTH_M
        env_cfg.k_eta = REAL_B3_K_ETA
        env_cfg.k_m = REAL_B3_K_M
        env_cfg.k_torque = REAL_B3_K_TORQUE
        env_cfg.tau_m = REAL_B3_TAU_M_S

        if isinstance(
            getattr(env_cfg, "dr_dict", None),
            dict,
        ):
            env_cfg.dr_dict = {
                key: 0.0
                for key in env_cfg.dr_dict
            }

        # No artificial simulator command queue for the firmware
        # competence benchmark.
        env_cfg.control_latency_steps = 0

    # BENCHMARK_SYNC_PATCH_V1
    # ------------------------------------------------------------
    # Frozen random-position benchmark definition.
    #
    # Isaac physics : 1000 Hz
    # Mellinger     : 500 Hz
    # Start         : level/stationary at z = 3 m
    # ------------------------------------------------------------
    BENCHMARK_CONTROL_RATE_HZ = 500
    BENCHMARK_START_Z_M = 3.0

    benchmark_decimation_float = (
        1.0
        / (
            float(env_cfg.sim.dt)
            * float(BENCHMARK_CONTROL_RATE_HZ)
        )
    )

    benchmark_decimation = int(
        round(benchmark_decimation_float)
    )

    if abs(
        benchmark_decimation_float
        - benchmark_decimation
    ) > 1.0e-9:
        raise RuntimeError(
            "Cannot represent the requested Mellinger rate exactly: "
            f"sim_dt={env_cfg.sim.dt}, "
            f"required_decimation={benchmark_decimation_float}"
        )

    env_cfg.policy_rate_hz = BENCHMARK_CONTROL_RATE_HZ
    env_cfg.decimation = benchmark_decimation
    env_cfg.sim.render_interval = benchmark_decimation

    # RL_TRANSFER_CASES_REAL_B3_V2
    #
    # args_cli.steps is the POST-STEP measured rollout.
    # Pre-hover must therefore be added to the Isaac episode horizon.
    if args_cli.rl_position_cases is not None:
        if args_cli.pre_hover_s <= 0.0:
            raise ValueError(
                "--rl_position_cases requires positive pre-hover. "
                "Use --pre_hover_s 2.0."
            )

        measured_duration_s = (
            float(args_cli.steps)
            / float(BENCHMARK_CONTROL_RATE_HZ)
        )

        env_cfg.episode_length_s = (
            float(args_cli.pre_hover_s)
            + measured_duration_s
            + 1.0 / float(BENCHMARK_CONTROL_RATE_HZ)
        )

        print()
        print("=" * 100)
        print("RL-TRANSFER BENCHMARK HORIZON")
        print("=" * 100)
        print(
            "pre-hover duration :",
            f"{float(args_cli.pre_hover_s):.6f} s",
        )
        print(
            "measured duration  :",
            f"{measured_duration_s:.6f} s",
        )
        print(
            "environment horizon:",
            f"{float(env_cfg.episode_length_s):.6f} s",
        )
        print("=" * 100)

    # Static trajectory used only to establish the safe reset state.
    env_cfg.trajectory_type = "lissaajous"
    env_cfg.trajectory_horizon = 0
    env_cfg.random_shift_trajectory = False

    env_cfg.lissajous_amplitudes = [
        0.0, 0.0, 0.0, 0.0
    ]
    env_cfg.lissajous_amplitudes_rand_ranges = [
        0.0, 0.0, 0.0, 0.0
    ]

    env_cfg.lissajous_frequencies = [
        0.0, 0.0, 0.0, 0.0
    ]
    env_cfg.lissajous_frequencies_rand_ranges = [
        0.0, 0.0, 0.0, 0.0
    ]

    env_cfg.lissajous_phases = [
        0.0, 0.0, 0.0, 0.0
    ]
    env_cfg.lissajous_phases_rand_ranges = [
        0.0, 0.0, 0.0, 0.0
    ]

    env_cfg.lissajous_offsets = [
        0.0, 0.0, BENCHMARK_START_Z_M, 0.0
    ]
    env_cfg.lissajous_offsets_rand_ranges = [
        0.0, 0.0, 0.0, 0.0
    ]

    # quadrotor_env.py's init_cfg="fixed" branch deliberately injects
    # -360 deg/s roll. Instead use the random-reset branch with all
    # randomization ranges exactly zero.
    env_cfg.init_cfg = "rand"
    env_cfg.init_pos_ranges = [0.0, 0.0, 0.0]
    env_cfg.init_lin_vel_ranges = [0.0, 0.0, 0.0]
    env_cfg.init_yaw_ranges = [0.0]
    env_cfg.init_ang_vel_ranges = [0.0, 0.0, 0.0]

    # Benchmark termination is ground collision + timeout.
    # rotorpy_done=True additionally compares against _pos_traj,
    # which is not our authoritative randomized position target.
    env_cfg.rotorpy_done = False

    print("\n" + "=" * 100)
    print("MELLINGER EVALUATION")
    print("=" * 100)
    print("Task:          ", args_cli.task)
    print("Num envs:      ", args_cli.num_envs)
    print("Seed:          ", args_cli.seed)
    print("Control mode:  ", env_cfg.control_mode)
    print("Policy rate:   ", env_cfg.policy_rate_hz, "Hz")
    print("Simulation dt: ", env_cfg.sim.dt)
    print("Decimation:     ", env_cfg.decimation)
    print(
        "Physical rate:  ",
        1.0
        / (
            float(env_cfg.sim.dt)
            * float(env_cfg.decimation)
        ),
        "Hz",
    )
    print("=" * 100)

    # MELLINGER_RUN_OUTPUT_SETUP
    latency_label = "zero_latency" if args_cli.zero_latency else "sampled_latency"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"mellinger_{args_cli.calibration_mode}"
        f"_seed{args_cli.seed}"
        f"_n{args_cli.num_envs}"
        f"_{latency_label}"
        f"_{run_stamp}"
    )

    run_dir = os.path.join(
        "/home/sumukh/AerialManipulation/videos/eval_mellinger",
        run_name,
    )
    os.makedirs(run_dir, exist_ok=True)

    print("\nRUN OUTPUT")
    print("-" * 100)
    print("run name :", run_name)
    print("run dir  :", run_dir)
    print("-" * 100)

    if args_cli.follow_robot >= 0:
        if args_cli.follow_robot >= env_cfg.scene.num_envs:
            raise ValueError(
                f"--follow_robot={args_cli.follow_robot}, but only "
                f"{env_cfg.scene.num_envs} environments exist."
            )

        # Match eval_rslrl.py follow-camera convention.
        env_cfg.viewer.eye = (0.75, 0.75, 0.75)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        env_cfg.viewer.resolution = (1080, 1920)
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.env_index = args_cli.follow_robot
        env_cfg.viewer.asset_name = "robot"

    envs_gym = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        video_folder = run_dir
        os.makedirs(video_folder, exist_ok=True)

        video_length = min(args_cli.video_length, args_cli.steps)

        print()
        print("VIDEO RECORDING")
        print("-" * 100)
        print("video folder :", video_folder)
        print("video length :", video_length, "environment steps")
        print("-" * 100)

        envs_gym = gym.wrappers.RecordVideo(
            envs_gym,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            name_prefix=run_name,
        )

    env = envs_gym.unwrapped
    device = env.device

    # ================================================================
    # REAL_B3_EVAL_THRUST_CURVE_V1
    # ================================================================
    #
    # sysidB100 identification:
    #
    #   F_i [N] =
    #       0.187453678 * u_i
    #       + 0.126414663 * u_i^2
    #
    # where u_i is the normalized Crazyflie motor command after the
    # first-order motor dynamics.
    #
    # The existing environment source remains completely untouched.
    # This replaces _apply_action ONLY on this evaluator's environment
    # instance and ONLY when --real_b3_thrust_curve is explicitly used.
    #
    # Everything else remains the existing plant:
    #
    #   - action queue / latency
    #   - SRT action conversion
    #   - motor first-order state
    #   - tau_m
    #   - rotor positions
    #   - force -> roll/pitch allocation
    #   - k_torque yaw relation
    #   - PhysX rigid-body dynamics
    #
    # ================================================================
    if args_cli.real_b3_thrust_curve:
        if args_cli.calibration_mode != "real_b3":
            raise RuntimeError(
                "Real-B3 thrust curve requires "
                "--calibration_mode real_b3."
            )

        if str(env.cfg.control_mode) != "SRT":
            raise RuntimeError(
                "Real-B3 evaluator thrust curve requires "
                'control_mode="SRT".'
            )

        if bool(
            getattr(
                env.cfg,
                "skip_motor_dynamics",
                False,
            )
        ):
            raise RuntimeError(
                "Real-B3 evaluator thrust curve requires "
                "motor dynamics to remain enabled."
            )

        motor_speed_min = float(
            env.cfg.motor_speed_min
        )

        motor_speed_max = float(
            env.cfg.motor_speed_max
        )

        if abs(motor_speed_min) > 1.0e-9:
            raise RuntimeError(
                "Expected normalized SRT motor minimum 0.0, "
                f"got {motor_speed_min}."
            )

        if abs(motor_speed_max - 1.0) > 1.0e-9:
            raise RuntimeError(
                "Expected normalized SRT motor maximum 1.0, "
                f"got {motor_speed_max}."
            )

        REAL_B3_THRUST_K1_N = 0.187453678
        REAL_B3_THRUST_K2_N = 0.126414663

        # Keep a reference solely for provenance/debugging.
        # It is NOT called while the override is active.
        _original_apply_action = env._apply_action

        def _real_b3_eval_apply_action():
            # This evaluator uses SRT exclusively. Refuse to silently
            # reinterpret CTBR/CTATT behavior if configuration changes.
            if str(env.cfg.control_mode) != "SRT":
                raise RuntimeError(
                    "REAL_B3_EVAL_THRUST_CURVE_V1 "
                    "received non-SRT control mode."
                )

            # Match the side effect of the original _apply_action().
            env.pd_loop_counter += 1

            # --------------------------------------------------------
            # ORIGINAL motor dynamics — unchanged.
            #
            # alpha = exp(-physics_dt / tau_m)
            #
            # u[k+1] =
            #     alpha*u[k]
            #     + (1-alpha)*u_des[k]
            # --------------------------------------------------------
            alpha = torch.exp(
                -env.physics_dt
                / env._tau_m
            ).unsqueeze(-1)

            env._motor_speeds = (
                alpha
                * env._motor_speeds
                + (
                    1.0
                    - alpha
                )
                * env._motor_speeds_des
            )

            env._motor_speeds = (
                env._motor_speeds.clamp(
                    env.cfg.motor_speed_min,
                    env.cfg.motor_speed_max,
                )
            )

            u_motor = env._motor_speeds

            # --------------------------------------------------------
            # ONLY changed physical relation:
            #
            # OLD:
            #
            #   F = k_eta * u^2
            #
            # REAL B3 sysidB100:
            #
            #   F = k1*u + k2*u^2
            #
            # Per-motor force in Newtons.
            # --------------------------------------------------------
            motor_forces = (
                REAL_B3_THRUST_K1_N
                * u_motor
                + REAL_B3_THRUST_K2_N
                * u_motor.square()
            )

            # Existing force -> total thrust / body moments allocation.
            #
            # This retains:
            #   rotor geometry
            #   arm length
            #   k_torque
            #   motor directions
            wrench = torch.bmm(
                env.f_to_TM,
                motor_forces.unsqueeze(2),
            ).squeeze(2)

            env._thrust[:, 0, 2] = (
                wrench[:, 0]
            )

            env._moment[:, 0, :] = (
                wrench[:, 1:]
            )

            env._robot.set_external_force_and_torque(
                env._thrust,
                env._moment,
                body_ids=env._body_id,
            )

        # Instance-local override. quadrotor_env.py is NOT edited.
        env._apply_action = (
            _real_b3_eval_apply_action
        )

        print()
        print("=" * 100)
        print(
            "REAL-B3 EVALUATOR-LOCAL THRUST CURVE ENABLED"
        )
        print("=" * 100)
        print(
            "environment source modified : NO"
        )
        print(
            "Mellinger source modified   : NO"
        )
        print(
            "motor dynamics retained     : YES"
        )
        print(
            "action latency retained      : YES"
        )
        print(
            "rotor allocation retained    : YES"
        )
        print()
        print(
            "identified force law:"
        )
        print(
            "F_i = "
            "0.187453678*u_i "
            "+ 0.126414663*u_i^2  [N]"
        )
        print()

        # Display the exact static difference at useful commands.
        legacy_k_eta = float(
            env._k_eta.reshape(-1)[0].item()
        )

        for u_value in (
            0.152587890625,
            0.47,
            0.72,
            0.80,
            0.90,
            1.00,
        ):
            old_force = (
                legacy_k_eta
                * u_value
                * u_value
            )

            new_force = (
                REAL_B3_THRUST_K1_N
                * u_value
                + REAL_B3_THRUST_K2_N
                * u_value
                * u_value
            )

            print(
                f"u={u_value:7.4f} | "
                f"old square={old_force:.6f} N | "
                f"real-B3 fit={new_force:.6f} N"
            )

        print("=" * 100)

    # REAL_B3_PHYSX_PLANT_V1
    # ------------------------------------------------------------
    # cfg.mass / cfg.Ixx/... initialize the environment-side model,
    # but DR is normally what writes mass/inertia into PhysX.
    # Because this benchmark intentionally disables DR, explicitly
    # make the actual rigid-body plant identical across all envs.
    # ------------------------------------------------------------
    if args_cli.calibration_mode == "real_b3":
        real_b3_env_ids = torch.arange(
            env.num_envs,
            dtype=torch.long,
            device=device,
        )

        # ----- mass -----
        env._robot_mass.fill_(REAL_B3_MASS_KG)
        env._robot_weight.copy_(
            env._robot_mass * env._gravity_magnitude
        )

        current_masses = (
            env._robot.root_physx_view
            .get_masses()
            .clone()
        )

        current_masses[
            :,
            env._body_id,
        ] = REAL_B3_MASS_KG

        env._robot.root_physx_view.set_masses(
            current_masses.cpu(),
            real_b3_env_ids.cpu(),
        )

        env._default_masses = current_masses.clone()

        if hasattr(env, "_robot_masses"):
            env._robot_masses = (
                env._robot.root_physx_view
                .get_masses()
                .to(device)
            )

        # ----- inertia -----
        current_inertias = (
            env._robot.root_physx_view
            .get_inertias()
            .clone()
        )

        real_b3_inertia_matrix = torch.diag(
            torch.tensor(
                [
                    REAL_B3_IYY if args_cli.swap_xy_inertia else REAL_B3_IXX,
                    REAL_B3_IXX if args_cli.swap_xy_inertia else REAL_B3_IYY,
                    REAL_B3_IZZ,
                ],
                dtype=current_inertias.dtype,
                device=current_inertias.device,
            )
        )

        current_inertias[
            :,
            env._body_id,
            :,
        ] = real_b3_inertia_matrix.reshape(1, 9)

        env._robot.root_physx_view.set_inertias(
            current_inertias.cpu(),
            real_b3_env_ids.cpu(),
        )

        print(
            "PHYSX inertia control : "
            f"swap_xy={bool(args_cli.swap_xy_inertia)} "
            f"Ixx={real_b3_inertia_matrix[0, 0].item():.9e} "
            f"Iyy={real_b3_inertia_matrix[1, 1].item():.9e} "
            f"Izz={real_b3_inertia_matrix[2, 2].item():.9e}"
        )

        inertia_internal = (
            real_b3_inertia_matrix
            .to(
                dtype=env.inertia_tensor.dtype,
                device=env.inertia_tensor.device,
            )
            .view(1, 3, 3)
            .expand_as(env.inertia_tensor)
        )

        env.inertia_tensor.copy_(inertia_internal)
        env._robot_inertia.copy_(inertia_internal)
        env.default_inertia = (
            env.inertia_tensor[0].clone()
        )

        # ----- actuator / rotor plant -----
        for attribute, value in (
            ("_arm_length", REAL_B3_ARM_LENGTH_M),
            ("_k_eta", REAL_B3_K_ETA),
            ("_k_m", REAL_B3_K_M),
            ("_k_torque", REAL_B3_K_TORQUE),
            ("_tau_m", REAL_B3_TAU_M_S),
        ):
            tensor = getattr(env, attribute, None)

            if not torch.is_tensor(tensor):
                raise RuntimeError(
                    f"real_b3 requires environment tensor {attribute}"
                )

            tensor.fill_(value)

        if not hasattr(env, "reinitialize_motor_dynamics"):
            raise RuntimeError(
                "real_b3 requires reinitialize_motor_dynamics()."
            )

        env.reinitialize_motor_dynamics(
            real_b3_env_ids
        )

        env.max_thrust.fill_(
            REAL_B3_K_ETA
            * float(env.cfg.motor_speed_max) ** 2
        )

        env.min_thrust.fill_(
            REAL_B3_K_ETA
            * float(env.cfg.motor_speed_min) ** 2
        )

        if hasattr(env, "vehicle_mass"):
            env.vehicle_mass = env._robot_mass

        if hasattr(env, "quad_inertia"):
            env.quad_inertia = env.inertia_tensor[0]

        physx_mass_check = (
            env._robot.root_physx_view
            .get_masses()[:, env._body_id]
        )

        physx_inertia_check = (
            env._robot.root_physx_view
            .get_inertias()[:, env._body_id, :]
            .view(env.num_envs, 3, 3)
        )

        print("\nREAL-B3 FIXED PHYSX PLANT")
        print("-" * 100)
        print(
            "PhysX body mass min/max :",
            float(physx_mass_check.min()),
            "/",
            float(physx_mass_check.max()),
        )
        print(
            "PhysX inertia env0      :",
            physx_inertia_check[0],
        )
        print(
            "k_eta                   :",
            float(env._k_eta[0]),
        )
        print(
            "k_torque                :",
            float(env._k_torque[0]),
        )
        print(
            "tau_m                   :",
            float(env._tau_m[0]),
        )
        print("-" * 100)

    # BENCHMARK_GC_BODY_GOAL_FIX_V1
    # ------------------------------------------------------------
    # In quadrotor_env.py the GC observation is assembled as:
    #
    #   current position -> get_frame_state_from_task("body")
    #   desired position -> get_goal_state_from_task("COM")
    #
    # For the manipulator asset, the latter applies the fixed
    # body/EE/COM transform (~9.06 cm). That is appropriate when
    # _desired_pos_w represents an end-effector target, but NOT for
    # this benchmark: we explicitly define _desired_pos_w as the
    # Crazyflie BODY target and also use body-frame reward/task goals.
    #
    # Therefore make the hard-coded GC "COM" request return the same
    # authoritative body target. Do not numerically compensate the
    # ~9 cm transform.
    if (
        env.cfg.task_body != "body"
        or env.cfg.goal_body != "body"
        or env.cfg.reward_task_body != "body"
        or env.cfg.reward_goal_body != "body"
    ):
        raise RuntimeError(
            "Mellinger benchmark requires body/body/body/body frame "
            "configuration before applying the GC goal-frame fix."
        )

    original_get_goal_state_from_task = (
        env.get_goal_state_from_task
    )

    def _benchmark_get_goal_state_from_task(
        goal_body,
    ):
        if goal_body == "COM":
            return (
                env._desired_pos_w,
                env._desired_ori_w,
            )

        return original_get_goal_state_from_task(
            goal_body
        )

    env.get_goal_state_from_task = (
        _benchmark_get_goal_state_from_task
    )

    actual_env_step_dt = float(env.step_dt)
    actual_env_step_rate_hz = (
        1.0 / actual_env_step_dt
    )

    if abs(
        actual_env_step_rate_hz
        - float(BENCHMARK_CONTROL_RATE_HZ)
    ) > 1.0e-6:
        raise RuntimeError(
            "Actual environment step rate mismatch: "
            f"expected={BENCHMARK_CONTROL_RATE_HZ} Hz, "
            f"actual={actual_env_step_rate_hz:.9f} Hz, "
            f"step_dt={actual_env_step_dt:.9f}"
        )

    if bool(
        getattr(
            env.cfg,
            "rotorpy_done",
            False,
        )
    ):
        raise RuntimeError(
            "Benchmark requires rotorpy_done=False."
        )

    print("\nBENCHMARK RATE CHECK")
    print("-" * 100)
    print(
        f"environment step dt  : "
        f"{actual_env_step_dt:.9f} s"
    )
    print(
        f"environment step rate: "
        f"{actual_env_step_rate_hz:.3f} Hz"
    )
    print("-" * 100)


    obs_dict, info = envs_gym.reset()

    # BENCHMARK_POST_RESET_ZERO_VELOCITY_V1
    # ------------------------------------------------------------
    # DirectRLEnv reset/initial observation can leave one environment
    # step of gravitational velocity before evaluator inspection.
    #
    # At 500 Hz:
    #     g * dt = 9.81 * 0.002 = 0.01962 m/s
    #
    # Re-impose the benchmark's exact stationary start AFTER reset,
    # then regenerate observations before sampling/commanding a goal.
    # ------------------------------------------------------------
    pre_zero_gc = obs_dict["gc"]

    if pre_zero_gc is None:
        raise RuntimeError(
            "GC observation missing immediately after reset."
        )

    pre_zero_speed = torch.linalg.norm(
        pre_zero_gc[:, 7:10],
        dim=1,
    )

    pre_zero_ang_rate = torch.linalg.norm(
        pre_zero_gc[:, 10:13],
        dim=1,
    )

    print("\nPOST-RESET ISAAC DRIFT BEFORE BENCHMARK ZEROING")
    print("-" * 100)
    print(
        "linear speed max   : "
        f"{pre_zero_speed.max().item():.9e} m/s"
    )
    print(
        "angular rate max   : "
        f"{pre_zero_ang_rate.max().item():.9e} deg/s"
    )
    print("-" * 100)

    benchmark_env_ids = torch.arange(
        env.num_envs,
        device=device,
        dtype=torch.long,
    )

    benchmark_zero_root_velocity = torch.zeros(
        (env.num_envs, 6),
        dtype=env._robot.data.root_state_w.dtype,
        device=device,
    )

    env._robot.write_root_velocity_to_sim(
        benchmark_zero_root_velocity,
        env_ids=benchmark_env_ids,
    )

    # BENCHMARK_INVALIDATE_BODY_VELOCITY_CACHE_V2
    #
    # write_root_velocity_to_sim() updates the root COM velocity and
    # PhysX state, but Isaac Lab's body velocity/state LazyBuffers may
    # still contain values read before that write.
    #
    # get_frame_state_from_task("body") uses body_lin_vel_w and
    # body_ang_vel_w, so explicitly invalidate those body-level caches
    # before regenerating GC observations.
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

    # Regenerate observations. Accessing body_lin_vel_w/body_ang_vel_w
    # now forces a fresh PhysX read of the zero velocity we just wrote.
    # _get_observations() does not advance physics.
    obs_dict = env._get_observations()

    post_zero_gc = obs_dict["gc"]

    if post_zero_gc is None:
        raise RuntimeError(
            "GC observation missing after benchmark velocity reset."
        )

    post_zero_speed = torch.linalg.norm(
        post_zero_gc[:, 7:10],
        dim=1,
    )

    post_zero_ang_rate = torch.linalg.norm(
        post_zero_gc[:, 10:13],
        dim=1,
    )

    print("\nBENCHMARK POST-RESET STATE CORRECTION")
    print("-" * 100)
    print(
        "linear speed max   : "
        f"{post_zero_speed.max().item():.9e} m/s"
    )
    print(
        "angular rate max   : "
        f"{post_zero_ang_rate.max().item():.9e} deg/s"
    )
    print("-" * 100)

    if post_zero_speed.max().item() > 1.0e-6:
        raise RuntimeError(
            "Failed to establish zero benchmark linear velocity: "
            f"{post_zero_speed.max().item():.9e} m/s"
        )

    if post_zero_ang_rate.max().item() > 1.0e-5:
        raise RuntimeError(
            "Failed to establish zero benchmark angular velocity: "
            f"{post_zero_ang_rate.max().item():.9e} deg/s"
        )

    if args_cli.zero_latency:
        if hasattr(env, "_control_latency_steps"):
            print(
                "Control latency before diagnostic override:",
                env._control_latency_steps,
            )

            # The environment rolls the queue left and writes the newest
            # command into queue[-1]. Therefore selecting the LAST queue
            # index means zero command delay.
            zero_delay_index = env._action_queue.shape[0] - 1
            env._control_latency_steps.fill_(zero_delay_index)

            print(
                "Control latency after diagnostic override :",
                env._control_latency_steps,
            )
            print(
                "Zero-delay queue index                   :",
                zero_delay_index,
            )
        else:
            print("[WARNING] env has no _control_latency_steps attribute")

    if obs_dict["gc"] is None:
        raise RuntimeError("GC observation was not produced.")

    # MELLINGER_CALIBRATION_V2
    # ------------------------------------------------------------
    # Actual randomized Isaac plant + controller calibration.
    # ------------------------------------------------------------
    num_envs = int(env.num_envs)

    def env_vector(name, default=float("nan")):
        value = getattr(env, name, None)

        if value is None:
            return torch.full(
                (num_envs,),
                float(default),
                dtype=torch.float32,
                device=device,
            )

        if not torch.is_tensor(value):
            value = torch.as_tensor(
                value,
                dtype=torch.float32,
                device=device,
            )
        else:
            value = value.detach().to(
                device=device,
                dtype=torch.float32,
            )

        value = value.reshape(-1)

        if value.numel() == 1 and num_envs > 1:
            value = value.repeat(num_envs)

        if value.numel() < num_envs:
            raise RuntimeError(
                f"{name} has only {value.numel()} values for "
                f"{num_envs} environments."
            )

        return value[:num_envs]

    actual_weight = env_vector("_robot_weight")
    actual_mass = actual_weight / 9.81
    actual_k_eta = env_vector("_k_eta")
    actual_tau_m = env_vector("_tau_m")
    actual_k_torque = env_vector("_k_torque")

    # Try to recover per-environment inertia when exposed by the env.
    actual_inertia_diag = torch.full(
        (num_envs, 3),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )

    for inertia_name in (
        "_robot_inertia",
        "_vehicle_inertia",
        "quad_inertia",
        "vehicle_inertia",
    ):
        inertia_value = getattr(env, inertia_name, None)

        if not torch.is_tensor(inertia_value):
            continue

        inertia_value = inertia_value.detach().to(
            device=device,
            dtype=torch.float32,
        )

        try:
            if inertia_value.ndim >= 3 and inertia_value.shape[-2:] == (3, 3):
                diag = torch.diagonal(
                    inertia_value,
                    dim1=-2,
                    dim2=-1,
                )
            elif inertia_value.ndim == 2 and inertia_value.shape[-1] == 3:
                diag = inertia_value
            else:
                continue

            if diag.shape[0] == 1 and num_envs > 1:
                diag = diag.repeat(num_envs, 1)

            if diag.shape[0] >= num_envs:
                actual_inertia_diag[:] = diag[:num_envs]
                break
        except Exception:
            pass

    # ------------------------------------------------------------
    # Effective action-queue delay.
    #
    # queue[-1] = newest command.
    # Therefore:
    #
    # actual_delay_steps = (queue_length - 1) - selected_index
    # ------------------------------------------------------------
    queue_length = int(env._action_queue.shape[0])

    raw_latency_index = env_vector(
        "_control_latency_steps",
        default=queue_length - 1,
    ).round().to(torch.long)

    selected_latency_index = torch.clamp(
        raw_latency_index,
        0,
        queue_length - 1,
    )

    actual_delay_steps = (
        (queue_length - 1) - selected_latency_index
    )

    delay_ms = (
        actual_delay_steps.to(torch.float32)
        * (1000.0 / float(env_cfg.policy_rate_hz))
    )

    # ------------------------------------------------------------
    # Controller calibration.
    # ------------------------------------------------------------
    nominal_mass = 0.04
    nominal_k_eta = 0.338
    nominal_weight = nominal_mass * 9.81

    nominal_hover_motor = math.sqrt(
        nominal_weight / (4.0 * nominal_k_eta)
    )

    nominal_mass_thrust = (
        65536.0
        * nominal_hover_motor
        / nominal_weight
    )

    controller = None
    controller_bank = None

    if args_cli.calibration_mode == "nominal":
        controller_mass_assumed = torch.full(
            (num_envs,),
            nominal_mass,
            dtype=torch.float32,
            device=device,
        )
        controller_k_eta_assumed = torch.full(
            (num_envs,),
            nominal_k_eta,
            dtype=torch.float32,
            device=device,
        )
        controller_mass_thrust = torch.full(
            (num_envs,),
            nominal_mass_thrust,
            dtype=torch.float32,
            device=device,
        )

        controller = MellingerFirmwareController(
            num_envs=num_envs,
            device=device,
            dt=1.0 / float(env_cfg.policy_rate_hz),
            mass=nominal_mass,
            mass_thrust=nominal_mass_thrust,
            idle_thrust=0.0,
            motor_command_scale=1.0,
        )
        controller.kd_omega_rp = args_cli.kd_omega_rp

    elif args_cli.calibration_mode == "real_b3":
        controller_mass_assumed = torch.full(
            (num_envs,),
            REAL_B3_MASS_KG,
            dtype=torch.float32,
            device=device,
        )

        controller_k_eta_assumed = torch.full(
            (num_envs,),
            REAL_B3_K_ETA,
            dtype=torch.float32,
            device=device,
        )

        controller_mass_thrust = torch.full(
            (num_envs,),
            REAL_B3_MASS_THRUST,
            dtype=torch.float32,
            device=device,
        )

        controller = MellingerFirmwareController(
            num_envs=num_envs,
            device=device,
            dt=1.0 / float(env_cfg.policy_rate_hz),
            mass=REAL_B3_MASS_KG,
            mass_thrust=REAL_B3_MASS_THRUST,
            idle_thrust=REAL_B3_IDLE_THRUST,
            motor_command_scale=1.0,
        )

        controller.kd_omega_rp = (
            args_cli.kd_omega_rp
        )

    else:
        controller_mass_assumed = actual_mass.clone()
        controller_k_eta_assumed = actual_k_eta.clone()

        same_plant_hover = torch.sqrt(
            actual_weight / (4.0 * actual_k_eta)
        )

        controller_mass_thrust = (
            65536.0
            * same_plant_hover
            / actual_weight
        )

        if num_envs == 1:
            controller = MellingerFirmwareController(
                num_envs=1,
                device=device,
                dt=1.0 / float(env_cfg.policy_rate_hz),
                mass=float(actual_mass[0].item()),
                mass_thrust=float(
                    controller_mass_thrust[0].item()
                ),
                idle_thrust=0.0,
                motor_command_scale=1.0,
            )
            controller.kd_omega_rp = args_cli.kd_omega_rp

        else:
            # True per-environment oracle calibration without changing
            # MellingerFirmwareController itself.
            controller_bank = []

            for env_index in range(num_envs):
                ctrl = MellingerFirmwareController(
                    num_envs=1,
                    device=device,
                    dt=1.0 / float(env_cfg.policy_rate_hz),
                    mass=float(
                        actual_mass[env_index].item()
                    ),
                    mass_thrust=float(
                        controller_mass_thrust[
                            env_index
                        ].item()
                    ),
                    idle_thrust=0.0,
                    motor_command_scale=1.0,
                )
                ctrl.kd_omega_rp = args_cli.kd_omega_rp
                controller_bank.append(ctrl)

    # MELLINGER_FIRST_DIVERGENCE_DEBUG_V1
    # Diagnostic only. Does not modify controller output.
    _mellinger_debug = {"active_calls": 0}

    def mellinger_action(gc_input):
        if controller_bank is None:
            action = controller.get_action(gc_input)

            # The pre-hover error is tiny. Start logging only after the
            # large benchmark position step has actually been applied.
            benchmark_error = torch.linalg.norm(
                gc_input[0, 13:16] - gc_input[0, 0:3]
            ).item()

            if benchmark_error > 0.5:
                _mellinger_debug["active_calls"] += 1
                k = _mellinger_debug["active_calls"]

                # 500 Hz controller => every 5 calls = 10 ms.
                # Log only the first 0.5 s of the maneuver.
                if k <= 1000 and (k <= 100 or k == 1 or k % 5 == 0):
                    fdes = controller.last_target_thrust[0]
                    fxy = torch.linalg.norm(fdes[0:2])

                    desired_force_tilt_deg = torch.rad2deg(
                        torch.atan2(fxy, fdes[2])
                    ).item()

                    pos = gc_input[0, 0:3]
                    vel = gc_input[0, 7:10]
                    omega_gc_deg = gc_input[0, 10:13]

                    eR = controller.last_eR[0]
                    ctrl = controller.last_control[0]
                    raw = controller.last_motor_raw[0]
                    capped = controller.last_motor_capped[0]
                    norm_fw = (
                        controller
                        .last_motor_normalized_firmware[0]
                    )
                    actual_motor = env._motor_speeds[0]

                    # _moment is the wrench currently held by the
                    # environment from the preceding physics application.
                    actual_moment = env._moment[0, 0, :]

                    # Passive desired-attitude geometry diagnostics.
                    heading_cross_norm = controller.last_heading_cross_norm[0]
                    cos_phi_raw = controller.last_cos_phi_raw[0]
                    phi_deg = controller.last_phi_deg[0]
                    x_c_des = controller.last_x_c_des[0]
                    actual_yaw_deg = controller.last_actual_yaw_deg[0]
                    desired_yaw_deg = controller.last_desired_yaw_deg[0]
                    tilt_error_deg = controller.last_tilt_error_deg[0]

                    t = (k - 1) / float(env_cfg.policy_rate_hz)

                    print(
                        "[MELDBG] "
                        f"t={t:.3f} "
                        f"pos={pos.detach().cpu().tolist()} "
                        f"vel={vel.detach().cpu().tolist()} "
                        f"omega_gc_deg={omega_gc_deg.detach().cpu().tolist()} "
                        f"Fdes={fdes.detach().cpu().tolist()} "
                        f"Fdes_tilt_deg={desired_force_tilt_deg:.3f} "
                        f"h={heading_cross_norm.item():.9e} "
                        f"cos_phi={cos_phi_raw.item():+.9f} "
                        f"phi_deg={phi_deg.item():.6f} "
                        f"x_c_des={x_c_des.detach().cpu().tolist()} "
                        f"actual_yaw_deg={actual_yaw_deg.item():+.6f} "
                        f"desired_yaw_deg={desired_yaw_deg.item():+.6f} "
                        f"tilt_error_deg={tilt_error_deg.item():.6f} "
                        f"eR={eR.detach().cpu().tolist()} "
                        f"control={ctrl.detach().cpu().tolist()} "
                        f"motor_raw={raw.detach().cpu().tolist()} "
                        f"motor_capped={capped.detach().cpu().tolist()} "
                        f"motor_fw={norm_fw.detach().cpu().tolist()} "
                        f"motor_actual={actual_motor.detach().cpu().tolist()} "
                        f"actual_moment={actual_moment.detach().cpu().tolist()}"
                    )

            return action

        return torch.cat(
            [
                ctrl.get_action(
                    gc_input[i : i + 1]
                )
                for i, ctrl in enumerate(controller_bank)
            ],
            dim=0,
        )

    def mellinger_motor_command():
        if controller_bank is None:
            return controller.last_motor_normalized

        return torch.cat(
            [
                ctrl.last_motor_normalized
                for ctrl in controller_bank
            ],
            dim=0,
        )

    print("\nACTUAL SIMULATED PLANT")
    print("-" * 100)
    print(
        f"mass kg      : min={actual_mass.min().item():.8f} "
        f"max={actual_mass.max().item():.8f}"
    )
    print(
        f"k_eta        : min={actual_k_eta.min().item():.8f} "
        f"max={actual_k_eta.max().item():.8f}"
    )
    print(
        f"k_torque     : min={actual_k_torque.min().item():.8f} "
        f"max={actual_k_torque.max().item():.8f}"
    )
    print(
        f"tau_m s      : min={actual_tau_m.min().item():.8f} "
        f"max={actual_tau_m.max().item():.8f}"
    )

    print("\nMELLINGER CALIBRATION")
    print("-" * 100)
    print("mode         :", args_cli.calibration_mode)

    if args_cli.calibration_mode == "same_plant":
        print(
            "interpretation : ORACLE / DIAGNOSTIC "
            "(controller knows randomized mass and k_eta)"
        )

    elif args_cli.calibration_mode == "nominal":
        print(
            "interpretation : FROZEN NOMINAL "
            "(fair robustness baseline)"
        )
        print(f"assumed mass : {nominal_mass:.8f} kg")
        print(f"assumed k_eta: {nominal_k_eta:.8f}")
        print(
            f"massThrust   : {nominal_mass_thrust:.8f}"
        )

    else:
        print(
            "interpretation : FIXED REAL-B3 PLANT + "
            "DEPLOYED FIRMWARE MELLINGER"
        )
        print(
            f"assumed mass : {REAL_B3_MASS_KG:.8f} kg"
        )
        print(
            f"plant k_eta  : {REAL_B3_K_ETA:.8f}"
        )
        print(
            f"massThrust   : {REAL_B3_MASS_THRUST:.8f}"
        )
        print(
            f"idleThrust   : {REAL_B3_IDLE_THRUST:.8f}"
        )

    print(
        "Mellinger kd_omega_rp :",
        args_cli.kd_omega_rp,
    )

    print("\nCONTROL LATENCY")
    print("-" * 100)
    print("queue length              :", queue_length)
    print(
        "selected queue index min/max:",
        int(selected_latency_index.min().item()),
        "/",
        int(selected_latency_index.max().item()),
    )
    print(
        "physical delay ms min/max   :",
        f"{delay_ms.min().item():.3f}",
        "/",
        f"{delay_ms.max().item():.3f}",
    )
    print("=" * 100)

    if args_cli.sign_probe and num_envs != 1:
        raise ValueError(
            "--sign_probe requires --num_envs 1."
        )

    gc = obs_dict["gc"]

    # ============================================================
    # STATIC ATTITUDE / RATE SIGN PROBE
    # ============================================================
    if args_cli.sign_probe:
        print("\n" + "=" * 100)
        print("STATIC MELLINGER -> ISAAC TORQUE SIGN PROBE")
        print("=" * 100)
        print(
            "For every positive perturbation, the corresponding physical "
            "torque should be NEGATIVE (restoring/damping)."
        )

        def quat_axis(axis, angle_deg):
            angle = math.radians(angle_deg)
            q = torch.zeros((1, 4), device=device)
            q[:, 0] = math.cos(angle / 2.0)

            if axis == "roll":
                q[:, 1] = math.sin(angle / 2.0)
            elif axis == "pitch":
                q[:, 2] = math.sin(angle / 2.0)
            elif axis == "yaw":
                q[:, 3] = math.sin(angle / 2.0)

            return q

        def evaluate_case(
            name,
            quat=None,
            omega_deg=None,
            initialize_rate=False,
            goal_offset=None,
            linear_vel=None,
        ):
            controller.reset()

            probe = torch.zeros(
                (1, gc.shape[1]),
                dtype=gc.dtype,
                device=device,
            )

            # Level vehicle at the origin with desired yaw = 0.
            probe[:, 3] = 1.0

            if quat is not None:
                probe[:, 3:7] = quat

            # Default: position == goal.
            probe[:, 13:16] = probe[:, 0:3]
            probe[:, 16] = 0.0

            if goal_offset is not None:
                probe[:, 13:16] += torch.tensor(
                    [goal_offset],
                    dtype=probe.dtype,
                    device=device,
                )

            if linear_vel is not None:
                probe[:, 7:10] = torch.tensor(
                    [linear_vel],
                    dtype=probe.dtype,
                    device=device,
                )

            if initialize_rate:
                # First controller call initializes the firmware-like
                # previous angular-rate state at zero.
                controller.get_action(probe.clone())

            if omega_deg is not None:
                probe[:, 10:13] = torch.tensor(
                    [omega_deg],
                    dtype=probe.dtype,
                    device=device,
                )

            controller.get_action(probe)

            # last_motor_normalized is already in ISAAC rotor order.
            u = controller.last_motor_normalized[0]

            # Evaluate the exact physical wrench that the Isaac plant
            # would generate from the commanded motor coordinates.
            motor_forces = env._k_eta[0] * u**2
            wrench = env.f_to_TM[0] @ motor_forces

            eR = controller.last_eR[0]
            legacy = controller.last_control[0]

            print("\n" + name)
            print("-" * 100)
            print(
                "eR                  =",
                [round(x, 7) for x in eR.detach().cpu().tolist()],
            )
            print(
                "legacy [T,R,P,Y]    =",
                [round(x, 3) for x in legacy.detach().cpu().tolist()],
            )
            print(
                "Isaac motors        =",
                [round(x, 6) for x in u.detach().cpu().tolist()],
            )
            print(
                "physical [F,Tx,Ty,Tz]=",
                [round(x, 9) for x in wrench.detach().cpu().tolist()],
            )

        # --------------------------------------------------------
        # Attitude proportional-feedback tests
        # --------------------------------------------------------
        evaluate_case(
            "+1 deg ROLL attitude error",
            quat=quat_axis("roll", +1.0),
        )

        evaluate_case(
            "-1 deg ROLL attitude error",
            quat=quat_axis("roll", -1.0),
        )

        evaluate_case(
            "+1 deg PITCH attitude error",
            quat=quat_axis("pitch", +1.0),
        )

        evaluate_case(
            "-1 deg PITCH attitude error",
            quat=quat_axis("pitch", -1.0),
        )

        evaluate_case(
            "+1 deg YAW attitude error",
            quat=quat_axis("yaw", +1.0),
        )

        evaluate_case(
            "-1 deg YAW attitude error",
            quat=quat_axis("yaw", -1.0),
        )

        # --------------------------------------------------------
        # Angular-rate damping tests
        #
        # gc_obs stores world angular velocity in deg/s.
        # At identity attitude, world == body.
        # --------------------------------------------------------
        evaluate_case(
            "+1 deg/s ROLL rate",
            omega_deg=[+1.0, 0.0, 0.0],
        )

        evaluate_case(
            "+1 deg/s PITCH rate",
            omega_deg=[0.0, +1.0, 0.0],
        )

        evaluate_case(
            "+1 deg/s YAW rate",
            omega_deg=[0.0, 0.0, +1.0],
        )

        # Include the firmware angular-acceleration D term.
        evaluate_case(
            "+1 deg/s ROLL rate STEP (includes kd_omega_rp)",
            omega_deg=[+1.0, 0.0, 0.0],
            initialize_rate=True,
        )

        evaluate_case(
            "+1 deg/s PITCH rate STEP (includes kd_omega_rp)",
            omega_deg=[0.0, +1.0, 0.0],
            initialize_rate=True,
        )

        # --------------------------------------------------------
        # Outer position-loop direction tests
        #
        # At level attitude:
        #
        # +x desired force requires +pitch  -> +Ty
        # +y desired force requires -roll   -> -Tx
        #
        # Position perturbation is expressed as GOAL - POSITION.
        # --------------------------------------------------------
        print("\n" + "=" * 100)
        print("OUTER POSITION / VELOCITY LOOP SIGN PROBE")
        print("=" * 100)

        evaluate_case(
            "+0.05 m X position error",
            goal_offset=[+0.05, 0.0, 0.0],
        )

        evaluate_case(
            "-0.05 m X position error",
            goal_offset=[-0.05, 0.0, 0.0],
        )

        evaluate_case(
            "+0.05 m Y position error",
            goal_offset=[0.0, +0.05, 0.0],
        )

        evaluate_case(
            "-0.05 m Y position error",
            goal_offset=[0.0, -0.05, 0.0],
        )

        evaluate_case(
            "+0.05 m Z position error",
            goal_offset=[0.0, 0.0, +0.05],
        )

        evaluate_case(
            "-0.05 m Z position error",
            goal_offset=[0.0, 0.0, -0.05],
        )

        # --------------------------------------------------------
        # Translational velocity damping.
        #
        # Positive velocity with zero position error should command
        # acceleration / tilt in the opposite direction.
        # --------------------------------------------------------
        evaluate_case(
            "+0.10 m/s X velocity",
            linear_vel=[+0.10, 0.0, 0.0],
        )

        evaluate_case(
            "+0.10 m/s Y velocity",
            linear_vel=[0.0, +0.10, 0.0],
        )

        evaluate_case(
            "+0.10 m/s Z velocity",
            linear_vel=[0.0, 0.0, +0.10],
        )

        print("\n" + "=" * 100)
        print("EXPECTED OUTER-LOOP SIGNS")
        print("=" * 100)
        print("+X position error -> Ty > 0")
        print("-X position error -> Ty < 0")
        print("+Y position error -> Tx < 0")
        print("-Y position error -> Tx > 0")
        print("+Z position error -> F  > plant weight")
        print("-Z position error -> F  < plant weight")
        print("+X velocity       -> Ty < 0")
        print("+Y velocity       -> Tx > 0")
        print("+Z velocity       -> F  < plant weight")
        print(
            f"plant weight      = "
            f"{actual_weight[0].item():.9f} N"
        )
        print("=" * 100)

        print("\n" + "=" * 100)
        print("EXPECTED SIGN TEST")
        print("=" * 100)
        print("+roll angle/rate  -> Tx < 0")
        print("+pitch angle/rate -> Ty < 0")
        print("+yaw angle/rate   -> Tz < 0")
        print("-roll angle       -> Tx > 0")
        print("-pitch angle      -> Ty > 0")
        print("-yaw angle        -> Tz > 0")
        print("=" * 100)

        envs_gym.close()
        return

    # VECTORIZED_FIRST_EPISODE_EVALUATION
    # ============================================================
    # First-episode-only vectorized evaluation.
    # ============================================================
    gc = obs_dict["gc"]

    # ================================================================
    # RL_TRANSFER_CASES_REAL_B3_V2
    # ================================================================
    #
    # Reproduce ONLY the original RL translation task:
    #
    #     delta_i = RL_EE_goal_i - RL_EE_start_i
    #
    # We deliberately do NOT reproduce the old absolute world position,
    # because some of those stored RL states are below the ground plane.
    #
    # Every environment remains:
    #   - fixed realistic B3
    #   - safe z=3 m initialization
    #   - level attitude
    #   - zero initial velocity/rate
    #   - zero artificial latency
    #
    # Existing pre-hover code establishes the running Mellinger state
    # before the displacement is applied.
    # ================================================================
    imported_rl_ee_start = None
    imported_rl_ee_goal = None
    imported_rl_ee_delta = None

    if args_cli.rl_position_cases is not None:
        if args_cli.calibration_mode != "real_b3":
            raise ValueError(
                "--rl_position_cases requires "
                "--calibration_mode real_b3."
            )

        if args_cli.pre_hover_s <= 0.0:
            raise ValueError(
                "--rl_position_cases requires positive pre-hover. "
                "Use --pre_hover_s 2.0."
            )

        if args_cli.goal_offset is not None:
            raise ValueError(
                "--rl_position_cases cannot be combined with "
                "--goal_offset."
            )

        if args_cli.goal_offset_range is not None:
            raise ValueError(
                "--rl_position_cases cannot be combined with "
                "--goal_offset_range."
            )

        if args_cli.hold_initial_pose:
            raise ValueError(
                "--rl_position_cases cannot be combined with "
                "--hold_initial_pose."
            )

        if "0DOF" not in str(args_cli.task):
            raise ValueError(
                "RL transfer import currently supports only the "
                "0DOF full-state layout."
            )

        case_path = Path(
            args_cli.rl_position_cases
        ).expanduser().resolve()

        if not case_path.is_file():
            raise FileNotFoundError(
                f"RL evaluation tensor not found: {case_path}"
            )

        rl_cases = torch.load(
            case_path,
            map_location="cpu",
            weights_only=False,
        )

        if not torch.is_tensor(rl_cases):
            raise TypeError(
                "Expected eval_full_states.pt to contain a tensor."
            )

        if rl_cases.ndim != 3:
            raise ValueError(
                "Expected RL tensor [N,T,D], got "
                f"{tuple(rl_cases.shape)}."
            )

        if int(rl_cases.shape[0]) != int(num_envs):
            raise ValueError(
                "RL case count does not match --num_envs: "
                f"file={rl_cases.shape[0]}, "
                f"envs={num_envs}."
            )

        if int(rl_cases.shape[1]) < 1:
            raise ValueError(
                "RL tensor contains no time samples."
            )

        if int(rl_cases.shape[2]) < 29:
            raise ValueError(
                "RL tensor is too short for the 0DOF full-state layout."
            )

        # 0DOF full-state:
        #   13:16 = initial EE XYZ
        #   26:29 = desired EE XYZ
        imported_rl_ee_start = (
            rl_cases[:, 0, 13:16]
            .to(
                device=device,
                dtype=gc.dtype,
            )
            .clone()
        )

        imported_rl_ee_goal = (
            rl_cases[:, 0, 26:29]
            .to(
                device=device,
                dtype=gc.dtype,
            )
            .clone()
        )

        imported_rl_ee_delta = (
            imported_rl_ee_goal
            - imported_rl_ee_start
        )

        if not torch.isfinite(
            imported_rl_ee_delta
        ).all():
            raise RuntimeError(
                "Imported RL EE displacement contains non-finite values."
            )

        transfer_norm = torch.linalg.norm(
            imported_rl_ee_delta,
            dim=1,
        )

        transfer_median = float(
            torch.quantile(
                transfer_norm,
                0.5,
            ).item()
        )

        print()
        print("=" * 100)
        print(
            "RL EE TRANSFERS -> FIXED REAL-B3 MELLINGER"
        )
        print("=" * 100)
        print("source file:", case_path)
        print("cases      :", num_envs)
        print()
        print(
            "COPIED FROM RL : EE displacement only "
            "(goal - initial EE)"
        )
        print(
            "NOT COPIED     : absolute position, plant, attitude, "
            "velocity, motors, latency"
        )
        print()
        print(
            "transfer norm min   :",
            f"{transfer_norm.min().item():.6f} m",
        )
        print(
            "transfer norm median:",
            f"{transfer_median:.6f} m",
        )
        print(
            "transfer norm mean  :",
            f"{transfer_norm.mean().item():.6f} m",
        )
        print(
            "transfer norm max   :",
            f"{transfer_norm.max().item():.6f} m",
        )

        print()
        print(
            "env 0 delta :",
            imported_rl_ee_delta[0]
            .detach()
            .cpu()
            .numpy(),
        )
        print(
            "env 99 delta:",
            imported_rl_ee_delta[-1]
            .detach()
            .cpu()
            .numpy(),
        )
        print("=" * 100)

    initial_pos = gc[:, 0:3].clone()
    initial_env_goal = gc[:, 13:16].clone()

    q0 = gc[:, 3:7]
    w0, x0, y0, z0 = q0.unbind(dim=1)

    initial_yaw = torch.atan2(
        2.0 * (w0 * z0 + x0 * y0),
        1.0 - 2.0 * (y0 * y0 + z0 * z0),
    )


    # ------------------------------------------------------------
    # Benchmark initial-state validation.
    # ------------------------------------------------------------
    initial_lin_vel = gc[:, 7:10].clone()
    initial_ang_vel_degps = gc[:, 10:13].clone()

    initial_speed = torch.linalg.norm(
        initial_lin_vel,
        dim=1,
    )

    initial_ang_rate_degps = torch.linalg.norm(
        initial_ang_vel_degps,
        dim=1,
    )

    initial_quat = gc[:, 3:7]

    quat_xyz_max = float(
        torch.abs(
            initial_quat[:, 1:4]
        ).max().item()
    )

    quat_w_error_max = float(
        torch.abs(
            torch.abs(initial_quat[:, 0])
            - 1.0
        ).max().item()
    )

    initial_z_min = float(
        initial_pos[:, 2].min().item()
    )

    initial_z_max = float(
        initial_pos[:, 2].max().item()
    )

    print("\nBENCHMARK INITIAL-STATE CHECK")
    print("-" * 100)
    print(
        f"body z min/max         : "
        f"{initial_z_min:.6f} / "
        f"{initial_z_max:.6f} m"
    )
    print(
        f"initial speed max      : "
        f"{initial_speed.max().item():.9e} m/s"
    )
    print(
        f"initial ang-rate max   : "
        f"{initial_ang_rate_degps.max().item():.9e} deg/s"
    )
    print(
        f"quat xyz abs max       : "
        f"{quat_xyz_max:.9e}"
    )
    print(
        f"| |qw| - 1 | max      : "
        f"{quat_w_error_max:.9e}"
    )
    print("-" * 100)

    if (
        initial_z_min < 2.5
        or initial_z_max > 3.5
    ):
        raise RuntimeError(
            "Benchmark start is not near the intended 3 m hover: "
            f"{initial_z_min:.6f}..{initial_z_max:.6f} m"
        )

    if initial_speed.max().item() > 1.0e-4:
        raise RuntimeError(
            "Benchmark initial velocity is not zero: "
            f"{initial_speed.max().item():.9e} m/s"
        )

    if initial_ang_rate_degps.max().item() > 1.0e-3:
        raise RuntimeError(
            "Benchmark initial angular velocity is not zero: "
            f"{initial_ang_rate_degps.max().item():.9e} deg/s"
        )

    if (
        quat_xyz_max > 1.0e-5
        or quat_w_error_max > 1.0e-5
    ):
        raise RuntimeError(
            "Benchmark initial attitude is not level/identity."
        )

    fixed_commanded_goal = None
    sampled_goal_offset = None

    if args_cli.goal_offset_range is not None:
        if args_cli.goal_offset is not None:
            raise ValueError(
                "--goal_offset and --goal_offset_range are mutually exclusive."
            )

        if args_cli.hold_initial_pose:
            raise ValueError(
                "--hold_initial_pose and --goal_offset_range are mutually exclusive."
            )


    if args_cli.rl_position_cases is not None:
        if imported_rl_ee_delta is None:
            raise RuntimeError(
                "Imported RL EE transfer vectors were not constructed."
            )

        fixed_commanded_goal = (
            initial_pos
            + imported_rl_ee_delta
        )

        reconstruction_error = float(
            torch.max(
                torch.abs(
                    fixed_commanded_goal
                    - initial_pos
                    - imported_rl_ee_delta
                )
            ).item()
        )

        print()
        print("=" * 100)
        print("RL TRANSFER TARGET CHECK")
        print("=" * 100)
        print(
            "env 0 benchmark start:",
            initial_pos[0]
            .detach()
            .cpu()
            .numpy(),
        )
        print(
            "env 0 RL transfer    :",
            imported_rl_ee_delta[0]
            .detach()
            .cpu()
            .numpy(),
        )
        print(
            "env 0 final target   :",
            fixed_commanded_goal[0]
            .detach()
            .cpu()
            .numpy(),
        )
        print(
            "max reconstruction error:",
            f"{reconstruction_error:.9e} m",
        )
        print("=" * 100)

    elif args_cli.hold_initial_pose:
        fixed_commanded_goal = initial_pos.clone()

    elif args_cli.goal_offset is not None:
        goal_offset_tensor = torch.tensor(
            args_cli.goal_offset,
            dtype=gc.dtype,
            device=device,
        ).view(1, 3)

        fixed_commanded_goal = (
            initial_pos + goal_offset_tensor
        )

    elif args_cli.goal_offset_range is not None:
        goal_offset_range_np = np.asarray(
            args_cli.goal_offset_range,
            dtype=np.float32,
        )

        if np.any(goal_offset_range_np < 0.0):
            raise ValueError(
                "--goal_offset_range values must all be non-negative."
            )

        # Dedicated benchmark RNG:
        # goal samples depend only on --seed and num_envs,
        # not on Isaac/Hydra/PyTorch RNG consumption.
        goal_rng = np.random.default_rng(
            int(args_cli.seed)
        )

        sampled_goal_offset_np = goal_rng.uniform(
            low=-goal_offset_range_np,
            high=goal_offset_range_np,
            size=(num_envs, 3),
        ).astype(np.float32)

        sampled_goal_offset = torch.as_tensor(
            sampled_goal_offset_np,
            dtype=gc.dtype,
            device=device,
        )

        fixed_commanded_goal = (
            initial_pos + sampled_goal_offset
        )


    # ============================================================
    # REAL_FLIGHT_PRE_HOVER_V1
    # ============================================================
    #
    # Real flight sequence:
    #
    #   Mellinger holds the old position setpoint
    #       -> controller/integrators/rate history evolve
    #       -> motor/vehicle state evolves
    #       -> position setpoint changes
    #
    # This phase deliberately occurs BEFORE the final benchmark goal
    # override below. Nothing is reset at the switch.
    # ============================================================
    if args_cli.pre_hover_s < 0.0:
        raise ValueError(
            "--pre_hover_s must be >= 0."
        )

    pre_hover_steps_float = (
        float(args_cli.pre_hover_s)
        * float(env_cfg.policy_rate_hz)
    )

    pre_hover_steps = int(
        round(pre_hover_steps_float)
    )

    pre_hover_actual_s = (
        pre_hover_steps
        / float(env_cfg.policy_rate_hz)
    )

    if pre_hover_steps > 0:
        if fixed_commanded_goal is None:
            raise ValueError(
                "--pre_hover_s requires --goal_offset or "
                "--goal_offset_range."
            )

        if args_cli.hold_initial_pose:
            raise ValueError(
                "--pre_hover_s is not meaningful together with "
                "--hold_initial_pose."
            )

        print()
        print("=" * 100)
        print("MELLINGER PRE-STEP HOVER")
        print("=" * 100)
        print(
            f"requested duration : "
            f"{float(args_cli.pre_hover_s):.6f} s"
        )
        print(
            f"actual duration    : "
            f"{pre_hover_actual_s:.6f} s"
        )
        print(
            f"controller steps   : "
            f"{pre_hover_steps}"
        )
        print(
            "initial setpoint    :",
            initial_pos[0]
            .detach()
            .cpu()
            .numpy(),
        )
        print(
            "final step target   :",
            fixed_commanded_goal[0]
            .detach()
            .cpu()
            .numpy(),
        )
        print("-" * 100)

        with torch.no_grad():
            for pre_step in range(
                pre_hover_steps
            ):
                gc_pre = obs_dict["gc"]

                if gc_pre is None:
                    raise RuntimeError(
                        "GC observation disappeared during pre-hover."
                    )

                # Before the step command is applied, the environment's
                # own static target must still equal the initial hover
                # setpoint.
                pre_goal_error = float(
                    torch.max(
                        torch.abs(
                            gc_pre[:, 13:16]
                            - initial_pos
                        )
                    ).item()
                )

                if pre_goal_error > 1.0e-5:
                    raise RuntimeError(
                        "Initial hover target drifted before the "
                        "benchmark step: "
                        f"step={pre_step}, "
                        f"max_error={pre_goal_error:.9e}"
                    )

                gc_controller_pre = (
                    gc_pre.clone()
                )

                # Explicitly give Mellinger the old position/yaw
                # command throughout pre-hover.
                gc_controller_pre[:, 13:16] = (
                    initial_pos
                )
                gc_controller_pre[:, 16] = (
                    initial_yaw
                )

                pre_actions = mellinger_action(
                    gc_controller_pre
                )

                (
                    obs_dict,
                    _pre_reward,
                    pre_terminated,
                    pre_truncated,
                    _pre_info,
                ) = envs_gym.step(
                    pre_actions
                )

                pre_done = (
                    pre_terminated
                    | pre_truncated
                )

                if bool(
                    pre_done.any().item()
                ):
                    done_ids = (
                        torch.nonzero(
                            pre_done,
                            as_tuple=False,
                        )
                        .squeeze(-1)
                        .detach()
                        .cpu()
                        .tolist()
                    )

                    raise RuntimeError(
                        "Environment ended during pre-hover at "
                        f"step={pre_step + 1}/"
                        f"{pre_hover_steps}; "
                        f"env_ids={done_ids}"
                    )

        # State immediately BEFORE the real-flight-like goal switch.
        pre_step_gc = obs_dict["gc"]

        pre_step_pos = (
            pre_step_gc[:, 0:3]
            .clone()
        )

        pre_step_vel = (
            pre_step_gc[:, 7:10]
            .clone()
        )

        pre_step_quat = (
            pre_step_gc[:, 3:7]
            .clone()
        )

        pre_step_speed = torch.linalg.norm(
            pre_step_vel,
            dim=1,
        )

        pre_qx = pre_step_quat[:, 1]
        pre_qy = pre_step_quat[:, 2]

        pre_body_z_dot_world_z = (
            1.0
            - 2.0
            * (
                pre_qx * pre_qx
                + pre_qy * pre_qy
            )
        )

        pre_step_tilt_deg = torch.rad2deg(
            torch.acos(
                torch.clamp(
                    pre_body_z_dot_world_z,
                    -1.0,
                    1.0,
                )
            )
        )

        pre_motor_command = (
            mellinger_motor_command()
        )

        actual_pre_motor = getattr(
            env,
            "_motor_speeds",
            None,
        )

        print()
        print("PRE-STEP STATE — ENV 0")
        print("-" * 100)
        print(
            "physical position :",
            pre_step_pos[0]
            .detach()
            .cpu()
            .numpy(),
        )
        print(
            f"speed             : "
            f"{pre_step_speed[0].item():.9f} m/s"
        )
        print(
            f"tilt              : "
            f"{pre_step_tilt_deg[0].item():.6f} deg"
        )
        print(
            "commanded motors  :",
            pre_motor_command[0]
            .detach()
            .cpu()
            .numpy(),
        )

        if torch.is_tensor(
            actual_pre_motor
        ):
            print(
                "actual motor state:",
                actual_pre_motor[0, :4]
                .detach()
                .cpu()
                .numpy(),
            )

        if (
            controller_bank is None
            and hasattr(
                controller,
                "i_pos",
            )
        ):
            print(
                "controller i_pos  :",
                controller.i_pos[0]
                .detach()
                .cpu()
                .numpy(),
            )

        print()
        print(
            "[Evaluator] Applying benchmark position step now. "
            "Controller/vehicle/motors are NOT reset."
        )
        print("=" * 100)

    # ============================================================
    # BENCHMARK GOAL SYNCHRONIZATION
    # ============================================================
    #
    # The sampled benchmark target becomes authoritative through:
    #
    #   _desired_pos_w
    #   _desired_pos_traj_w
    #
    # The environment then generates GC/reward/visualization from
    # those buffers. Mellinger consumes the resulting GC target.
    benchmark_fixed_goal = None

    if fixed_commanded_goal is not None:
        benchmark_fixed_goal = (
            fixed_commanded_goal
            .detach()
            .clone()
        )

        original_update_goal_state = (
            env.update_goal_state
        )

        def _goal_expanded_for(buffer):
            if buffer.shape[0] != num_envs:
                raise RuntimeError(
                    "Unexpected desired-position first dimension: "
                    f"{tuple(buffer.shape)}"
                )

            if buffer.shape[-1] != 3:
                raise RuntimeError(
                    "Unexpected desired-position buffer shape: "
                    f"{tuple(buffer.shape)}"
                )

            view_shape = (
                num_envs,
                *([1] * (buffer.ndim - 2)),
                3,
            )

            return (
                benchmark_fixed_goal
                .view(view_shape)
                .expand_as(buffer)
            )

        def _apply_benchmark_goal():
            env._desired_pos_w.copy_(
                _goal_expanded_for(
                    env._desired_pos_w
                )
            )

            if hasattr(
                env,
                "_desired_pos_traj_w",
            ):
                env._desired_pos_traj_w.copy_(
                    _goal_expanded_for(
                        env._desired_pos_traj_w
                    )
                )

        def _benchmark_update_goal_state(
            *args,
            **kwargs,
        ):
            result = original_update_goal_state(
                *args,
                **kwargs,
            )

            # Let normal environment bookkeeping happen first.
            # Benchmark position is authoritative afterward.
            _apply_benchmark_goal()

            return result

        env.update_goal_state = (
            _benchmark_update_goal_state
        )

        # Establish benchmark target before controller action #1.
        env.update_goal_state()
        obs_dict = env._get_observations()
        gc = obs_dict["gc"]

        initial_env_goal = (
            gc[:, 13:16].clone()
        )

        def _benchmark_goal_sync_errors(
            gc_tensor,
        ):
            errors = {}

            errors["gc"] = float(
                torch.max(
                    torch.abs(
                        gc_tensor[:, 13:16]
                        - benchmark_fixed_goal
                    )
                ).item()
            )

            errors["_desired_pos_w"] = float(
                torch.max(
                    torch.abs(
                        env._desired_pos_w
                        - _goal_expanded_for(
                            env._desired_pos_w
                        )
                    )
                ).item()
            )

            if hasattr(
                env,
                "_desired_pos_traj_w",
            ):
                errors[
                    "_desired_pos_traj_w"
                ] = float(
                    torch.max(
                        torch.abs(
                            env._desired_pos_traj_w
                            - _goal_expanded_for(
                                env._desired_pos_traj_w
                            )
                        )
                    ).item()
                )

            return errors

        sync_errors = (
            _benchmark_goal_sync_errors(gc)
        )

        max_sync_error = max(
            sync_errors.values()
        )

        if max_sync_error > 1.0e-6:
            raise RuntimeError(
                "Benchmark target synchronization failed: "
                f"{sync_errors}"
            )

        # Static-target derivative sanity check.
        desired_velocity_max = float(
            torch.abs(
                env._pos_traj[1]
            ).max().item()
        )

        desired_acceleration_max = float(
            torch.abs(
                env._pos_traj[2]
            ).max().item()
        )

        desired_yaw_rate_max = float(
            torch.abs(
                env._yaw_traj[1]
            ).max().item()
        )

        if (
            desired_velocity_max > 1.0e-7
            or desired_acceleration_max > 1.0e-7
            or desired_yaw_rate_max > 1.0e-7
        ):
            raise RuntimeError(
                "Benchmark reference is not static: "
                f"vel={desired_velocity_max:.9e}, "
                f"acc={desired_acceleration_max:.9e}, "
                f"yaw_rate={desired_yaw_rate_max:.9e}"
            )

        print(
            "\n[Evaluator] Benchmark goal synchronized."
        )

        for name, error in sync_errors.items():
            print(
                f"  {name:24s}: "
                f"{error:.9e} m"
            )

        print(
            f"  desired velocity max    : "
            f"{desired_velocity_max:.9e}"
        )
        print(
            f"  desired acceleration max: "
            f"{desired_acceleration_max:.9e}"
        )
        print(
            f"  desired yaw-rate max    : "
            f"{desired_yaw_rate_max:.9e}"
        )

    # Mellinger sees exactly the goal exposed by environment GC.
    initial_controller_goal = (
        initial_env_goal.clone()
    )

    commanded_goal_offset = (
        initial_controller_goal - initial_pos
    )

    initial_dist = torch.linalg.norm(
        initial_controller_goal - initial_pos,
        dim=1,
    )

    print("\nINITIAL CONDITION — ENV 0")
    print("-" * 100)
    print(
        "position         :",
        initial_pos[0].detach().cpu().numpy(),
    )
    print(
        "environment goal :",
        initial_env_goal[0].detach().cpu().numpy(),
    )
    print(
        "controller goal  :",
        initial_controller_goal[0]
        .detach()
        .cpu()
        .numpy(),
    )
    print(
        f"distance         : "
        f"{initial_dist[0].item():.6f} m"
    )
    print("=" * 100)

    def quat_wxyz_to_rpy(q):
        w, x, y, z = q.unbind(dim=-1)

        roll = torch.atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )

        sin_pitch = 2.0 * (w * y - z * x)
        pitch = torch.asin(
            torch.clamp(
                sin_pitch,
                -1.0,
                1.0,
            )
        )

        yaw = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

        return torch.stack(
            (roll, pitch, yaw),
            dim=-1,
        )

    max_steps = int(args_cli.steps)
    n = num_envs

    def nan_log(*shape):
        return torch.full(
            shape,
            float("nan"),
            dtype=torch.float32,
            device=device,
        )

    position_log = nan_log(n, max_steps, 3)
    goal_log = nan_log(n, max_steps, 3)
    error_xyz_log = nan_log(n, max_steps, 3)
    error_norm_log = nan_log(n, max_steps)

    quat_log = nan_log(n, max_steps, 4)
    rpy_log = nan_log(n, max_steps, 3)
    tilt_deg_log = nan_log(n, max_steps)

    linear_velocity_log = nan_log(
        n,
        max_steps,
        3,
    )
    angular_velocity_log = nan_log(
        n,
        max_steps,
        3,
    )
    speed_log = nan_log(n, max_steps)

    actions_log = nan_log(n, max_steps, 4)
    motor_command_log = nan_log(
        n,
        max_steps,
        4,
    )
    actual_motor_log = nan_log(
        n,
        max_steps,
        4,
    )

    reward_log = nan_log(n, max_steps)

    valid_log = torch.zeros(
        (n, max_steps),
        dtype=torch.bool,
        device=device,
    )

    terminated_log = torch.zeros(
        (n, max_steps),
        dtype=torch.bool,
        device=device,
    )

    truncated_log = torch.zeros(
        (n, max_steps),
        dtype=torch.bool,
        device=device,
    )

    finished = torch.zeros(
        n,
        dtype=torch.bool,
        device=device,
    )

    first_terminated = torch.zeros(
        n,
        dtype=torch.bool,
        device=device,
    )

    first_truncated = torch.zeros(
        n,
        dtype=torch.bool,
        device=device,
    )

    first_end_step = torch.full(
        (n,),
        -1,
        dtype=torch.long,
        device=device,
    )

    steps_run = 0

    with torch.no_grad():
        for step in range(max_steps):
            active = ~finished

            if not bool(active.any().item()):
                break

            gc = obs_dict["gc"]
            gc_controller = gc.clone()

            if (
                args_cli.validate_goal_sync
                and benchmark_fixed_goal is not None
            ):
                sync_errors = (
                    _benchmark_goal_sync_errors(gc)
                )

                max_sync_error = max(
                    sync_errors.values()
                )

                if max_sync_error > 1.0e-6:
                    raise RuntimeError(
                        "Benchmark goal drift detected at "
                        f"step={step}: {sync_errors}"
                    )


                # Position target comes exclusively from synchronized GC.
                controller_goal = gc[:, 13:16]

                # Heading-reference causality experiment.
                # Keep pre-hover at initial_yaw; change only the
                # desired heading supplied to Mellinger after the step.
                gc_controller[:, 16] = math.radians(
                    args_cli.desired_yaw_deg
                )

                actions = mellinger_action(
                    gc_controller
                )

                motor_command = (
                    mellinger_motor_command()
                )

                pos = gc[:, 0:3]
                quat = gc[:, 3:7]
                vel = gc[:, 7:10]
                ang_vel = gc[:, 10:13]

                error_xyz = controller_goal - pos

                error_norm = torch.linalg.norm(
                    error_xyz,
                    dim=1,
                )

                speed = torch.linalg.norm(
                    vel,
                    dim=1,
                )

                rpy = quat_wxyz_to_rpy(quat)

                # Exact body-Z tilt relative to world Z.
                quat_x = quat[:, 1]
                quat_y = quat[:, 2]

                body_z_dot_world_z = (
                    1.0
                    - 2.0
                    * (
                        quat_x * quat_x
                        + quat_y * quat_y
                    )
                )

                tilt_deg = torch.rad2deg(
                    torch.acos(
                        torch.clamp(
                            body_z_dot_world_z,
                            -1.0,
                            1.0,
                        )
                    )
                )

                position_log[active, step] = (
                    pos[active]
                )
                goal_log[active, step] = (
                    controller_goal[active]
                )
                error_xyz_log[active, step] = (
                    error_xyz[active]
                )
                error_norm_log[active, step] = (
                    error_norm[active]
                )

                quat_log[active, step] = quat[active]
                rpy_log[active, step] = rpy[active]
                tilt_deg_log[active, step] = (
                    tilt_deg[active]
                )

                linear_velocity_log[
                    active,
                    step,
                ] = vel[active]

                angular_velocity_log[
                    active,
                    step,
                ] = ang_vel[active]

                speed_log[active, step] = speed[active]

                actions_log[active, step] = (
                    actions[active]
                )

                motor_command_log[
                    active,
                    step,
                ] = motor_command[active]

                valid_log[active, step] = True

                if (
                    bool(active[0].item())
                    and (
                        step == 0
                        or (step + 1)
                        % args_cli.print_every
                        == 0
                    )
                ):
                    m = motor_command[0]

                    print(
                        f"step={step+1:4d} "
                        f"t={(step+1)/env_cfg.policy_rate_hz:7.3f}s  "
                        f"dist={error_norm[0].item():8.4f}m  "
                        f"speed={speed[0].item():8.4f}m/s  "
                        f"tilt={tilt_deg[0].item():7.2f}deg  "
                        f"motors=["
                        f"{m[0].item():.4f}, "
                        f"{m[1].item():.4f}, "
                        f"{m[2].item():.4f}, "
                        f"{m[3].item():.4f}]"
                    )

                obs_dict, reward, terminated, truncated, info = (
                    envs_gym.step(actions)
                )

                reward_flat = reward.reshape(-1)

                reward_log[active, step] = (
                    reward_flat[active]
                )

                terminated_log[
                    active,
                    step,
                ] = terminated[active]

                truncated_log[
                    active,
                    step,
                ] = truncated[active]

                actual_motor = getattr(
                    env,
                    "_motor_speeds",
                    None,
                )

                if torch.is_tensor(actual_motor):
                    actual_motor = actual_motor.detach()

                    if actual_motor.ndim == 1:
                        actual_motor = (
                            actual_motor
                            .view(1, -1)
                            .repeat(n, 1)
                        )

                    if (
                        actual_motor.shape[0] >= n
                        and actual_motor.shape[1] >= 4
                    ):
                        actual_motor_log[
                            active,
                            step,
                        ] = actual_motor[
                            active,
                            :4,
                        ]

                done = terminated | truncated
                newly_done = active & done

                if bool(newly_done.any().item()):
                    first_terminated[
                        newly_done
                    ] = terminated[newly_done]

                    first_truncated[
                        newly_done
                    ] = truncated[newly_done]

                    first_end_step[
                        newly_done
                    ] = step + 1

                    finished[newly_done] = True

                steps_run = step + 1

    unfinished = first_end_step < 0
    first_end_step[unfinished] = steps_run

    print("\n" + "=" * 100)
    print("FIRST-EPISODE EVALUATION COMPLETE")
    print("=" * 100)
    print("environments :", n)
    print("steps run    :", steps_run)
    print(
        "terminated   :",
        int(first_terminated.sum().item()),
    )
    print(
        "truncated    :",
        int(first_truncated.sum().item()),
    )
    print(
        "through horizon without done:",
        int((~finished).sum().item()),
    )
    print("=" * 100)

    # ------------------------------------------------------------
    # Move logs to CPU once, after rollout.
    # ------------------------------------------------------------
    pos_np = (
        position_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )
    goal_np = (
        goal_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )
    err_xyz_np = (
        error_xyz_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )
    err_norm_np = (
        error_norm_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )
    quat_np = (
        quat_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )
    rpy_np = (
        rpy_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )
    tilt_np = (
        tilt_deg_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )
    vel_np = (
        linear_velocity_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )
    ang_np = (
        angular_velocity_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )
    speed_np = (
        speed_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )
    actions_np = (
        actions_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )
    motors_np = (
        motor_command_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )
    actual_motors_np = (
        actual_motor_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )
    reward_np = (
        reward_log[:, :steps_run]
        .detach()
        .cpu()
        .numpy()
    )

    end_steps_np = (
        first_end_step
        .detach()
        .cpu()
        .numpy()
    )

    terminated_np = (
        first_terminated
        .detach()
        .cpu()
        .numpy()
    )

    truncated_np = (
        first_truncated
        .detach()
        .cpu()
        .numpy()
    )

    mass_np = (
        actual_mass.detach().cpu().numpy()
    )
    k_eta_np = (
        actual_k_eta.detach().cpu().numpy()
    )
    tau_m_np = (
        actual_tau_m.detach().cpu().numpy()
    )
    k_torque_np = (
        actual_k_torque.detach().cpu().numpy()
    )
    inertia_np = (
        actual_inertia_diag
        .detach()
        .cpu()
        .numpy()
    )

    selected_latency_np = (
        selected_latency_index
        .detach()
        .cpu()
        .numpy()
    )
    raw_latency_np = (
        raw_latency_index
        .detach()
        .cpu()
        .numpy()
    )
    delay_steps_np = (
        actual_delay_steps
        .detach()
        .cpu()
        .numpy()
    )
    delay_ms_np = (
        delay_ms.detach().cpu().numpy()
    )

    controller_mass_np = (
        controller_mass_assumed
        .detach()
        .cpu()
        .numpy()
    )
    controller_k_eta_np = (
        controller_k_eta_assumed
        .detach()
        .cpu()
        .numpy()
    )
    controller_mass_thrust_np = (
        controller_mass_thrust
        .detach()
        .cpu()
        .numpy()
    )

    thresholds = [
        0.50,
        0.25,
        0.15,
        0.10,
        0.05,
        0.025,
        0.01,
    ]

    def threshold_key(prefix, threshold):
        label = (
            f"{threshold:.3f}"
            .rstrip("0")
            .rstrip(".")
            .replace(".", "p")
        )
        return f"{prefix}_{label}m_s"

    def first_hit_time(values, threshold, rate):
        hits = np.where(values <= threshold)[0]

        if len(hits) == 0:
            return float("nan")

        return float(hits[0]) / float(rate)

    def first_hit_speed(
        values,
        speeds,
        threshold,
    ):
        hits = np.where(values <= threshold)[0]

        if len(hits) == 0:
            return float("nan")

        index = int(hits[0])

        if (
            index >= len(speeds)
            or not np.isfinite(speeds[index])
        ):
            return float("nan")

        return float(speeds[index])


    def enter_and_stay_time(
        values,
        threshold,
        rate,
    ):
        if len(values) == 0:
            return float("nan")

        suffix_max = np.maximum.accumulate(
            values[::-1]
        )[::-1]

        hits = np.where(
            suffix_max <= threshold
        )[0]

        if len(hits) == 0:
            return float("nan")

        return float(hits[0]) / float(rate)

    metric_rows = []

    # BENCHMARK_FINAL_METRICS_V1
    # Every environment already has the authoritative displacement that
    # was actually commanded after reset/goal synchronization.
    commanded_goal_offset_np = (
        commanded_goal_offset
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    for env_index in range(n):
        length = int(end_steps_np[env_index])

        if length <= 0:
            continue

        e = err_norm_np[env_index, :length]
        exyz = err_xyz_np[env_index, :length]
        pxyz = pos_np[env_index, :length]
        gxyz = goal_np[env_index, :length]
        v = speed_np[env_index, :length]
        omega = ang_np[env_index, :length]
        tilt = tilt_np[env_index, :length]
        motors = motors_np[env_index, :length]

        omega_norm = np.linalg.norm(
            omega,
            axis=1,
        )

        motor_sat = (
            (motors <= 1.0e-6)
            | (motors >= 1.0 - 1.0e-6)
        )

        if length >= 2:
            motor_diff = np.diff(
                motors,
                axis=0,
            )

            mean_abs_motor_step = float(
                np.nanmean(
                    np.abs(motor_diff)
                )
            )

            rms_motor_step = float(
                np.sqrt(
                    np.nanmean(
                        motor_diff ** 2
                    )
                )
            )
        else:
            mean_abs_motor_step = float("nan")
            rms_motor_step = float("nan")

        # Actual simulated motor state after environment latency and
        # motor dynamics. Keep this separate from controller command.
        actual_motors = (
            actual_motors_np[
                env_index,
                :length,
            ]
        )

        actual_motor_finite = np.isfinite(
            actual_motors
        )

        if np.any(actual_motor_finite):
            actual_motor_min = float(
                np.min(
                    actual_motors[
                        actual_motor_finite
                    ]
                )
            )
            actual_motor_max = float(
                np.max(
                    actual_motors[
                        actual_motor_finite
                    ]
                )
            )

            actual_motor_sat_mask = (
                (
                    actual_motors <= 1.0e-3
                )
                |
                (
                    actual_motors >= 1.0 - 1.0e-3
                )
            ) & actual_motor_finite

            actual_motor_saturation_fraction = float(
                np.sum(actual_motor_sat_mask)
                / np.sum(actual_motor_finite)
            )
        else:
            actual_motor_min = float("nan")
            actual_motor_max = float("nan")
            actual_motor_saturation_fraction = (
                float("nan")
            )

        if length >= 2:
            actual_motor_diff = np.diff(
                actual_motors,
                axis=0,
            )

            finite_diff = np.isfinite(
                actual_motor_diff
            )

            if np.any(finite_diff):
                actual_mean_abs_motor_step = float(
                    np.mean(
                        np.abs(
                            actual_motor_diff[
                                finite_diff
                            ]
                        )
                    )
                )

                actual_rms_motor_step = float(
                    np.sqrt(
                        np.mean(
                            actual_motor_diff[
                                finite_diff
                            ] ** 2
                        )
                    )
                )
            else:
                actual_mean_abs_motor_step = (
                    float("nan")
                )
                actual_rms_motor_step = float("nan")
        else:
            actual_mean_abs_motor_step = float("nan")
            actual_rms_motor_step = float("nan")

        row = {
            "env_index": env_index,
            "seed": args_cli.seed,
            "calibration_mode": (
                args_cli.calibration_mode
            ),
            "zero_latency_forced": bool(
                args_cli.zero_latency
            ),
            "episode_length_steps": length,
            "episode_duration_s": (
                length
                / float(env_cfg.policy_rate_hz)
            ),
            "terminated": bool(
                terminated_np[env_index]
            ),
            "truncated": bool(
                truncated_np[env_index]
            ),
            "initial_error_m": float(e[0]),
            "minimum_error_m": float(
                np.nanmin(e)
            ),
            "final_error_m": float(e[-1]),
            "final_error_x_m": float(
                exyz[-1, 0]
            ),
            "final_error_y_m": float(
                exyz[-1, 1]
            ),
            "final_error_z_m": float(
                exyz[-1, 2]
            ),
            "peak_speed_mps": float(
                np.nanmax(v)
            ),
            "final_speed_mps": float(v[-1]),
            "peak_angular_rate_degps": float(
                np.nanmax(omega_norm)
            ),
            "peak_tilt_deg": float(
                np.nanmax(tilt)
            ),
            "motor_min": float(
                np.nanmin(motors)
            ),
            "motor_max": float(
                np.nanmax(motors)
            ),
            "motor_saturation_fraction": float(
                np.nanmean(motor_sat)
            ),
            "mean_abs_motor_step": (
                mean_abs_motor_step
            ),
            "rms_motor_step": rms_motor_step,

            # env._motor_speeds: realized simulated motor state.
            "actual_motor_min": actual_motor_min,
            "actual_motor_max": actual_motor_max,
            "actual_motor_saturation_fraction": (
                actual_motor_saturation_fraction
            ),
            "actual_mean_abs_motor_step": (
                actual_mean_abs_motor_step
            ),
            "actual_rms_motor_step": (
                actual_rms_motor_step
            ),
            "plant_mass_kg": float(
                mass_np[env_index]
            ),
            "plant_k_eta": float(
                k_eta_np[env_index]
            ),
            "plant_tau_m_s": float(
                tau_m_np[env_index]
            ),
            "plant_k_torque": float(
                k_torque_np[env_index]
            ),
            "plant_Ixx": float(
                inertia_np[env_index, 0]
            ),
            "plant_Iyy": float(
                inertia_np[env_index, 1]
            ),
            "plant_Izz": float(
                inertia_np[env_index, 2]
            ),
            "raw_latency_index": int(
                raw_latency_np[env_index]
            ),
            "selected_latency_index": int(
                selected_latency_np[env_index]
            ),
            "delay_steps": int(
                delay_steps_np[env_index]
            ),
            "delay_ms": float(
                delay_ms_np[env_index]
            ),
            "controller_mass_kg": float(
                controller_mass_np[env_index]
            ),
            "controller_k_eta": float(
                controller_k_eta_np[env_index]
            ),
            "controller_mass_thrust": float(
                controller_mass_thrust_np[
                    env_index
                ]
            ),
        }

        for threshold in thresholds:
            row[
                threshold_key(
                    "first",
                    threshold,
                )
            ] = first_hit_time(
                e,
                threshold,
                env_cfg.policy_rate_hz,
            )

            row[
                threshold_key(
                    "stay",
                    threshold,
                )
            ] = enter_and_stay_time(
                e,
                threshold,
                env_cfg.policy_rate_hz,
            )

        # Speed at the instant each position threshold is first entered.
        # This separates "passed through the catch region quickly" from
        # "arrived slow enough to plausibly capture/settle".
        for threshold in thresholds:
            row[
                threshold_key(
                    "speed_first_mps",
                    threshold,
                )
            ] = first_hit_speed(
                e,
                speed_np[env_index, :length],
                threshold,
            )

        # Signed-axis / motion-direction overshoot.
        row["directional_overshoot_m"] = float("nan")
        row["overshoot_x_m"] = float("nan")
        row["overshoot_y_m"] = float("nan")
        row["overshoot_z_m"] = float("nan")

        goal_offset_env = (
            commanded_goal_offset_np[env_index]
        )

        offset_norm = float(
            np.linalg.norm(goal_offset_env)
        )

        if offset_norm > 1.0e-12:
            direction = (
                goal_offset_env / offset_norm
            )

            initial_position_env = (
                initial_pos[env_index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            displacement_along_trajectory = (
                pxyz - initial_position_env
            )

            progress = (
                displacement_along_trajectory
                @ direction
            )

            row[
                "directional_overshoot_m"
            ] = float(
                max(
                    np.nanmax(progress)
                    - offset_norm,
                    0.0,
                )
            )

        # Per-axis signed overshoot beyond the commanded goal.
        for axis, key in enumerate(
            (
                "overshoot_x_m",
                "overshoot_y_m",
                "overshoot_z_m",
            )
        ):
            delta = goal_offset_env[axis]

            if abs(delta) <= 1.0e-12:
                row[key] = 0.0

            elif delta > 0.0:
                row[key] = float(
                    max(
                        np.nanmax(
                            pxyz[:, axis]
                            - gxyz[:, axis]
                        ),
                        0.0,
                    )
                )

            else:
                row[key] = float(
                    max(
                        np.nanmax(
                            gxyz[:, axis]
                            - pxyz[:, axis]
                        ),
                        0.0,
                    )
                )

        row["commanded_dx_m"] = float(
            goal_offset_env[0]
        )
        row["commanded_dy_m"] = float(
            goal_offset_env[1]
        )
        row["commanded_dz_m"] = float(
            goal_offset_env[2]
        )
        row["commanded_distance_m"] = (
            offset_norm
        )

        metric_rows.append(row)

    # ------------------------------------------------------------
    # Aggregate statistics.
    # ------------------------------------------------------------
    def finite_stats(key):
        values = np.asarray(
            [
                row[key]
                for row in metric_rows
            ],
            dtype=np.float64,
        )

        values = values[np.isfinite(values)]

        if len(values) == 0:
            return {
                "mean": None,
                "median": None,
                "std": None,
            }

        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
        }

    aggregate = {
        "run_name": run_name,
        "calibration_mode": (
            args_cli.calibration_mode
        ),
        "num_envs": n,
        "seed": args_cli.seed,
        "policy_rate_hz": float(
            env_cfg.policy_rate_hz
        ),
        "queue_length": queue_length,
        "zero_latency_forced": bool(
            args_cli.zero_latency
        ),
        "terminated_count": int(
            np.sum(terminated_np)
        ),
        "truncated_count": int(
            np.sum(truncated_np)
        ),
        "done_fraction": float(
            np.mean(
                terminated_np
                | truncated_np
            )
        ),
        "through_horizon_fraction": float(
            np.mean(
                ~(
                    terminated_np
                    | truncated_np
                )
            )
        ),
        "final_error_m": finite_stats(
            "final_error_m"
        ),
        "minimum_error_m": finite_stats(
            "minimum_error_m"
        ),
        "peak_speed_mps": finite_stats(
            "peak_speed_mps"
        ),
        "peak_tilt_deg": finite_stats(
            "peak_tilt_deg"
        ),
        "motor_saturation_fraction": (
            finite_stats(
                "motor_saturation_fraction"
            )
        ),
        "thresholds": {},
    }

    for threshold in thresholds:
        first_key = threshold_key(
            "first",
            threshold,
        )
        stay_key = threshold_key(
            "stay",
            threshold,
        )

        speed_first_key = threshold_key(
            "speed_first_mps",
            threshold,
        )

        first_values = np.asarray(
            [row[first_key] for row in metric_rows],
            dtype=np.float64,
        )

        stay_values = np.asarray(
            [row[stay_key] for row in metric_rows],
            dtype=np.float64,
        )

        aggregate["thresholds"][
            str(threshold)
        ] = {
            "first_entry": finite_stats(
                first_key
            ),
            "enter_and_stay": finite_stats(
                stay_key
            ),
            "speed_at_first_entry_mps": (
                finite_stats(
                    speed_first_key
                )
            ),
            "reached_fraction": float(
                np.mean(
                    np.isfinite(first_values)
                )
            ),
            "stay_fraction": float(
                np.mean(
                    np.isfinite(stay_values)
                )
            ),
        }

    # ------------------------------------------------------------
    # Save raw data, per-env CSV, aggregate JSON.
    # ------------------------------------------------------------
    raw_path = os.path.join(
        run_dir,
        "trajectories.pt",
    )

    def _snapshot_env_value(*names):
        for name in names:
            if not hasattr(env, name):
                continue

            value = getattr(env, name)

            if torch.is_tensor(value):
                return value.detach().cpu()

            if isinstance(
                value,
                (
                    int,
                    float,
                    bool,
                    str,
                ),
            ):
                return value

        return None

    selected_latency_index = (
        _snapshot_env_value(
            "_control_latency_steps",
            "control_latency_steps",
        )
    )

    if torch.is_tensor(
        selected_latency_index
    ):
        physical_delay_steps = (
            (queue_length - 1)
            - selected_latency_index
        )

        physical_delay_seconds = (
            physical_delay_steps.to(
                torch.float64
            )
            * float(actual_env_step_dt)
        )
    else:
        physical_delay_steps = None
        physical_delay_seconds = None

    plant_parameters = {
        # Values already explicitly read by evaluator.
        "mass_kg": actual_mass.detach().cpu(),
        "k_eta": actual_k_eta.detach().cpu(),
        "k_torque": (
            actual_k_torque.detach().cpu()
        ),
        "tau_m_s": actual_tau_m.detach().cpu(),

        # Additional randomized plant/controller quantities when
        # exposed by this environment version.
        "inertia": _snapshot_env_value(
            "_robot_inertia",
            "inertia_tensor",
        ),
        "arm_length_m": _snapshot_env_value(
            "_arm_length",
            "arm_length",
        ),
        "k_m": _snapshot_env_value(
            "_k_m",
            "k_m",
        ),
        "kp_att": _snapshot_env_value(
            "_kp_att",
            "kp_att",
        ),
        "kd_att": _snapshot_env_value(
            "_kd_att",
            "kd_att",
        ),
        "kp_omega": _snapshot_env_value(
            "_kp_omega",
            "kp_omega",
        ),
        "kd_omega": _snapshot_env_value(
            "_kd_omega",
            "kd_omega",
        ),
        "thrust_to_weight": _snapshot_env_value(
            "_thrust_to_weight",
            "thrust_to_weight",
        ),

        # Preserve both the environment queue index and the actual
        # physical delay implied by that queue representation.
        "control_latency_selected_index": (
            selected_latency_index
        ),
        "control_latency_physical_steps": (
            physical_delay_steps
        ),
        "control_latency_physical_seconds": (
            physical_delay_seconds
        ),
    }

    torch.save(
        {
            "metadata": {
                "run_name": run_name,
                "calibration_mode": (
                    args_cli.calibration_mode
                ),
                "seed": args_cli.seed,
                "num_envs": n,
                "policy_rate_hz": float(
                    env_cfg.policy_rate_hz
                ),
                "steps_run": steps_run,
                "goal_offset": (
                    args_cli.goal_offset
                ),
                "hold_initial_pose": bool(
                    args_cli.hold_initial_pose
                ),
                "pre_hover_s": float(
                    args_cli.pre_hover_s
                ),
                "pre_hover_steps": int(
                    pre_hover_steps
                ),
                "zero_latency": bool(
                    args_cli.zero_latency
                ),
                "queue_length": queue_length,
            },
            "plant_parameters": plant_parameters,
            "initial_position": (
                initial_pos.detach().cpu()
            ),
            "absolute_goal": (
                initial_controller_goal
                .detach()
                .cpu()
            ),
            "commanded_goal_offset": (
                commanded_goal_offset.detach().cpu()
            ),
            "position": (
                position_log[:, :steps_run]
                .cpu()
            ),
            "goal": (
                goal_log[:, :steps_run]
                .cpu()
            ),
            "error_xyz": (
                error_xyz_log[:, :steps_run]
                .cpu()
            ),
            "error_norm": (
                error_norm_log[:, :steps_run]
                .cpu()
            ),
            "quaternion_wxyz": (
                quat_log[:, :steps_run]
                .cpu()
            ),
            "rpy_rad": (
                rpy_log[:, :steps_run]
                .cpu()
            ),
            "tilt_deg": (
                tilt_deg_log[:, :steps_run]
                .cpu()
            ),
            "linear_velocity": (
                linear_velocity_log[
                    :,
                    :steps_run,
                ].cpu()
            ),
            "angular_velocity_degps": (
                angular_velocity_log[
                    :,
                    :steps_run,
                ].cpu()
            ),
            "speed": (
                speed_log[:, :steps_run]
                .cpu()
            ),
            "actions_srt": (
                actions_log[:, :steps_run]
                .cpu()
            ),
            "motor_command_normalized": (
                motor_command_log[
                    :,
                    :steps_run,
                ].cpu()
            ),
            "actual_motor_state": (
                actual_motor_log[
                    :,
                    :steps_run,
                ].cpu()
            ),
            "reward": (
                reward_log[:, :steps_run]
                .cpu()
            ),
            "valid_first_episode": (
                valid_log[:, :steps_run]
                .cpu()
            ),
            "terminated": (
                terminated_log[
                    :,
                    :steps_run,
                ].cpu()
            ),
            "truncated": (
                truncated_log[
                    :,
                    :steps_run,
                ].cpu()
            ),
            "first_end_step": (
                first_end_step.cpu()
            ),
        },
        raw_path,
    )

    metrics_path = os.path.join(
        run_dir,
        "per_env_metrics.csv",
    )

    if metric_rows:
        with open(
            metrics_path,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    metric_rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(metric_rows)

    aggregate_path = os.path.join(
        run_dir,
        "aggregate_summary.json",
    )

    with open(
        aggregate_path,
        "w",
    ) as f:
        json.dump(
            aggregate,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Console aggregate summary.
    # ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("MELLINGER AGGREGATE SUMMARY")
    print("=" * 100)
    print("calibration mode :", args_cli.calibration_mode)
    print("num envs         :", n)
    print(
        "terminated       :",
        aggregate["terminated_count"],
    )
    print(
        "truncated        :",
        aggregate["truncated_count"],
    )
    print(
        "through horizon  :",
        f"{100.0 * aggregate['through_horizon_fraction']:.2f}%",
    )

    for key, label in (
        ("final_error_m", "final error"),
        ("minimum_error_m", "minimum error"),
        ("peak_speed_mps", "peak speed"),
        ("peak_tilt_deg", "peak tilt"),
    ):
        stats = aggregate[key]

        print(
            f"{label:17s}: "
            f"mean={stats['mean']}  "
            f"median={stats['median']}  "
            f"std={stats['std']}"
        )

    print("\nTHRESHOLD PERFORMANCE")
    print("-" * 100)

    for threshold in thresholds:
        tstats = aggregate["thresholds"][
            str(threshold)
        ]

        print(
            f"{threshold:6.3f} m | "
            f"reached={100.0*tstats['reached_fraction']:6.2f}% | "
            f"stay={100.0*tstats['stay_fraction']:6.2f}% | "
            f"mean first="
            f"{tstats['first_entry']['mean']} | "
            f"mean stay="
            f"{tstats['enter_and_stay']['mean']}"
        )

    print("\nSaved:")
    print(" raw trajectories :", raw_path)
    print(" per-env metrics  :", metrics_path)
    print(" aggregate summary:", aggregate_path)

    # ------------------------------------------------------------
    # Followed-robot plots analogous to eval_rslrl.py.
    # ------------------------------------------------------------
    if args_cli.follow_robot >= 0:
        import matplotlib.pyplot as plt

        robot = args_cli.follow_robot
        length = int(end_steps_np[robot])

        t = (
            np.arange(length)
            / float(env_cfg.policy_rate_hz)
        )

        pos = pos_np[robot, :length]
        goal = goal_np[robot, :length]
        vel = vel_np[robot, :length]
        rpy = rpy_np[robot, :length]
        ang = ang_np[robot, :length]
        action = actions_np[robot, :length]
        motors = motors_np[robot, :length]
        err_xyz = err_xyz_np[robot, :length]
        err_norm = err_norm_np[robot, :length]

        # Same 4x3 state layout as eval_rslrl.py.
        fig, axes = plt.subplots(
            4,
            3,
            figsize=(15, 14),
            sharex=True,
        )

        labels = ("X", "Y", "Z")

        for axis in range(3):
            axes[0, axis].plot(
                t,
                pos[:, axis],
                label=f"Quad {labels[axis]}",
            )
            axes[0, axis].plot(
                t,
                goal[:, axis],
                linestyle="--",
                label=f"Goal {labels[axis]}",
            )
            axes[0, axis].legend(
                loc="best"
            )

            axes[1, axis].plot(
                t,
                vel[:, axis],
                label=f"Vel {labels[axis]}",
            )
            axes[1, axis].legend(
                loc="best"
            )

        angle_labels = (
            "Roll",
            "Pitch",
            "Yaw",
        )

        for axis in range(3):
            axes[2, axis].plot(
                t,
                rpy[:, axis],
                label=angle_labels[axis],
            )
            axes[2, axis].legend(
                loc="best"
            )

            axes[3, axis].plot(
                t,
                ang[:, axis],
                label=(
                    f"Ang Vel "
                    f"{labels[axis]} "
                    f"(deg/s)"
                ),
            )
            axes[3, axis].legend(
                loc="best"
            )
            axes[3, axis].set_xlabel(
                "Time (s)"
            )

        fig.tight_layout()

        state_plot = os.path.join(
            run_dir,
            "state_plot.png",
        )
        fig.savefig(state_plot)
        plt.close(fig)

        # SRT actions.
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 8),
            sharex=True,
        )

        for channel, ax in enumerate(
            axes.flat
        ):
            ax.plot(
                t,
                action[:, channel],
                label=f"Action {channel + 1}",
            )
            ax.legend(loc="best")
            ax.set_xlabel("Time (s)")

        fig.tight_layout()

        action_plot = os.path.join(
            run_dir,
            "actions.png",
        )
        fig.savefig(action_plot)
        plt.close(fig)

        # Normalized motor commands.
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 8),
            sharex=True,
        )

        for channel, ax in enumerate(
            axes.flat
        ):
            ax.plot(
                t,
                motors[:, channel],
                label=f"Motor {channel + 1}",
            )
            ax.legend(loc="best")
            ax.set_xlabel("Time (s)")

        fig.tight_layout()

        motor_plot = os.path.join(
            run_dir,
            "motor_commands.png",
        )
        fig.savefig(motor_plot)
        plt.close(fig)

        # Signed XYZ error.
        fig = plt.figure(
            figsize=(10, 6)
        )

        for axis in range(3):
            plt.plot(
                t,
                err_xyz[:, axis],
                label=(
                    f"{labels[axis]} error"
                ),
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Goal - position (m)")
        plt.legend(loc="best")
        plt.tight_layout()

        signed_error_plot = os.path.join(
            run_dir,
            "signed_position_error.png",
        )
        plt.savefig(signed_error_plot)
        plt.close(fig)

        # Position-error norm.
        fig = plt.figure(
            figsize=(10, 6)
        )
        plt.plot(
            t,
            err_norm,
            label="Position error norm",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Position error (m)")
        plt.legend(loc="best")
        plt.tight_layout()

        error_norm_plot = os.path.join(
            run_dir,
            "position_error_norm.png",
        )
        plt.savefig(error_norm_plot)
        plt.close(fig)

        print("\nPlots:")
        print(" ", state_plot)
        print(" ", action_plot)
        print(" ", motor_plot)
        print(" ", signed_error_plot)
        print(" ", error_norm_plot)

    envs_gym.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
