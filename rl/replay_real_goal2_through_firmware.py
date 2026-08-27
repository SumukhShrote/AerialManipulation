from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from controllers.cf_mellinger_firmware import (
    CrazyflieFirmwareMellinger,
)


DATA_PATH = Path(
    "/home/sumukh/AerialManipulation/"
    "b3_goal1_goal2_real_state.npz"
)

CSV_PATH = Path(
    "/home/sumukh/AerialManipulation/"
    "b3_goal2_real_state_firmware_replay.csv"
)

RATE_HZ = 500.0
DT = 1.0 / RATE_HZ


def normalize_quaternions_wxyz(q):
    q = np.asarray(q, dtype=np.float64).copy()

    q /= np.linalg.norm(
        q,
        axis=1,
        keepdims=True,
    )

    # Keep signs continuous.
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0.0:
            q[i] *= -1.0

    return q


def wxyz_to_xyzw(q):
    return q[:, [1, 2, 3, 0]]


def xyzw_to_wxyz(q):
    return q[:, [3, 0, 1, 2]]


def interp_columns(t, x, tq):
    out = np.empty(
        (len(tq), x.shape[1]),
        dtype=np.float64,
    )

    for j in range(x.shape[1]):
        out[:, j] = np.interp(
            tq,
            t,
            x[:, j],
        )

    return out


def score_rmse(candidate, reference, mask):
    c = candidate[mask]
    r = reference[mask]

    err = np.sqrt(
        np.mean(
            np.square(c - r)
        )
    )

    scale = np.sqrt(
        np.mean(
            np.square(r)
        )
    )

    return err / max(scale, 1.0e-9)


def rpy_deg_from_rotation(rot):
    # Standard XYZ roll/pitch/yaw.
    return rot.as_euler(
        "xyz",
        degrees=True,
    )


d = np.load(
    DATA_PATH,
    allow_pickle=False,
)

state_topic = str(
    d["state_topic"].item()
)

t = np.asarray(
    d["t"],
    dtype=np.float64,
)

pos = np.asarray(
    d["pos"],
    dtype=np.float64,
)

quat_wxyz = normalize_quaternions_wxyz(
    d["quat_wxyz"]
)

lin_raw = np.asarray(
    d["lin_raw"],
    dtype=np.float64,
)

ang_raw = np.asarray(
    d["ang_raw"],
    dtype=np.float64,
)

t_goal1 = float(
    d["t_goal1"]
)

t_goal2 = float(
    d["t_goal2"]
)

goal1 = np.asarray(
    d["goal1"],
    dtype=np.float64,
)

goal2 = np.asarray(
    d["goal2"],
    dtype=np.float64,
)

yaw_goal = float(
    d["goal_yaw_rad"]
)

delivery = float(
    d["assumed_delivery_delay_s"]
)

t_goal1_delivery = (
    t_goal1 + delivery
)

t_goal2_delivery = (
    t_goal2 + delivery
)


# =============================================================================
# Quaternion representation.
# =============================================================================

rot = Rotation.from_quat(
    wxyz_to_xyzw(
        quat_wxyz
    )
)

slerp_raw = Slerp(
    t,
    rot,
)


# =============================================================================
# FRAME AUDIT: LINEAR VELOCITY
#
# get_action_from_gc expects WORLD linear velocity.
# Determine whether the recorded twist is already world or is body.
# =============================================================================

median_dt = float(
    np.median(
        np.diff(t)
    )
)

tu = np.arange(
    t[0],
    t[-1],
    median_dt,
)

pos_u = interp_columns(
    t,
    pos,
    tu,
)

lin_u = interp_columns(
    t,
    lin_raw,
    tu,
)

ang_u = interp_columns(
    t,
    ang_raw,
    tu,
)

rot_u = slerp_raw(tu)

# Smooth position derivative to obtain an independent world velocity.
win = int(
    round(
        0.075 / median_dt
    )
)

if win % 2 == 0:
    win += 1

win = max(
    7,
    min(
        win,
        len(tu) - 1
        if len(tu) % 2 == 0
        else len(tu),
    ),
)

if win % 2 == 0:
    win -= 1

vel_from_position = np.column_stack(
    [
        savgol_filter(
            pos_u[:, j],
            win,
            3,
            deriv=1,
            delta=median_dt,
        )
        for j in range(3)
    ]
)

lin_candidate_world = lin_u

lin_candidate_body_to_world = (
    rot_u.apply(
        lin_u
    )
)

lin_eval = (
    (tu >= t_goal1_delivery)
    & (tu <= t_goal2_delivery + 0.12)
)

# Avoid scoring totally stationary points only.
lin_dynamic = (
    np.linalg.norm(
        vel_from_position,
        axis=1,
    ) > 0.05
)

lin_mask = (
    lin_eval
    & lin_dynamic
)

if np.count_nonzero(lin_mask) < 20:
    lin_mask = lin_eval

lin_score_world = score_rmse(
    lin_candidate_world,
    vel_from_position,
    lin_mask,
)

lin_score_body = score_rmse(
    lin_candidate_body_to_world,
    vel_from_position,
    lin_mask,
)

if lin_score_world <= lin_score_body:
    linear_frame = "WORLD"
else:
    linear_frame = "BODY"


# =============================================================================
# FRAME AUDIT: ANGULAR VELOCITY
#
# Derive WORLD omega directly from the quaternion increments.
# Compare:
#   raw angular twist interpreted as WORLD
#   raw angular twist interpreted as BODY and rotated to WORLD.
# =============================================================================

relative_world = (
    rot_u[1:]
    * rot_u[:-1].inv()
)

omega_q_world = (
    relative_world.as_rotvec()
    / median_dt
)

tmid = 0.5 * (
    tu[:-1]
    + tu[1:]
)

ang_candidate_world_samples = ang_u

ang_candidate_body_to_world_samples = (
    rot_u.apply(
        ang_u
    )
)

ang_world_mid = 0.5 * (
    ang_candidate_world_samples[:-1]
    + ang_candidate_world_samples[1:]
)

ang_body_mid = 0.5 * (
    ang_candidate_body_to_world_samples[:-1]
    + ang_candidate_body_to_world_samples[1:]
)

# Mild smoothing for the quaternion finite-difference reference.
if len(omega_q_world) >= 11:
    awin = 11

    omega_q_world_sm = np.column_stack(
        [
            savgol_filter(
                omega_q_world[:, j],
                awin,
                3,
            )
            for j in range(3)
        ]
    )
else:
    omega_q_world_sm = omega_q_world

ang_eval = (
    (tmid >= t_goal1_delivery)
    & (tmid <= t_goal2_delivery + 0.12)
)

ang_dynamic = (
    np.linalg.norm(
        omega_q_world_sm,
        axis=1,
    ) > 0.15
)

ang_mask = (
    ang_eval
    & ang_dynamic
)

if np.count_nonzero(ang_mask) < 20:
    ang_mask = ang_eval

ang_score_world = score_rmse(
    ang_world_mid,
    omega_q_world_sm,
    ang_mask,
)

ang_score_body = score_rmse(
    ang_body_mid,
    omega_q_world_sm,
    ang_mask,
)

if ang_score_world <= ang_score_body:
    angular_frame = "WORLD"
else:
    angular_frame = "BODY"


print("=" * 110)
print("REAL-STATE INPUT FRAME AUDIT")
print("=" * 110)

print("state topic                 :", state_topic)
print("state median dt             :", f"{median_dt:.9f} s")
print("state nominal Hz            :", f"{1.0/median_dt:.3f}")

print()
print("LINEAR TWIST")
print(
    "  normalized RMSE if WORLD :",
    f"{lin_score_world:.5f}",
)
print(
    "  normalized RMSE if BODY  :",
    f"{lin_score_body:.5f}",
)
print(
    "  selected frame           :",
    linear_frame,
)

print()
print("ANGULAR TWIST")
print(
    "  normalized RMSE if WORLD :",
    f"{ang_score_world:.5f}",
)
print(
    "  normalized RMSE if BODY  :",
    f"{ang_score_body:.5f}",
)
print(
    "  selected frame           :",
    angular_frame,
)

print("=" * 110)


# =============================================================================
# Resolve recorded velocities into WORLD coordinates.
# =============================================================================

if linear_frame == "WORLD":
    lin_world_samples = lin_raw.copy()
else:
    lin_world_samples = rot.apply(
        lin_raw
    )

if angular_frame == "WORLD":
    ang_world_samples = ang_raw.copy()
else:
    ang_world_samples = rot.apply(
        ang_raw
    )


# =============================================================================
# Build exact 500 Hz replay.
#
# Start at GOAL1 estimated firmware delivery.
# Feed measured real motion for the entire GOAL1 -> GOAL2 interval.
# Switch to GOAL2 at the first 500 Hz firmware tick at/after its
# estimated delivery time.
# =============================================================================

t_end = (
    t_goal2_delivery + 0.120
)

if (
    t_goal1_delivery < t[0]
    or t_end > t[-1]
):
    raise RuntimeError(
        "Extracted real state does not cover the full replay interval.\n"
        f"state=[{t[0]:.9f}, {t[-1]:.9f}]\n"
        f"needed=[{t_goal1_delivery:.9f}, {t_end:.9f}]"
    )

num_steps = int(
    math.floor(
        (
            t_end
            - t_goal1_delivery
        )
        / DT
    )
) + 1

t500 = (
    t_goal1_delivery
    + np.arange(
        num_steps,
        dtype=np.float64,
    ) * DT
)

switch_indices = np.flatnonzero(
    t500 >= t_goal2_delivery
)

if len(switch_indices) == 0:
    raise RuntimeError(
        "No 500 Hz sample reaches GOAL2 delivery."
    )

switch_idx = int(
    switch_indices[0]
)

actual_switch_time = float(
    t500[switch_idx]
)

switch_quantization = (
    actual_switch_time
    - t_goal2_delivery
)

print()
print("=" * 110)
print("GOAL DELIVERY / 500 HZ ALIGNMENT")
print("=" * 110)

print(
    "GOAL1 ROS timestamp         :",
    f"{t_goal1:.9f}",
)
print(
    "GOAL2 ROS timestamp         :",
    f"{t_goal2:.9f}",
)
print(
    "assumed delivery delay      :",
    f"{delivery:.6f} s",
)
print(
    "GOAL1 firmware delivery     :",
    f"{t_goal1_delivery:.9f}",
)
print(
    "GOAL2 firmware delivery     :",
    f"{t_goal2_delivery:.9f}",
)
print(
    "500 Hz actual switch tick   :",
    f"{actual_switch_time:.9f}",
)
print(
    "switch quantization         :",
    f"{switch_quantization*1000.0:.3f} ms",
)
print(
    "GOAL1->GOAL2 elapsed        :",
    f"{t_goal2-t_goal1:.9f} s",
)

print("=" * 110)


# =============================================================================
# Interpolate measured real state to 500 Hz.
# =============================================================================

pos500 = interp_columns(
    t,
    pos,
    t500,
)

lin500 = interp_columns(
    t,
    lin_world_samples,
    t500,
)

ang500 = interp_columns(
    t,
    ang_world_samples,
    t500,
)

rot500 = slerp_raw(
    t500
)

quat500_wxyz = xyzw_to_wxyz(
    rot500.as_quat()
)

rpy500_deg = rpy_deg_from_rotation(
    rot500
)

ang500_body = (
    rot500.inv()
    .apply(
        ang500
    )
)


# =============================================================================
# Actual firmware replay.
# =============================================================================

controller = CrazyflieFirmwareMellinger(
    mass_kg=0.046,
    mass_thrust=132000.0,
)

controller.reset()

records = []

first_fw_keys_printed = False

for i in range(num_steps):
    goal = (
        goal1
        if i < switch_idx
        else goal2
    )

    gc = torch.zeros(
        (1, 17),
        dtype=torch.float32,
    )

    gc[0, 0:3] = torch.from_numpy(
        pos500[i]
    ).to(torch.float32)

    gc[0, 3:7] = torch.from_numpy(
        quat500_wxyz[i]
    ).to(torch.float32)

    gc[0, 7:10] = torch.from_numpy(
        lin500[i]
    ).to(torch.float32)

    # Contract is WORLD angular velocity in DEG/S.
    gc[0, 10:13] = torch.from_numpy(
        np.degrees(
            ang500[i]
        )
    ).to(torch.float32)

    gc[0, 13:16] = torch.from_numpy(
        goal
    ).to(torch.float32)

    gc[0, 16] = float(
        yaw_goal
    )

    _, fw = controller.get_action_from_gc(
        gc,
        device="cpu",
    )

    if not first_fw_keys_printed:
        first_fw_keys_printed = True

        print()
        print("=" * 110)
        print("FIRMWARE RESULT KEYS")
        print("=" * 110)
        print(sorted(fw.keys()))
        print("=" * 110)

    pwm = tuple(
        int(v)
        for v in fw["motor_pwm"]
    )

    uncapped = None

    for key in (
        "motor_uncapped",
        "uncapped_motor",
        "motor_pwm_uncapped",
        "uncapped",
    ):
        if key in fw:
            try:
                uncapped = tuple(
                    int(v)
                    for v in fw[key]
                )
            except Exception:
                uncapped = None

            if uncapped is not None:
                break

    t_rel = (
        t500[i]
        - actual_switch_time
    )

    records.append(
        {
            "index": i,
            "time_abs": t500[i],
            "time_from_goal2_delivery": t_rel,
            "goal": 1 if i < switch_idx else 2,

            "x": pos500[i, 0],
            "y": pos500[i, 1],
            "z": pos500[i, 2],

            "roll_deg": rpy500_deg[i, 0],
            "pitch_deg": rpy500_deg[i, 1],
            "yaw_deg": rpy500_deg[i, 2],

            "p_body_radps": ang500_body[i, 0],
            "q_body_radps": ang500_body[i, 1],
            "r_body_radps": ang500_body[i, 2],

            "vx_world": lin500[i, 0],
            "vy_world": lin500[i, 1],
            "vz_world": lin500[i, 2],

            "fw_thrust": float(fw["thrust"]),
            "fw_roll": float(fw["roll"]),
            "fw_pitch": float(fw["pitch"]),
            "fw_yaw": float(fw["yaw"]),

            "was_capped": int(
                bool(
                    fw.get(
                        "was_capped",
                        False,
                    )
                )
            ),

            "pwm1": pwm[0],
            "pwm2": pwm[1],
            "pwm3": pwm[2],
            "pwm4": pwm[3],

            "uncapped1":
                np.nan
                if uncapped is None
                else uncapped[0],

            "uncapped2":
                np.nan
                if uncapped is None
                else uncapped[1],

            "uncapped3":
                np.nan
                if uncapped is None
                else uncapped[2],

            "uncapped4":
                np.nan
                if uncapped is None
                else uncapped[3],
        }
    )


# =============================================================================
# Save complete replay.
# =============================================================================

fieldnames = list(
    records[0].keys()
)

with CSV_PATH.open(
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        records
    )


# =============================================================================
# Critical GOAL2 diagnostic.
# =============================================================================

post = [
    row
    for row in records
    if (
        row["time_from_goal2_delivery"]
        >= -1.0e-9
        and row["time_from_goal2_delivery"]
        <= 0.120 + 1.0e-9
    )
]

print()
print("=" * 132)
print("REAL TRAJECTORY -> ACTUAL B3 SIL MELLINGER")
print("CLOCK ZERO = FIRST 500 HZ TICK RECEIVING GOAL2")
print("=" * 132)

print(
    f"{'t':>7}"
    f"{'roll':>9}"
    f"{'pitch':>9}"
    f"{'p':>10}"
    f"{'q':>10}"
    f"{'fw_roll':>12}"
    f"{'fw_pitch':>12}"
    f"{'thrust':>11}"
    f"{'cap':>6}"
    f"{'PWMmin':>9}"
    f"{'PWMmax':>9}"
)

targets = [
    0.000,
    0.010,
    0.020,
    0.030,
    0.040,
    0.050,
    0.060,
    0.070,
    0.080,
    0.090,
    0.100,
    0.110,
    0.120,
]

for target in targets:
    row = min(
        post,
        key=lambda x: abs(
            x["time_from_goal2_delivery"]
            - target
        ),
    )

    pwm_vals = [
        row["pwm1"],
        row["pwm2"],
        row["pwm3"],
        row["pwm4"],
    ]

    print(
        f"{row['time_from_goal2_delivery']:7.3f}"
        f"{row['roll_deg']:9.2f}"
        f"{row['pitch_deg']:9.2f}"
        f"{row['p_body_radps']:10.3f}"
        f"{row['q_body_radps']:10.3f}"
        f"{row['fw_roll']:12.1f}"
        f"{row['fw_pitch']:12.1f}"
        f"{row['fw_thrust']:11.1f}"
        f"{row['was_capped']:6d}"
        f"{min(pwm_vals):9d}"
        f"{max(pwm_vals):9d}"
    )


fw_roll = np.asarray(
    [
        row["fw_roll"]
        for row in post
    ],
    dtype=np.float64,
)

post_t = np.asarray(
    [
        row["time_from_goal2_delivery"]
        for row in post
    ],
    dtype=np.float64,
)

# First time the controller leaves the strong negative clamp.
release = None

for i in range(len(post)):
    if fw_roll[i] > -30000.0:
        release = post_t[i]
        break

reversal = None

for i in range(len(post)):
    if fw_roll[i] > 10.0:
        reversal = post_t[i]
        break


def nearest_value(target):
    i = int(
        np.argmin(
            np.abs(
                post_t - target
            )
        )
    )

    return (
        post_t[i],
        fw_roll[i],
    )


t20, r20 = nearest_value(0.020)
t40, r40 = nearest_value(0.040)
t60, r60 = nearest_value(0.060)
t80, r80 = nearest_value(0.080)

print()
print("=" * 132)
print("CRITICAL RESULT")
print("=" * 132)

print(
    "first fw_roll > -30000 :",
    release,
)

print(
    "first fw_roll > +10    :",
    reversal,
)

print()
print("REAL-STATE REPLAY")
print(
    f"  ~20 ms fw_roll = {r20:+.1f}"
)
print(
    f"  ~40 ms fw_roll = {r40:+.1f}"
)
print(
    f"  ~60 ms fw_roll = {r60:+.1f}"
)
print(
    f"  ~80 ms fw_roll = {r80:+.1f}"
)

print()
print("CURRENT SIM GAIN-0.57 REFERENCE")
print("  ~20 ms fw_roll = -32000")
print("  ~40 ms fw_roll = -13520")
print("  ~60 ms fw_roll =    +299")
print("  ~80 ms fw_roll =    +520")

print()
print("Interpretation:")
print(
    "  If REAL-state replay stays strongly negative much longer than the"
)
print(
    "  sim reference, the SIL firmware is capable of the correct command"
)
print(
    "  and the SIMULATED STATE TRAJECTORY is causing the early back-off."
)
print(
    "  If REAL-state replay backs off around 40-60 ms in the same way,"
)
print(
    "  stop plant tuning: the replay/input/controller contract is wrong."
)

print()
print("[PASS] full replay CSV:", CSV_PATH)
print("=" * 132)
