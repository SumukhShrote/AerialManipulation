from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CF_FW_ROOT = REPO_ROOT / "third_party" / "crazyflie-firmware"
CF_FW_BUILD = CF_FW_ROOT / "build"

if str(CF_FW_BUILD) not in sys.path:
    sys.path.insert(0, str(CF_FW_BUILD))

try:
    import cffirmware as cf
except ImportError as exc:
    raise ImportError(
        f"Could not import Crazyflie firmware bindings from {CF_FW_BUILD}. "
        "Build them with: "
        "cd third_party/crazyflie-firmware && "
        "make bindings_python PYTHON=\"$(which python)\""
    ) from exc


FIRMWARE_COMMIT = "20515a264bb47235309d2f25a93c7bab037721a6"

B3_MASS_KG = 0.046
B3_MASS_THRUST = 132000.0

# B3_GYRO_LPF_80HZ_V1
#
# Exact Crazyflie lpf2p coefficients for:
#   sample_freq = 1000 Hz
#   cutoff_freq = 80 Hz
#
# Derived from the pinned firmware implementation in filter.c.
B3_GYRO_LPF_B0 = 0.046131802093312926
B3_GYRO_LPF_B1 = 0.09226360418662585
B3_GYRO_LPF_B2 = 0.046131802093312926
B3_GYRO_LPF_A1 = -1.3072850288493234
B3_GYRO_LPF_A2 = 0.4918122372225752


def _set_xyz(v, xyz: Sequence[float]) -> None:
    v.x = float(xyz[0])
    v.y = float(xyz[1])
    v.z = float(xyz[2])


def _quat_wxyz_to_euler_deg(
    quat_wxyz: Sequence[float],
) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


class CrazyflieFirmwareMellinger:
    """Actual Bitcraze controllerMellinger + quadrotor power distribution."""

    def __init__(
        self,
        *,
        mass_kg: float = B3_MASS_KG,
        mass_thrust: float = B3_MASS_THRUST,
    ) -> None:
        self.controller = cf.controllerMellinger_t()
        cf.controllerMellingerInit(self.controller)

        # B3 runtime controller parameters.
        self.controller.mass = float(mass_kg)
        self.controller.massThrust = float(mass_thrust)

        # B3_SIM_MELLINGER_RATE_DAMPING_V1
        #
        # Optional simulation-only RP angular-rate damping override.
        # Default firmware behavior is unchanged unless the environment
        # variable is explicitly supplied.
        _sim_kw_xy = os.environ.get("B3_MELLINGER_KW_XY")

        if _sim_kw_xy is not None:
            self.controller.kw_xy = float(_sim_kw_xy)

            print(
                "[B3 Mellinger] simulation kw_xy override:",
                float(self.controller.kw_xy),
            )

        self.control = cf.control_t()
        self.setpoint = cf.setpoint_t()
        self.sensors = cf.sensorData_t()
        self.state = cf.state_t()

        self.uncapped = cf.motors_thrust_uncapped_t()
        self.pwm = cf.motors_thrust_pwm_t()

        cf.powerDistributionInit()

        self.stabilizer_step = 0

        # B3_GYRO_LPF_80HZ_V1
        self._gyro_lpf_enabled = (
            os.environ.get("B3_GYRO_LPF_ENABLE", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        self._reset_gyro_lpf_state()

        self._configure_position_setpoint_modes()

        # B3_SIM_MELLINGER_TRAJECTORY_V1
        #
        # Simulation baseline: feed the firmware Mellinger a smooth
        # minimum-jerk p/v/a trajectory instead of an instantaneous
        # multi-meter position step.
        self._sim_traj_enabled = (
            os.environ.get("B3_SIM_TRAJ_ENABLE", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )

        self._sim_traj_max_speed_mps = float(
            os.environ.get(
                "B3_SIM_TRAJ_MAX_SPEED_MPS",
                "1.5",
            )
        )

        self._sim_traj_min_duration_s = float(
            os.environ.get(
                "B3_SIM_TRAJ_MIN_DURATION_S",
                "0.8",
            )
        )

        self._sim_traj_start = None
        self._sim_traj_target = None
        self._sim_traj_elapsed_s = 0.0
        self._sim_traj_duration_s = 0.0

    def _reset_gyro_lpf_state(self) -> None:
        # Pinned Crazyflie lpf2pInit() zeros both delay elements.
        self._gyro_lpf_d1 = [0.0, 0.0, 0.0]
        self._gyro_lpf_d2 = [0.0, 0.0, 0.0]

        # The active Isaac bridge supplies body rate at 500 Hz while the
        # physical BMI088/filter path runs at 1000 Hz. Preserve the previous
        # 500-Hz raw sample so we can reconstruct the intervening 1-ms sample.
        self._gyro_prev_raw_deg_s = None

        self._last_gyro_raw_deg_s = (0.0, 0.0, 0.0)
        self._last_gyro_filtered_deg_s = (0.0, 0.0, 0.0)

    def _gyro_lpf_apply_1khz_sample(
        self,
        sample_deg_s: Sequence[float],
    ) -> tuple[float, float, float]:
        out = []

        for axis in range(3):
            sample = float(sample_deg_s[axis])

            # Exact lpf2pApply() recurrence from the pinned firmware.
            d0 = (
                sample
                - self._gyro_lpf_d1[axis] * B3_GYRO_LPF_A1
                - self._gyro_lpf_d2[axis] * B3_GYRO_LPF_A2
            )

            if not math.isfinite(d0):
                d0 = sample

            y = (
                d0 * B3_GYRO_LPF_B0
                + self._gyro_lpf_d1[axis] * B3_GYRO_LPF_B1
                + self._gyro_lpf_d2[axis] * B3_GYRO_LPF_B2
            )

            self._gyro_lpf_d2[axis] = self._gyro_lpf_d1[axis]
            self._gyro_lpf_d1[axis] = d0

            out.append(float(y))

        return tuple(out)

    def _filter_gyro_for_500hz_bridge(
        self,
        raw_deg_s: Sequence[float],
    ) -> tuple[float, float, float]:
        raw = tuple(float(v) for v in raw_deg_s)

        self._last_gyro_raw_deg_s = raw

        if not self._gyro_lpf_enabled:
            self._gyro_prev_raw_deg_s = raw
            self._last_gyro_filtered_deg_s = raw
            return raw

        prev = self._gyro_prev_raw_deg_s

        if prev is None:
            # Startup difference is washed out by the existing pre-hover.
            midpoint = raw
        else:
            # Approximate the real 1-ms BMI088 sample halfway between the
            # two 2-ms PhysX/controller samples.
            midpoint = tuple(
                0.5 * (float(prev[i]) + raw[i])
                for i in range(3)
            )

        # Real software gyro LPF updates at 1 kHz:
        #   t + 1 ms : interpolated physical rate
        #   t + 2 ms : current PhysX physical rate
        self._gyro_lpf_apply_1khz_sample(midpoint)
        filtered = self._gyro_lpf_apply_1khz_sample(raw)

        self._gyro_prev_raw_deg_s = raw
        self._last_gyro_filtered_deg_s = filtered

        return filtered

    def _sim_trajectory_setpoint(
        self,
        current_position_w: Sequence[float],
        target_position_w: Sequence[float],
    ):
        current = tuple(
            float(v)
            for v in current_position_w
        )

        target = tuple(
            float(v)
            for v in target_position_w
        )

        if not self._sim_traj_enabled:
            return (
                target,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            )

        target_changed = (
            self._sim_traj_target is None
            or max(
                abs(
                    target[i]
                    - self._sim_traj_target[i]
                )
                for i in range(3)
            ) > 1.0e-6
        )

        if target_changed:
            self._sim_traj_start = current
            self._sim_traj_target = target
            self._sim_traj_elapsed_s = 0.0

            delta = tuple(
                target[i] - current[i]
                for i in range(3)
            )

            distance = math.sqrt(
                sum(
                    v * v
                    for v in delta
                )
            )

            if distance < 1.0e-8:
                self._sim_traj_duration_s = 0.0
            else:
                # A quintic minimum-jerk trajectory has
                # max(ds/dt) = 1.875 / T.
                #
                # Choose T so that the translational speed does not
                # exceed B3_SIM_TRAJ_MAX_SPEED_MPS.
                duration_from_speed = (
                    1.875
                    * distance
                    / max(
                        self._sim_traj_max_speed_mps,
                        1.0e-6,
                    )
                )

                self._sim_traj_duration_s = max(
                    self._sim_traj_min_duration_s,
                    duration_from_speed,
                )

        if (
            self._sim_traj_duration_s <= 0.0
            or self._sim_traj_elapsed_s
            >= self._sim_traj_duration_s
        ):
            return (
                self._sim_traj_target,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            )

        T = self._sim_traj_duration_s

        tau = min(
            max(
                self._sim_traj_elapsed_s / T,
                0.0,
            ),
            1.0,
        )

        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau3 * tau
        tau5 = tau4 * tau

        # Quintic minimum jerk:
        # s(0)=0, s(1)=1
        # sdot(0)=sdot(1)=0
        # sddot(0)=sddot(1)=0
        s = (
            10.0 * tau3
            - 15.0 * tau4
            + 6.0 * tau5
        )

        s_dot = (
            30.0 * tau2
            - 60.0 * tau3
            + 30.0 * tau4
        ) / T

        s_ddot = (
            60.0 * tau
            - 180.0 * tau2
            + 120.0 * tau3
        ) / (T * T)

        delta = tuple(
            self._sim_traj_target[i]
            - self._sim_traj_start[i]
            for i in range(3)
        )

        position = tuple(
            self._sim_traj_start[i]
            + s * delta[i]
            for i in range(3)
        )

        velocity = tuple(
            s_dot * delta[i]
            for i in range(3)
        )

        acceleration = tuple(
            s_ddot * delta[i]
            for i in range(3)
        )

        # compute() is called at the 500-Hz Mellinger rate.
        self._sim_traj_elapsed_s += 1.0 / 500.0

        return (
            position,
            velocity,
            acceleration,
        )

    def _configure_position_setpoint_modes(self) -> None:
        sp = self.setpoint

        sp.mode.x = cf.modeAbs
        sp.mode.y = cf.modeAbs
        sp.mode.z = cf.modeAbs

        sp.mode.roll = cf.modeDisable
        sp.mode.pitch = cf.modeDisable
        sp.mode.yaw = cf.modeAbs
        sp.mode.quat = cf.modeDisable

        sp.velocity_body = False

        sp.attitudeRate.roll = 0.0
        sp.attitudeRate.pitch = 0.0
        sp.attitudeRate.yaw = 0.0

        sp.jerk.x = 0.0
        sp.jerk.y = 0.0
        sp.jerk.z = 0.0

    def reset(self) -> None:
        cf.controllerMellingerReset(self.controller)
        self.stabilizer_step = 0
        self._reset_gyro_lpf_state()

        self._sim_traj_start = None
        self._sim_traj_target = None
        self._sim_traj_elapsed_s = 0.0
        self._sim_traj_duration_s = 0.0

    def set_goal(
        self,
        position_w: Sequence[float],
        *,
        yaw_deg: float = 0.0,
        velocity_w: Sequence[float] = (0.0, 0.0, 0.0),
        acceleration_w: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        sp = self.setpoint

        _set_xyz(sp.position, position_w)
        _set_xyz(sp.velocity, velocity_w)
        _set_xyz(sp.acceleration, acceleration_w)

        sp.attitude.yaw = float(yaw_deg)

    def set_state(
        self,
        *,
        position_w: Sequence[float],
        velocity_w: Sequence[float],
        quat_wxyz: Sequence[float],
        body_rates_rad_s: Sequence[float],
        accel_g: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> None:
        state = self.state
        sensors = self.sensors

        _set_xyz(state.position, position_w)
        _set_xyz(state.velocity, velocity_w)

        w, x, y, z = [float(v) for v in quat_wxyz]

        # Crazyflie quaternion_t exposes x, y, z, w.
        state.attitudeQuaternion.x = x
        state.attitudeQuaternion.y = y
        state.attitudeQuaternion.z = z
        state.attitudeQuaternion.w = w

        roll_deg, pitch_deg_standard, yaw_deg = (
            _quat_wxyz_to_euler_deg(quat_wxyz)
        )

        # Firmware state_t.attitude uses the legacy Crazyflie convention:
        # pitch is inverted.
        state.attitude.roll = roll_deg
        state.attitude.pitch = -pitch_deg_standard
        state.attitude.yaw = yaw_deg

        wx, wy, wz = [float(v) for v in body_rates_rad_s]

        # sensorData_t.gyro is deg/s.
        #
        # controller_mellinger.c performs:
        #   roll  = +radians(gyro.x)
        #   pitch = -radians(gyro.y)
        #   yaw   = +radians(gyro.z)
        #
        # B3_GYRO_LPF_80HZ_V1:
        # The real BMI088 sensor path applies Crazyflie's 2-pole 80-Hz LPF
        # at 1 kHz before Mellinger consumes sensorData_t.gyro.
        gyro_raw_deg_s = (
            math.degrees(wx),
            math.degrees(wy),
            math.degrees(wz),
        )

        gyro_deg_s = self._filter_gyro_for_500hz_bridge(
            gyro_raw_deg_s
        )

        sensors.gyro.x = gyro_deg_s[0]
        sensors.gyro.y = gyro_deg_s[1]
        sensors.gyro.z = gyro_deg_s[2]

        _set_xyz(sensors.acc, accel_g)

    def step_1000hz(self) -> Dict[str, object]:
        """
        Advance one Crazyflie stabilizer tick.

        Firmware Mellinger itself executes at 500 Hz through RATE_DO_EXECUTE.
        """

        cf.controllerMellinger(
            self.controller,
            self.control,
            self.setpoint,
            self.sensors,
            self.state,
            int(self.stabilizer_step),
        )

        cf.powerDistribution(
            self.control,
            self.uncapped,
        )

        capped = cf.powerDistributionCap(
            self.uncapped,
            self.pwm,
        )

        result = {
            "stabilizer_step": int(self.stabilizer_step),
            "gyro_lpf_enabled": bool(self._gyro_lpf_enabled),
            "gyro_raw_deg_s": self._last_gyro_raw_deg_s,
            "gyro_filtered_deg_s": self._last_gyro_filtered_deg_s,
            "control_mode": int(self.control.controlMode),
            "thrust": float(self.control.thrust),
            "roll": int(self.control.roll),
            "pitch": int(self.control.pitch),
            "yaw": int(self.control.yaw),
            "motor_uncapped": (
                int(self.uncapped.motors.m1),
                int(self.uncapped.motors.m2),
                int(self.uncapped.motors.m3),
                int(self.uncapped.motors.m4),
            ),
            "motor_pwm": (
                int(self.pwm.motors.m1),
                int(self.pwm.motors.m2),
                int(self.pwm.motors.m3),
                int(self.pwm.motors.m4),
            ),
            "was_capped": bool(capped),
            "cmd_thrust": float(self.control.thrust),
            "z_axis_desired": (
                float(self.controller.z_axis_desired.x),
                float(self.controller.z_axis_desired.y),
                float(self.controller.z_axis_desired.z),
            ),
        }

        self.stabilizer_step += 1
        return result

    # FIRMWARE_500HZ_GC_BRIDGE_V1
    def step_500hz(self) -> Dict[str, object]:
        """
        Execute one real Mellinger update.

        The Crazyflie stabilizer counter is 1000 Hz, while Mellinger
        executes on even ticks at 500 Hz. The Isaac comparison loop
        itself is already 500 Hz, so each call advances the firmware
        stabilizer tick by two.
        """
        if self.stabilizer_step % 2 != 0:
            raise RuntimeError(
                f"Expected even stabilizer tick, got {self.stabilizer_step}"
            )

        result = self.step_1000hz()

        # step_1000hz incremented by one. Skip the held odd tick.
        self.stabilizer_step += 1

        return result

    @staticmethod
    def _world_to_body_vector(
        quat_wxyz: Sequence[float],
        vector_w: Sequence[float],
    ) -> tuple[float, float, float]:
        w, x, y, z = [float(v) for v in quat_wxyz]
        vx, vy, vz = [float(v) for v in vector_w]

        # Body -> world rotation.
        r00 = 1.0 - 2.0 * (y*y + z*z)
        r01 = 2.0 * (x*y - z*w)
        r02 = 2.0 * (x*z + y*w)

        r10 = 2.0 * (x*y + z*w)
        r11 = 1.0 - 2.0 * (x*x + z*z)
        r12 = 2.0 * (y*z - x*w)

        r20 = 2.0 * (x*z - y*w)
        r21 = 2.0 * (y*z + x*w)
        r22 = 1.0 - 2.0 * (x*x + y*y)

        # world -> body = R^T
        return (
            r00 * vx + r10 * vy + r20 * vz,
            r01 * vx + r11 * vy + r21 * vz,
            r02 * vx + r12 * vy + r22 * vz,
        )

    def get_action_from_gc(
        self,
        gc,
        *,
        device=None,
    ):
        """
        Convert the existing Isaac GC state into the actual Crazyflie
        firmware structs, run controllerMellinger + powerDistribution,
        then return an Isaac SRT action.

        GC layout:
          0:3   body position world [m]
          3:7   body quaternion WXYZ
          7:10  body linear velocity world [m/s]
          10:13 body angular velocity world [deg/s]
          13:16 COM position goal world [m]
          16    yaw goal [rad]
        """
        import torch

        if gc.ndim != 2 or gc.shape[0] != 1 or gc.shape[1] < 17:
            raise RuntimeError(
                f"Unexpected GC shape: {tuple(gc.shape)}"
            )

        row = (
            gc[0]
            .detach()
            .cpu()
            .to(torch.float64)
        )

        position_w = row[0:3].tolist()
        quat_wxyz = row[3:7].tolist()
        velocity_w = row[7:10].tolist()

        omega_w_rad_s = [
            math.radians(float(v))
            for v in row[10:13].tolist()
        ]

        omega_b_rad_s = self._world_to_body_vector(
            quat_wxyz,
            omega_w_rad_s,
        )

        goal_w = row[13:16].tolist()
        goal_yaw_deg = math.degrees(
            float(row[16].item())
        )

        (
            firmware_goal_w,
            firmware_goal_velocity_w,
            firmware_goal_acceleration_w,
        ) = self._sim_trajectory_setpoint(
            position_w,
            goal_w,
        )

        self.set_goal(
            firmware_goal_w,
            yaw_deg=goal_yaw_deg,
            velocity_w=firmware_goal_velocity_w,
            acceleration_w=firmware_goal_acceleration_w,
        )

        self.set_state(
            position_w=position_w,
            velocity_w=velocity_w,
            quat_wxyz=quat_wxyz,
            body_rates_rad_s=omega_b_rad_s,
        )

        fw = self.step_500hz()

        # Firmware native motor order:
        #   M1 (+x,-y)
        #   M2 (-x,-y)
        #   M3 (-x,+y)
        #   M4 (+x,+y)
        #
        # Isaac order:
        #   1 = M4
        #   2 = M1
        #   3 = M2
        #   4 = M3
        pwm_fw = fw["motor_pwm"]

        motor_fw = tuple(
            max(0.0, min(1.0, float(v) / 65536.0))
            for v in pwm_fw
        )

        motor_isaac = (
            motor_fw[3],
            motor_fw[0],
            motor_fw[1],
            motor_fw[2],
        )

        target_device = gc.device if device is None else device

        motor_tensor = torch.tensor(
            motor_isaac,
            dtype=gc.dtype,
            device=target_device,
        ).view(1, 4)

        # Existing SRT environment consumes [-1, +1].
        action = 2.0 * motor_tensor - 1.0

        fw["motor_normalized_firmware"] = motor_fw
        fw["motor_normalized_isaac"] = motor_isaac

        return action, fw

    def controller_parameters(self) -> Dict[str, float]:
        names = (
            "mass",
            "massThrust",
            "kp_xy",
            "kd_xy",
            "ki_xy",
            "i_range_xy",
            "kp_z",
            "kd_z",
            "ki_z",
            "i_range_z",
            "kR_xy",
            "kw_xy",
            "ki_m_xy",
            "i_range_m_xy",
            "kR_z",
            "kw_z",
            "ki_m_z",
            "i_range_m_z",
            "kd_omega_rp",
        )

        return {
            name: float(getattr(self.controller, name))
            for name in names
        }
