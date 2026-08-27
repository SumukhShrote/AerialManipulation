import math
import torch


class MellingerFirmwareController:
    """
    PyTorch port of the Crazyflie firmware Mellinger position/attitude controller.

    Input gc_obs layout:
      0:3   position world [m]
      3:7   quaternion wxyz
      7:10  linear velocity world [m/s]
      10:13 angular velocity world [deg/s]
      13:16 goal position world [m]
      16    desired yaw [rad]

    Output:
      SRT action in [-1, 1]^4, suitable for the existing environment's
      control_mode="SRT".

    This first implementation includes:
      - firmware Mellinger position control
      - firmware geometric attitude control
      - firmware integral terms
      - firmware roll/pitch angular-acceleration D term
      - firmware legacy quad mixer
      - firmware common-mode upper saturation
      - configurable idle thrust

    It intentionally does NOT yet include battery compensation.
    """

    UINT16_MAX = 65535.0

    def __init__(
        self,
        num_envs,
        device,
        dt,
        mass=0.046,
        mass_thrust=132000.0,
        idle_thrust=0.0,
        motor_command_scale=1.0,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dt = float(dt)

        # Real firmware runtime values
        self.mass = float(mass)
        self.mass_thrust = float(mass_thrust)

        self.kp_xy = 0.4
        self.kd_xy = 0.2
        self.ki_xy = 0.05
        self.i_range_xy = 2.0

        self.kp_z = 1.25
        self.kd_z = 0.4
        self.ki_z = 0.05
        self.i_range_z = 0.4

        self.kR_xy = 70000.0
        self.kw_xy = 20000.0
        self.ki_m_xy = 0.0
        self.i_range_m_xy = 1.0

        self.kR_z = 60000.0
        self.kw_z = 12000.0
        self.ki_m_z = 500.0
        self.i_range_m_z = 1500.0

        self.kd_omega_rp = 200.0

        self.idle_thrust = float(idle_thrust)
        self.motor_command_scale = float(motor_command_scale)

        self.i_pos = torch.zeros(num_envs, 3, device=self.device)
        self.i_att = torch.zeros(num_envs, 3, device=self.device)

        self.prev_omega_roll = torch.zeros(num_envs, device=self.device)
        self.prev_omega_pitch = torch.zeros(num_envs, device=self.device)
        self.derivative_initialized = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )

        self.last_control = None
        self.last_motor_raw = None
        self.last_motor_capped = None
        self.last_motor_normalized = None
        self.last_target_thrust = None
        self.last_eR = None

        # Diagnostic-only desired-attitude geometry.
        self.last_heading_cross_norm = None
        self.last_cos_phi_raw = None
        self.last_phi_deg = None
        self.last_x_c_des = None
        self.last_actual_yaw_deg = None
        self.last_desired_yaw_deg = None
        self.last_tilt_error_deg = None

    def reset(self, env_mask=None):
        if env_mask is None:
            env_mask = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        self.i_pos[env_mask] = 0.0
        self.i_att[env_mask] = 0.0
        self.prev_omega_roll[env_mask] = 0.0
        self.prev_omega_pitch[env_mask] = 0.0
        self.derivative_initialized[env_mask] = False

    @staticmethod
    def _normalize(v, eps=1e-8):
        return v / torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _quat_wxyz_to_rotmat(q):
        w, x, y, z = q.unbind(dim=1)

        R = torch.empty(
            (q.shape[0], 3, 3),
            dtype=q.dtype,
            device=q.device,
        )

        R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        R[:, 0, 1] = 2.0 * (x * y - w * z)
        R[:, 0, 2] = 2.0 * (x * z + w * y)

        R[:, 1, 0] = 2.0 * (x * y + w * z)
        R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        R[:, 1, 2] = 2.0 * (y * z - w * x)

        R[:, 2, 0] = 2.0 * (x * z - w * y)
        R[:, 2, 1] = 2.0 * (y * z + w * x)
        R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

        return R

    def get_action(self, gc_obs):
        pos_w = gc_obs[:, 0:3]
        quat_wxyz = gc_obs[:, 3:7]
        vel_w = gc_obs[:, 7:10]

        # Environment deliberately supplies this GC quantity in deg/s.
        omega_w_deg = gc_obs[:, 10:13]

        goal_pos_w = gc_obs[:, 13:16]
        desired_yaw = gc_obs[:, 16]

        R = self._quat_wxyz_to_rotmat(quat_wxyz)

        # Isaac gives world-frame angular velocity; firmware gyro is body-frame.
        omega_b_deg = torch.bmm(
            R.transpose(1, 2),
            omega_w_deg.unsqueeze(2),
        ).squeeze(2)

        # ------------------------------------------------------------
        # Position controller
        # ------------------------------------------------------------
        r_error = goal_pos_w - pos_w
        v_error = -vel_w

        self.i_pos[:, 0] += r_error[:, 0] * self.dt
        self.i_pos[:, 1] += r_error[:, 1] * self.dt
        self.i_pos[:, 2] += r_error[:, 2] * self.dt

        self.i_pos[:, 0:2].clamp_(
            -self.i_range_xy,
            self.i_range_xy,
        )
        self.i_pos[:, 2].clamp_(
            -self.i_range_z,
            self.i_range_z,
        )

        target_thrust = torch.zeros_like(pos_w)

        target_thrust[:, 0] = (
            self.kp_xy * r_error[:, 0]
            + self.kd_xy * v_error[:, 0]
            + self.ki_xy * self.i_pos[:, 0]
        )

        target_thrust[:, 1] = (
            self.kp_xy * r_error[:, 1]
            + self.kd_xy * v_error[:, 1]
            + self.ki_xy * self.i_pos[:, 1]
        )

        target_thrust[:, 2] = (
            self.mass * 9.81
            + self.kp_z * r_error[:, 2]
            + self.kd_z * v_error[:, 2]
            + self.ki_z * self.i_pos[:, 2]
        )

        z_axis = R[:, :, 2]

        # Firmware F = F_des dot z_B
        current_thrust = torch.sum(target_thrust * z_axis, dim=1)

        z_des = self._normalize(target_thrust)

        x_c_des = torch.stack(
            [
                torch.cos(desired_yaw),
                torch.sin(desired_yaw),
                torch.zeros_like(desired_yaw),
            ],
            dim=1,
        )

        heading_cross = torch.cross(
            z_des,
            x_c_des,
            dim=1,
        )

        heading_cross_norm = torch.linalg.norm(
            heading_cross,
            dim=1,
        )

        y_des = self._normalize(
            heading_cross
        )

        x_des = torch.cross(
            y_des,
            z_des,
            dim=1,
        )

        R_des = torch.stack(
            [x_des, y_des, z_des],
            dim=2,
        )

        # ------------------------------------------------------------
        # Diagnostic-only attitude geometry.
        #
        # Relative attitude:
        #   R_err = R_des^T R
        #
        # For a proper rotation matrix:
        #   cos(phi) = (trace(R_err) - 1) / 2
        #
        # Keep the raw value for auditing, then clamp only for acos.
        # Nothing here affects controller output.
        # ------------------------------------------------------------
        R_des_T_R_diag = torch.bmm(
            R_des.transpose(1, 2),
            R,
        )

        trace_R_des_T_R = (
            R_des_T_R_diag[:, 0, 0]
            + R_des_T_R_diag[:, 1, 1]
            + R_des_T_R_diag[:, 2, 2]
        )

        cos_phi_raw = 0.5 * (
            trace_R_des_T_R - 1.0
        )

        phi_deg = torch.acos(
            torch.clamp(
                cos_phi_raw,
                -1.0,
                1.0,
            )
        ) * (180.0 / math.pi)

        actual_yaw_deg = torch.atan2(
            R[:, 1, 0],
            R[:, 0, 0],
        ) * (180.0 / math.pi)

        desired_yaw_deg = (
            desired_yaw * (180.0 / math.pi)
        )

        # Diagnostic-only thrust-axis tilt error.
        # Unlike phi_deg, this excludes heading/yaw discrepancy.
        tilt_cos_raw = torch.sum(
            z_axis * z_des,
            dim=1,
        )

        tilt_error_deg = torch.acos(
            torch.clamp(
                tilt_cos_raw,
                -1.0,
                1.0,
            )
        ) * (180.0 / math.pi)

        # ------------------------------------------------------------
        # Firmware geometric attitude error
        #
        # Rdes^T R - R^T Rdes
        # ------------------------------------------------------------
        eRM = (
            torch.bmm(R_des.transpose(1, 2), R)
            - torch.bmm(R.transpose(1, 2), R_des)
        )

        eR = torch.stack(
            [
                eRM[:, 2, 1],
                -eRM[:, 0, 2],
                eRM[:, 1, 0],
            ],
            dim=1,
        )

        # IMPORTANT:
        # Do NOT apply the Crazyflie firmware's additional pitch-axis
        # attitude-error sign flip here.
        #
        # gc_obs already supplies the Isaac body orientation directly.
        # With the firmware motor mixer mapped into the Isaac rotor order,
        # positive control_pitch produces negative physical y torque.
        #
        # The geometric error expression above already has the corresponding
        # sign in the Isaac convention. Applying another eR_y sign inversion
        # makes pitch attitude feedback positive rather than restoring.

        # ------------------------------------------------------------
        # Body angular velocity convention used by firmware
        # ------------------------------------------------------------
        omega_b_rad = omega_b_deg * (math.pi / 180.0)

        state_rate_roll = omega_b_rad[:, 0]
        state_rate_pitch = -omega_b_rad[:, 1]
        state_rate_yaw = omega_b_rad[:, 2]

        # Desired attitude rates are zero for this position-hover test.
        ew = torch.stack(
            [
                -state_rate_roll,
                -state_rate_pitch,
                -state_rate_yaw,
            ],
            dim=1,
        )

        err_d_roll = torch.zeros_like(state_rate_roll)
        err_d_pitch = torch.zeros_like(state_rate_pitch)

        initialized = self.derivative_initialized

        err_d_roll[initialized] = -(
            state_rate_roll[initialized]
            - self.prev_omega_roll[initialized]
        ) / self.dt

        err_d_pitch[initialized] = -(
            state_rate_pitch[initialized]
            - self.prev_omega_pitch[initialized]
        ) / self.dt

        self.prev_omega_roll = state_rate_roll.clone()
        self.prev_omega_pitch = state_rate_pitch.clone()
        self.derivative_initialized[:] = True

        # ------------------------------------------------------------
        # Attitude integral
        # ------------------------------------------------------------
        self.i_att += (-eR) * self.dt

        self.i_att[:, 0:2].clamp_(
            -self.i_range_m_xy,
            self.i_range_m_xy,
        )

        self.i_att[:, 2].clamp_(
            -self.i_range_m_z,
            self.i_range_m_z,
        )

        # ------------------------------------------------------------
        # Firmware legacy moment commands
        # ------------------------------------------------------------
        Mx = (
            -self.kR_xy * eR[:, 0]
            + self.kw_xy * ew[:, 0]
            + self.ki_m_xy * self.i_att[:, 0]
            + self.kd_omega_rp * err_d_roll
        )

        My = (
            -self.kR_xy * eR[:, 1]
            + self.kw_xy * ew[:, 1]
            + self.ki_m_xy * self.i_att[:, 1]
            + self.kd_omega_rp * err_d_pitch
        )

        Mz = (
            -self.kR_z * eR[:, 2]
            + self.kw_z * ew[:, 2]
            + self.ki_m_z * self.i_att[:, 2]
        )

        control_thrust = self.mass_thrust * current_thrust

        control_roll = torch.clamp(Mx, -32000.0, 32000.0)
        control_pitch = torch.clamp(My, -32000.0, 32000.0)
        control_yaw = torch.clamp(-Mz, -32000.0, 32000.0)

        positive_thrust = control_thrust > 0.0

        control_roll = torch.where(
            positive_thrust, control_roll, torch.zeros_like(control_roll)
        )
        control_pitch = torch.where(
            positive_thrust, control_pitch, torch.zeros_like(control_pitch)
        )
        control_yaw = torch.where(
            positive_thrust, control_yaw, torch.zeros_like(control_yaw)
        )

        if torch.any(~positive_thrust):
            self.i_pos[~positive_thrust] = 0.0
            self.i_att[~positive_thrust] = 0.0

        # ------------------------------------------------------------
        # Exact legacy power distribution structure
        # ------------------------------------------------------------
        # C casts these to int16 after division.
        r = torch.trunc(control_roll / 2.0)
        p = torch.trunc(control_pitch / 2.0)

        motor_raw = torch.stack(
            [
                control_thrust - r + p + control_yaw,
                control_thrust - r - p - control_yaw,
                control_thrust + r - p + control_yaw,
                control_thrust + r + p - control_yaw,
            ],
            dim=1,
        )

        # Firmware upper saturation preserves motor differences by
        # subtracting a common reduction.
        highest = torch.max(motor_raw, dim=1).values
        reduction = torch.clamp(
            highest - self.UINT16_MAX,
            min=0.0,
        )

        motor_capped = motor_raw - reduction.unsqueeze(1)

        motor_capped = torch.clamp(
            motor_capped,
            min=self.idle_thrust,
            max=self.UINT16_MAX,
        )

        # Initial mapping under test:
        # Crazyflie motor ratio -> normalized motor coordinate.
        # motor_capped is in the native Crazyflie firmware motor order:
        #
        #   FW M1 = (+x, -y)
        #   FW M2 = (-x, -y)
        #   FW M3 = (-x, +y)
        #   FW M4 = (+x, +y)
        #
        # The Isaac plant uses:
        #
        #   Isaac 1 = (+x, +y) = FW M4
        #   Isaac 2 = (+x, -y) = FW M1
        #   Isaac 3 = (-x, -y) = FW M2
        #   Isaac 4 = (-x, +y) = FW M3
        #
        # Therefore the firmware motor vector must be permuted before
        # being sent through the Isaac SRT interface.
        motor_normalized_firmware = (
            motor_capped / 65536.0
        ) * self.motor_command_scale

        motor_normalized_firmware = torch.clamp(
            motor_normalized_firmware,
            0.0,
            1.0,
        )

        motor_normalized = motor_normalized_firmware[:, [3, 0, 1, 2]]

        # Existing SRT interface expects [-1, 1].
        actions = 2.0 * motor_normalized - 1.0

        self.last_control = torch.stack(
            [
                control_thrust,
                control_roll,
                control_pitch,
                control_yaw,
            ],
            dim=1,
        )

        self.last_motor_raw = motor_raw
        self.last_motor_capped = motor_capped
        self.last_motor_normalized_firmware = motor_normalized_firmware
        self.last_motor_normalized = motor_normalized
        self.last_target_thrust = target_thrust
        self.last_eR = eR

        self.last_heading_cross_norm = heading_cross_norm
        self.last_cos_phi_raw = cos_phi_raw
        self.last_phi_deg = phi_deg
        self.last_x_c_des = x_c_des
        self.last_actual_yaw_deg = actual_yaw_deg
        self.last_desired_yaw_deg = desired_yaw_deg
        self.last_tilt_error_deg = tilt_error_deg

        return actions
