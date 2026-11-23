from enum import IntEnum

import numpy as np
import torch

import engines.engine as engine
import envs.char_env as char_env
import envs.base_env as base_env
import anim.motion_lib as motion_lib
import util.torch_util as torch_util
import util.camera as camera
from envs.boxing_controller import BoxingReferenceController
from util.logger import Logger


class CombatState(IntEnum):
    IDLE = 0
    APPROACH = 1
    ATTACK = 2
    RECOVER = 3
    RETREAT = 4
    HIT_STUN = 5
    DOWN = 6
    BLOCK = 7
    SLIP = 8
    BACKSTEP = 9


class CombatEnv(char_env.CharEnv):
    """Two-character combat sandbox with independently timed demo controllers."""

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        self._opponent_start_pos = np.asarray(
            env_config.get("opponent_start_pos", [2.0, 0.0, 0.882416]),
            dtype=np.float32)
        self._opponent_start_rot = np.asarray(
            env_config.get("opponent_start_rot", [0.0, 0.0, 1.0, 0.0]),
            dtype=np.float32)
        self._ring_bounds = np.asarray(
            env_config.get("ring_bounds", [-1.0, 3.0, -2.0, 2.0]),
            dtype=np.float32)
        self._ring_floor_height = float(env_config.get("ring_floor_height", 0.36))
        self._ring_asset = env_config.get("ring_asset", "data/assets/objects/boxing_ring.xml")
        self._attack_body_names = env_config.get(
            "attack_bodies", ["right_hand", "left_hand", "right_foot", "left_foot"])
        self._target_body_names = env_config.get(
            "target_bodies", ["torso", "head", "pelvis"])
        if (self._attack_body_names[:2] != ["right_hand", "left_hand"] or
                not {"head", "torso", "pelvis"}.issubset(self._target_body_names)):
            raise ValueError("Boxing punches require both hands and head/torso/pelvis targets")
        self._hit_distance = float(env_config.get("hit_distance", 0.18))
        self._hit_cooldown = float(env_config.get("hit_cooldown", 0.25))
        self._hit_damage = float(env_config.get("hit_damage", 10.0))
        self._hit_reward = float(env_config.get("hit_reward", 1.0))
        self._max_health = float(env_config.get("max_health", 100.0))
        self._min_attack_speed = float(env_config.get("min_attack_speed", 0.5))
        self._min_approach_speed = float(env_config.get("min_approach_speed", 0.1))
        self._contact_force_threshold = float(
            env_config.get("contact_force_threshold", 1.0))
        self._punch_demo_enabled = bool(env_config.get("punch_demo", False))
        self._punch_motion_file = env_config.get(
            "punch_motion_file", "data/motions/humanoid/humanoid_punch.pkl")
        self._punch_demo_yaw = float(env_config.get("punch_demo_yaw", -0.638))
        self._punch_active_start = float(env_config.get("punch_active_start", 0.70))
        self._punch_active_end = float(env_config.get("punch_active_end", 1.30))
        self._opponent_attack_enabled = bool(
            env_config.get("opponent_attack", False))
        self._stagger_duration = float(env_config.get("stagger_duration", 0.35))
        self._stagger_speed = float(env_config.get("stagger_speed", 0.55))
        self._retreat_speed = float(env_config.get("retreat_speed", 0.30))
        self._recover_range = env_config.get("recover_duration", [1.0, 1.5])
        self._attack_cooldown_range = env_config.get("attack_cooldown", [1.5, 3.0])
        self._attack_range = float(env_config.get("attack_range", 1.15))
        self._approach_speed = float(env_config.get("approach_speed", 0.65))
        self._retreat_duration = float(env_config.get("retreat_duration", 0.5))
        self._block_duration = float(env_config.get("block_duration", 0.35))
        self._block_damage_scale = float(env_config.get("block_damage_scale", 0.35))
        self._boxing_reference = BoxingReferenceController(
            attack_range=self._attack_range,
            preferred_distance=float(env_config.get("preferred_distance", 1.0)),
            retreat_distance=float(env_config.get("retreat_distance", 0.72)),
            guard_height=float(env_config.get("guard_height", 1.05)))
        self._reference_control_enabled = bool(env_config.get("reference_control", False))
        self._motor_pose_enabled = bool(env_config.get("motor_pose", False))
        self._residual_scale = float(env_config.get("residual_scale", 0.25))
        self._opponent_ids = []

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device,
                         visualize=visualize, record_video=record_video)
        return

    def _build_env(self, env_id, env_config):
        super()._build_env(env_id, env_config)

        opponent_id = self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.articulated,
            asset_file=env_config["char_file"],
            name="opponent",
            start_pos=self._opponent_start_pos,
            start_rot=self._opponent_start_rot,
            color=self._get_opponent_color(),
        )

        if env_id == 0:
            self._opponent_ids.append(opponent_id)
        else:
            assert self._opponent_ids[0] == opponent_id
        self._build_ring(env_id)
        return

    def _build_ring(self, env_id):
        xmin, xmax, ymin, ymax = self._ring_bounds
        # Asset has a 4 m square fighting area, centered on its local origin.
        if not np.allclose([xmax - xmin, ymax - ymin], [4.0, 4.0]):
            raise ValueError("Boxing ring asset requires 4 x 4 metre ring_bounds")
        self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.rigid,
            asset_file=self._ring_asset,
            name="boxing_ring",
            fix_root=True,
            start_pos=np.asarray([0.5 * (xmin + xmax), 0.5 * (ymin + ymax),
                                  self._ring_floor_height], dtype=np.float32),
        )
        return

    def _build_camera(self, env_config):
        xmin, xmax, ymin, ymax = self._ring_bounds
        center = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax),
                           self._ring_floor_height + 0.8])
        self._camera = camera.Camera(
            mode=camera.CameraMode[env_config.get("camera_mode", "still")],
            engine=self._engine,
            pos=center + np.array([3.0, -4.8, 2.8]),
            target=center,
            track_obj_id=self._get_char_id())

    def _get_opponent_id(self):
        return self._opponent_ids[0]

    def _get_opponent_color(self):
        if self._engine.get_name() == "newton":
            return np.array([0.7, 0.25, 0.25])
        return np.array([0.9, 0.35, 0.35])

    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)

        opponent_body_names = env_config.get(
            "opponent_key_bodies", env_config.get("key_bodies", []))
        opponent_body_ids = []
        opponent_id = self._get_opponent_id()
        for body_name in opponent_body_names:
            body_id = self._engine.find_obj_body_id(opponent_id, body_name)
            assert body_id != -1
            opponent_body_ids.append(body_id)
        self._opponent_key_body_ids = torch.tensor(
            opponent_body_ids, device=self._device, dtype=torch.long)

        fighter_id = self._get_char_id()
        attack_body_ids = []
        for body_name in self._attack_body_names:
            body_id = self._engine.find_obj_body_id(fighter_id, body_name)
            assert body_id != -1
            attack_body_ids.append(body_id)
        target_body_ids = []
        opponent_id = self._get_opponent_id()
        for body_name in self._target_body_names:
            body_id = self._engine.find_obj_body_id(opponent_id, body_name)
            assert body_id != -1
            target_body_ids.append(body_id)

        self._attack_body_ids = torch.tensor(
            attack_body_ids, device=self._device, dtype=torch.long)
        self._target_body_ids = torch.tensor(
            target_body_ids, device=self._device, dtype=torch.long)

        def joint_dof(name):
            joint_id = next(i for i, joint in enumerate(self._kin_char_model._joints)
                            if joint is not None and joint.name == name)
            return self._kin_char_model.get_joint_dof_idx(joint_id)
        self._reference_joint_ids = (
            (joint_dof("right_shoulder"), joint_dof("right_shoulder") + 1,
             joint_dof("right_elbow")),
            (joint_dof("left_shoulder"), joint_dof("left_shoulder") + 1,
             joint_dof("left_elbow")))
        self._torso_dof = joint_dof("abdomen")
        self._leg_dofs = [(joint_dof(s + "_hip"), joint_dof(s + "_knee"),
                           joint_dof(s + "_ankle")) for s in ("right", "left")]
        self._foot_ids = self._build_body_ids_tensor(["right_foot", "left_foot"])

        opponent_attack_body_ids = []
        for body_name in self._attack_body_names:
            body_id = self._engine.find_obj_body_id(opponent_id, body_name)
            assert body_id != -1
            opponent_attack_body_ids.append(body_id)
        fighter_target_body_ids = []
        for body_name in self._target_body_names:
            body_id = self._engine.find_obj_body_id(fighter_id, body_name)
            assert body_id != -1
            fighter_target_body_ids.append(body_id)

        self._opponent_attack_body_ids = torch.tensor(
            opponent_attack_body_ids, device=self._device, dtype=torch.long)
        self._opponent_target_body_ids = torch.tensor(
            fighter_target_body_ids, device=self._device, dtype=torch.long)

        num_envs = self.get_num_envs()
        self._opponent_health = torch.full(
            [num_envs], self._max_health, device=self._device, dtype=torch.float)
        self._fighter_health = torch.full(
            [num_envs], self._max_health, device=self._device, dtype=torch.float)
        self._hit_cooldown_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._hit_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._attack_active_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._attack_hit_consumed = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._stagger_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._fighter_stagger_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._prev_attack_pos = torch.zeros(
            [num_envs, len(self._attack_body_ids), 3],
            device=self._device, dtype=torch.float)
        self._attack_speed_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._approach_speed_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._attack_contact_force_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._target_contact_force_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._knocked_down = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._opponent_hit_cooldown_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._opponent_hit_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._opponent_attack_active_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._opponent_punch_hit_consumed = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._opponent_prev_attack_pos = torch.zeros(
            [num_envs, len(self._opponent_attack_body_ids), 3],
            device=self._device, dtype=torch.float)
        self._opponent_attack_speed_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._opponent_approach_speed_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)

        self._combat_state = torch.zeros((num_envs, 2), device=self._device,
                                         dtype=torch.long)
        self._state_timer = torch.zeros((num_envs, 2), device=self._device)
        self._attack_cooldown = torch.zeros_like(self._state_timer)
        self._recover_timer = torch.zeros_like(self._state_timer)
        self._attack_variant = torch.zeros((num_envs, 2), device=self._device, dtype=torch.long)
        self._combo_followup_variant = torch.zeros_like(self._attack_variant)
        self._combo_followup_start = torch.zeros_like(self._state_timer)
        self._attack_length = torch.zeros_like(self._state_timer)
        self._combo = torch.zeros((num_envs, 2), device=self._device, dtype=torch.bool)
        self._combo_second = torch.zeros_like(self._combo)
        self._stamina = torch.ones_like(self._state_timer)
        self._spacing_target = torch.full_like(self._state_timer, 0.82)
        self._orbit_sign = torch.ones_like(self._state_timer)
        self._defense_count = torch.zeros_like(self._attack_variant)
        self._combat_pos = torch.zeros((num_envs, 2, 3), device=self._device)
        self._support_anchor = torch.zeros((num_envs, 2, 3), device=self._device)
        self._support_foot = torch.full((num_envs, 2), -1, device=self._device, dtype=torch.long)
        self._support_slip = torch.zeros((num_envs, 2), device=self._device)
        self._support_raw_slip = torch.zeros_like(self._support_slip)
        self._foot_canvas_contact = torch.zeros((num_envs, 2, 2), device=self._device, dtype=torch.bool)
        self._support_ik_error = torch.zeros_like(self._support_slip)
        self._motor_pose_error = torch.zeros_like(self._support_slip)
        self._motor_root_correction = torch.zeros_like(self._support_slip)

        if self._punch_demo_enabled:
            self._punch_motion_lib = motion_lib.MotionLib(
                motion_file=self._punch_motion_file,
                kin_char_model=self._kin_char_model,
                device=self._device)
            self._punch_motion_ids = torch.zeros(
                [num_envs], device=self._device, dtype=torch.long)
        if self._opponent_attack_enabled:
            self._opponent_punch_motion_lib = motion_lib.MotionLib(
                motion_file=self._punch_motion_file,
                kin_char_model=self._kin_char_model,
                device=self._device)
            self._opponent_punch_motion_ids = torch.zeros(
                [num_envs], device=self._device, dtype=torch.long)
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            self._reset_opponent(env_ids)
            self._combat_state[env_ids] = CombatState.IDLE
            self._state_timer[env_ids] = 0.0
            self._recover_timer[env_ids] = 0.0
            self._stamina[env_ids] = 1.0
            self._spacing_target[env_ids, 0] = 0.78
            self._spacing_target[env_ids, 1] = 0.90
            self._orbit_sign[env_ids, 0] = 1
            self._orbit_sign[env_ids, 1] = -1
            self._defense_count[env_ids] = 0
            self._combo_followup_variant[env_ids] = 0
            self._combo_followup_start[env_ids] = 0
            self._attack_length[env_ids] = 0
            self._combo[env_ids] = False
            self._combo_second[env_ids] = False
            self._attack_variant[env_ids, 0] = 2
            self._attack_variant[env_ids, 1] = 0
            self._attack_cooldown[env_ids, 0] = 0.35
            self._attack_cooldown[env_ids, 1] = 1.05
            self._combat_pos[env_ids, 0] = self._engine.get_root_pos(self._get_char_id())[env_ids]
            self._combat_pos[env_ids, 1] = self._engine.get_root_pos(self._get_opponent_id())[env_ids]
            self._support_foot[env_ids] = -1
            self._support_slip[env_ids] = 0
            self._support_raw_slip[env_ids] = 0
            self._foot_canvas_contact[env_ids] = False
            self._support_ik_error[env_ids] = 0
            self._motor_pose_error[env_ids] = 0
            self._motor_root_correction[env_ids] = 0
            if self._motor_pose_enabled:
                for obj_id in (self._get_char_id(), self._get_opponent_id()):
                    initial_target = self._engine.get_dof_pos(obj_id).clone()
                    initial_target[env_ids] = self._init_dof_pos
                    self._engine.set_cmd(obj_id, initial_target)
            self._set_character_color(self._get_char_id(), self._get_char_color(), env_ids)
            self._set_character_color(self._get_opponent_id(), self._get_opponent_color(), env_ids)
            self._opponent_health[env_ids] = self._max_health
            self._fighter_health[env_ids] = self._max_health
            self._hit_cooldown_buf[env_ids] = 0.0
            self._hit_buf[env_ids] = False
            self._attack_active_buf[env_ids] = False
            self._attack_hit_consumed[env_ids] = False
            self._stagger_buf[env_ids] = 0.0
            self._fighter_stagger_buf[env_ids] = 0.0
            self._attack_speed_buf[env_ids] = 0.0
            self._approach_speed_buf[env_ids] = 0.0
            self._attack_contact_force_buf[env_ids] = 0.0
            self._target_contact_force_buf[env_ids] = 0.0
            self._knocked_down[env_ids] = False
            fighter_body_pos = self._engine.get_body_pos(self._get_char_id())
            self._prev_attack_pos[env_ids] = fighter_body_pos[
                env_ids][:, self._attack_body_ids, :]
            opponent_body_pos = self._engine.get_body_pos(self._get_opponent_id())
            self._opponent_prev_attack_pos[env_ids] = opponent_body_pos[
                env_ids][:, self._opponent_attack_body_ids, :]
            self._opponent_hit_cooldown_buf[env_ids] = 0.0
            self._opponent_hit_buf[env_ids] = False
            self._opponent_attack_active_buf[env_ids] = False
            self._opponent_punch_hit_consumed[env_ids] = False
            self._opponent_attack_speed_buf[env_ids] = 0.0
            self._opponent_approach_speed_buf[env_ids] = 0.0
        return

    def _apply_action(self, actions):
        if not self._reference_control_enabled:
            return super()._apply_action(actions)
        if self._punch_demo_enabled:
            if self._motor_pose_enabled:
                # Targets from the previous post-physics update remain active.
                return
            # The demo writes kinematic poses after simulation. Keep motors at
            # the same pose instead of adding a residual that is then erased.
            self._engine.set_cmd(self._get_char_id(),
                                 self._engine.get_dof_pos(self._get_char_id()).clone())
            return
        intent, _, _, _ = self._boxing_reference.compute(
            self._engine.get_root_pos(self._get_char_id()),
            self._engine.get_root_pos(self._get_opponent_id()),
            self._combat_state[:, 0], self._fighter_health, self._opponent_health)
        base = self._init_dof_pos.expand(actions.shape[0], -1)
        reference = self._boxing_reference.reference_action(
            base, intent, 0, self._reference_joint_ids[0])
        residual = actions - base
        target = reference + self._residual_scale * residual
        target = torch.minimum(torch.maximum(target, self._action_bound_low), self._action_bound_high)
        self._engine.set_cmd(self._get_char_id(), target)
        return

    def _update_misc(self):
        super()._update_misc()
        was_staggered = self._stagger_buf > 0.0
        self._stagger_buf[:] = torch.clamp_min(
            self._stagger_buf - self._engine.get_timestep(), 0.0)
        was_fighter_staggered = self._fighter_stagger_buf > 0.0
        self._fighter_stagger_buf[:] = torch.clamp_min(
            self._fighter_stagger_buf - self._engine.get_timestep(), 0.0)
        if torch.any(torch.logical_and(was_staggered, self._stagger_buf <= 0.0)):
            self._set_character_color(self._get_opponent_id(),
                                      self._get_opponent_color(),
                                      ((was_staggered & (self._stagger_buf <= 0)) &
                                       (self._opponent_health > 0)).nonzero().flatten())
        if torch.any(torch.logical_and(
                was_fighter_staggered, self._fighter_stagger_buf <= 0.0)):
            self._set_character_color(self._get_char_id(),
                                      self._get_char_color(),
                                      ((was_fighter_staggered & (self._fighter_stagger_buf <= 0)) &
                                       (self._fighter_health > 0)).nonzero().flatten())
        self._update_combat_states()
        if self._punch_demo_enabled:
            self._update_punch_demo()
        if self._opponent_attack_enabled:
            self._update_opponent_punch()
        for side, obj_id in enumerate((self._get_char_id(), self._get_opponent_id())):
            forces = self._engine.get_contact_forces(obj_id)[:, self._foot_ids]
            foot_height = self._engine.get_body_pos(obj_id)[:, self._foot_ids, 2]
            on_canvas = (foot_height - self._ring_floor_height).abs() < 0.12
            self._foot_canvas_contact[:, side] = (
                torch.linalg.vector_norm(forces, dim=-1) > 1.0) & on_canvas
        self._enforce_ring_bounds()
        self._update_hits()
        self._update_opponent_hits()
        self._enforce_ring_bounds()
        return

    def _enforce_ring_bounds(self):
        xmin, xmax, ymin, ymax = self._ring_bounds
        for obj_id in [self._get_char_id(), self._get_opponent_id()]:
            root_pos = self._engine.get_root_pos(obj_id)
            clamped_pos = root_pos.clone()
            clamped_pos[:, 0] = torch.clamp(clamped_pos[:, 0], xmin, xmax)
            clamped_pos[:, 1] = torch.clamp(clamped_pos[:, 1], ymin, ymax)
            changed = ((clamped_pos[:, 0] <= xmin + 1e-6) |
                       (clamped_pos[:, 0] >= xmax - 1e-6) |
                       (clamped_pos[:, 1] <= ymin + 1e-6) |
                       (clamped_pos[:, 1] >= ymax - 1e-6))
            if not torch.any(changed):
                continue

            root_vel = self._engine.get_root_vel(obj_id)[changed].clone()
            root_vel[:, 0] = torch.where(
                ((clamped_pos[changed, 0] <= xmin + 1e-6) & (root_vel[:, 0] < 0.0)) |
                ((clamped_pos[changed, 0] >= xmax - 1e-6) & (root_vel[:, 0] > 0.0)),
                torch.zeros_like(root_vel[:, 0]), root_vel[:, 0])
            root_vel[:, 1] = torch.where(
                ((clamped_pos[changed, 1] <= ymin + 1e-6) & (root_vel[:, 1] < 0.0)) |
                ((clamped_pos[changed, 1] >= ymax - 1e-6) & (root_vel[:, 1] > 0.0)),
                torch.zeros_like(root_vel[:, 1]), root_vel[:, 1])
            changed_ids = changed.nonzero(as_tuple=False).flatten()
            dof_pos = self._engine.get_dof_pos(obj_id)[changed]
            root_rot = self._engine.get_root_rot(obj_id)[changed]
            joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
            body_pos, body_rot = self._kin_char_model.forward_kinematics(
                clamped_pos[changed], root_rot, joint_rot)
            self._engine.set_root_pos(changed_ids, obj_id, clamped_pos[changed])
            self._engine.set_root_vel(changed_ids, obj_id, root_vel)
            self._engine.set_body_pos(changed_ids, obj_id, body_pos)
            self._engine.set_body_rot(changed_ids, obj_id, body_rot)
        return

    def _enter_state(self, side, mask, state):
        self._combat_state[mask, side] = state
        self._state_timer[mask, side] = 0.0

    def _sample_duration(self, bounds, count):
        return torch.empty(count, device=self._device).uniform_(*bounds)

    def _update_combat_states(self):
        dt = self._engine.get_timestep()
        self._state_timer += dt
        self._attack_cooldown.sub_(dt).clamp_min_(0.0)
        self._recover_timer.sub_(dt).clamp_min_(0.0)
        resting = self._combat_state != CombatState.ATTACK
        self._stamina.add_(resting * (0.12 * dt)).clamp_(0, 1)
        distance = torch.linalg.vector_norm(
            self._combat_pos[:, 1, :2] - self._combat_pos[:, 0, :2], dim=-1)
        finished = (self._fighter_health <= 0) | (self._opponent_health <= 0)
        for side, enabled in enumerate((self._punch_demo_enabled,
                                        self._opponent_attack_enabled)):
            state = self._combat_state[:, side].clone()
            health = self._fighter_health if side == 0 else self._opponent_health
            self._enter_state(side, finished & (health <= 0), CombatState.DOWN)
            self._enter_state(side, finished & (health > 0), CombatState.IDLE)
            if not enabled:
                continue
            live = ~finished
            stun = (state == CombatState.HIT_STUN) & live
            self._enter_state(side, stun & (self._state_timer[:, side] >=
                                           self._stagger_duration), CombatState.RETREAT)
            retreat = (state == CombatState.RETREAT) & live
            self._enter_state(side, retreat & (self._state_timer[:, side] >=
                                              self._retreat_duration), CombatState.IDLE)
            blocking = (state == CombatState.BLOCK) & live
            self._enter_state(side, blocking & (self._state_timer[:, side] >=
                                               self._block_duration), CombatState.IDLE)
            evading = ((state == CombatState.SLIP) | (state == CombatState.BACKSTEP)) & live
            self._enter_state(side, evading & (self._state_timer[:, side] >= 0.6), CombatState.RECOVER)
            self._recover_timer[evading & (self._state_timer[:, side] == 0), side] = 0.3
            ended = (state == CombatState.ATTACK) & live & (
                self._state_timer[:, side] >= self._attack_length[:, side])
            self._enter_state(side, ended, CombatState.RECOVER)
            finished_variant = torch.where(self._combo_second[:, side],
                                           self._combo_followup_variant[:, side],
                                           self._attack_variant[:, side])
            self._recover_timer[ended, side] = self._sample_duration(
                self._recover_range, int(ended.sum())) + self._boxing_reference.punch_recovery_adjustment(
                    finished_variant[ended])
            recovering = (state == CombatState.RECOVER) & live
            self._enter_state(side, recovering & (self._recover_timer[:, side] <= 0),
                              CombatState.IDLE)
            deciding = ((state == CombatState.IDLE) |
                        (state == CombatState.APPROACH)) & live
            approach_limit = torch.where(
                (self._attack_cooldown[:, side] <= 0) & (self._stamina[:, side] >= 0.25),
                self._attack_range, self._spacing_target[:, side] + 0.12)
            far = distance > approach_limit
            self._enter_state(side, deciding & far, CombatState.APPROACH)
            ready = deciding & ~far & (self._attack_cooldown[:, side] <= 0)
            ready &= self._stamina[:, side] >= 0.25
            # Sequential reservation avoids both controllers starting in one tick.
            ready &= self._combat_state[:, 1 - side] != CombatState.ATTACK
            threat = deciding & ~far & ~ready & (self._combat_state[:, 1 - side] == CombatState.ATTACK)
            self._enter_state(side, threat, CombatState.BLOCK)
            choice = self._defense_count[:, side] % 3
            self._enter_state(side, threat & (choice == 1), CombatState.SLIP)
            self._enter_state(side, threat & (choice == 2), CombatState.BACKSTEP)
            self._defense_count[threat, side] += 1
            self._enter_state(side, deciding & ~far & ~ready & ~threat, CombatState.IDLE)
            self._enter_state(side, ready, CombatState.ATTACK)
            count = int(ready.sum())
            previous = self._attack_variant[ready, side]
            punch_count = len(self._boxing_reference.PUNCH_NAMES) if self._reference_control_enabled else 3
            chosen = (previous + torch.randint(1, punch_count, (count,), device=self._device)) % punch_count
            followup = (chosen + torch.randint(1, punch_count, (count,), device=self._device)) % punch_count
            self._attack_variant[ready, side] = chosen
            self._combo_followup_variant[ready, side] = followup
            self._spacing_target[ready, side] = self._sample_duration((0.72, 0.96), count)
            self._orbit_sign[ready, side] = torch.where(
                torch.rand(count, device=self._device) < 0.5, -1.0, 1.0)
            self._combo[ready, side] = self._reference_control_enabled & (
                torch.rand(count, device=self._device) < 0.5) & (
                self._stamina[ready, side] >= 0.5)
            self._combo_second[ready, side] = False
            first_duration = self._boxing_reference.punch_duration(chosen)
            second_duration = self._boxing_reference.punch_duration(followup)
            followup_start = first_duration + 0.16
            self._combo_followup_start[ready, side] = followup_start
            self._attack_length[ready, side] = torch.where(
                self._combo[ready, side], followup_start + second_duration + 0.08,
                first_duration + 0.08)
            first_cost = self._boxing_reference.punch_stamina_cost(chosen)
            second_cost = self._boxing_reference.punch_stamina_cost(followup)
            self._stamina[ready, side] -= first_cost + torch.where(
                self._combo[ready, side], second_cost, 0.0)
            self._attack_cooldown[ready, side] = self._attack_length[ready, side] + self._sample_duration(
                self._attack_cooldown_range, count)
            consumed = self._attack_hit_consumed if side == 0 else self._opponent_punch_hit_consumed
            consumed[ready] = False

    def _update_punch_demo(self):
        self._pose_combatant(0, self._get_char_id(), self._punch_motion_lib,
                             self._punch_motion_ids, self._attack_active_buf)

    def _update_opponent_punch(self):
        self._pose_combatant(1, self._get_opponent_id(), self._opponent_punch_motion_lib,
                             self._opponent_punch_motion_ids, self._opponent_attack_active_buf)

    def _pose_combatant(self, side, obj_id, lib, motion_ids, active):
        state = self._combat_state[:, side]
        attacking = state == CombatState.ATTACK
        variant = self._attack_variant[:, side]
        local_time = self._state_timer[:, side].clone()
        second = attacking & self._combo[:, side] & (
            local_time >= self._combo_followup_start[:, side])
        new_second = second & ~self._combo_second[:, side]
        consumed = self._attack_hit_consumed if side == 0 else self._opponent_punch_hit_consumed
        consumed[new_second] = False
        self._combo_second[:, side] |= second
        variant = torch.where(second, self._combo_followup_variant[:, side], variant)
        local_time = torch.where(second, local_time - self._combo_followup_start[:, side], local_time)
        punch_time = local_time * torch.where(
            variant == 0, torch.full_like(variant, 0.78, dtype=torch.float32),
            torch.ones_like(variant, dtype=torch.float32))
        time = torch.where(attacking, punch_time, 0.0)
        active_start = torch.where(variant == 0, 0.45, self._punch_active_start)
        active_end = torch.where(variant == 0, 0.95, self._punch_active_end)
        active[:] = attacking & (time >= active_start) & (time <= active_end)
        pos = self._combat_pos[:, side].clone()
        if self._motor_pose_enabled:
            pos[:, 2] = self._engine.get_root_pos(obj_id)[:, 2]
        delta = self._combat_pos[:, 1 - side, :2] - pos[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
        direction = delta / distance.unsqueeze(-1)
        speed = torch.zeros_like(distance)
        speed[state == CombatState.APPROACH] = self._approach_speed
        speed = torch.minimum(speed, (distance - self._attack_range).clamp_min(0) /
                              self._engine.get_timestep())
        speed[state == CombatState.RETREAT] = -self._retreat_speed
        speed[state == CombatState.BACKSTEP] = -self._approach_speed
        speed[state == CombatState.HIT_STUN] = -self._stagger_speed
        footwork = ((state == CombatState.IDLE) | (state == CombatState.RECOVER)) & (
            self._attack_cooldown[:, side] > 0) & (self._fighter_health > 0) & (
            self._opponent_health > 0)
        spacing_error = distance - self._spacing_target[:, side]
        spacing_speed = spacing_error.sign() * (spacing_error.abs() - 0.08).clamp_min(0)
        speed = torch.where(footwork, (1.3 * spacing_speed).clamp(-0.28, 0.28), speed)
        mobile = ((state == CombatState.APPROACH) | (state == CombatState.RETREAT) |
                  (state == CombatState.BACKSTEP) | footwork)
        step_phase = torch.remainder(self._time_buf / 0.65 + side * 0.5, 1.0)
        # Ease into and out of each traveling step; pause at footfall.
        rise = ((step_phase - 0.15) / 0.10).clamp(0, 1)
        fall = ((0.65 - step_phase) / 0.10).clamp(0, 1)
        step_drive = 2.5 * torch.minimum(rise, fall)
        speed = torch.where(mobile, speed * step_drive, speed)
        vel = torch.zeros_like(pos)
        vel[:, :2] = direction * speed.unsqueeze(-1)
        tangent = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
        vel[:, :2] += tangent * (0.25 * self._orbit_sign[:, side] *
                                step_drive * footwork)[:, None]
        xmin, xmax, ymin, ymax = self._ring_bounds
        center = pos.new_tensor([(xmin + xmax) / 2, (ymin + ymax) / 2])
        near_edge = ((pos[:, 0] < xmin + 0.25) | (pos[:, 0] > xmax - 0.25) |
                     (pos[:, 1] < ymin + 0.25) | (pos[:, 1] > ymax - 0.25))
        inward = center - pos[:, :2]
        inward /= torch.linalg.vector_norm(inward, dim=-1).clamp_min(1e-6).unsqueeze(-1)
        vel[near_edge & mobile, :2] = (inward * (self._approach_speed * step_drive)[:, None])[near_edge & mobile]
        pos += vel * self._engine.get_timestep()
        pos[:, 0].clamp_(xmin, xmax)
        pos[:, 1].clamp_(ymin, ymax)
        self._combat_pos[:, side] = pos
        reference_pos = pos.clone()
        # Clip yaw calibration aligns recorded punch direction with opponent bearing.
        yaw = torch.atan2(delta[:, 1], delta[:, 0]) + self._punch_demo_yaw
        axis = torch.zeros_like(pos)
        axis[:, 2] = 1
        quat = torch_util.axis_angle_to_quat(axis, yaw)
        clip_pos, root_rot, clip_vel, root_ang_vel, joint_rot, dof_vel = lib.calc_motion_frame(motion_ids, time)
        first_pos = lib.calc_motion_frame(motion_ids, torch.zeros_like(time))[0]
        displacement = torch_util.quat_rotate(quat, clip_pos - first_pos)
        pos += displacement
        pos[:, 0].clamp_(xmin, xmax)
        pos[:, 1].clamp_(ymin, ymax)
        vel += torch.where(attacking.unsqueeze(-1),
                           torch_util.quat_rotate(quat, clip_vel), 0.0)
        root_rot = torch_util.quat_mul(quat, root_rot)
        root_ang_vel = torch_util.quat_rotate(quat, root_ang_vel)
        root_ang_vel[~attacking] = 0
        dof_vel[~attacking] = 0
        dof_pos = lib.joint_rot_to_dof(joint_rot)
        if self._reference_control_enabled:
            dof_pos, active[:] = self._boxing_reference.boxing_pose(
                self._init_dof_pos.expand_as(dof_pos), self._reference_joint_ids,
                variant, local_time, attacking)
            # Reference animation owns the pose; do not mix recorded root motion
            # with a different arm trajectory or pretend motor residuals survive.
            pos = reference_pos
            duration = self._boxing_reference.punch_duration(variant)
            phase = (local_time / duration).clamp(0, 1)
            turn = torch.sin(torch.pi * phase).square() * attacking
            torso_turn = torch.where(variant == 1, 0.18,
                                     torch.where(variant == 2, -0.16,
                                                 torch.where(variant == 3, 0.22,
                                                             torch.where(variant == 4, 0.12, -0.07))))
            dof_pos[:, self._torso_dof + 2] = torso_turn * turn
            slip = torch.sin(torch.pi * (self._state_timer[:, side] / 0.6).clamp(0, 1))
            dof_pos[:, self._torso_dof] = 0.3 * slip * (state == CombatState.SLIP)
            stun_phase = (self._state_timer[:, side] / self._stagger_duration).clamp(0, 1)
            recoil = torch.sin(torch.pi * stun_phase) * (state == CombatState.HIT_STUN)
            block_phase = (self._state_timer[:, side] / self._block_duration).clamp(0, 1)
            brace = torch.sin(torch.pi * block_phase) * (state == CombatState.BLOCK)
            dof_pos[:, self._torso_dof + 1] = (
                0.13 * turn * ((variant == 2) | (variant == 3)) +
                0.08 * turn * (variant == 5) - 0.22 * recoil + 0.10 * brace)
            for shoulder, _, elbow in self._reference_joint_ids:
                dof_pos[:, shoulder + 1] -= 0.10 * brace
                dof_pos[:, elbow] += torch.sign(dof_pos[:, elbow]) * 0.16 * brace
            # Aim each swing along travel direction. A lateral shuffle should
            # move the hips sideways, not replay a forward walking stride.
            gait = torch.sin(2 * torch.pi * step_phase) * mobile
            forward_speed = torch.sum(vel[:, :2] * direction, dim=-1)
            lateral_speed = torch.sum(vel[:, :2] * tangent, dim=-1)
            travel_speed = forward_speed.abs() + lateral_speed.abs() + 1e-6
            forward_weight = forward_speed / travel_speed
            lateral_weight = lateral_speed / travel_speed
            for leg, (hip, knee, ankle) in enumerate(self._leg_dofs):
                swing = gait * (1 if leg == 0 else -1)
                dof_pos[:, hip] = self._init_dof_pos[hip] + 0.18 * swing * lateral_weight
                dof_pos[:, hip + 1] = self._init_dof_pos[hip + 1] - 0.18 * swing * forward_weight
                dof_pos[:, knee] = self._init_dof_pos[knee] + 0.24 * swing.clamp_min(0)
                dof_pos[:, ankle + 1] = (self._init_dof_pos[ankle + 1] -
                                          0.12 * swing.clamp_min(0) * forward_weight.abs())
                if leg == 0:
                    rear_pivot = 0.10 * turn * (variant == 3)
                    dof_pos[:, hip + 2] = self._init_dof_pos[hip + 2] + rear_pivot
                    dof_pos[:, ankle + 2] = self._init_dof_pos[ankle + 2] - rear_pivot
            root_rot = torch_util.axis_angle_to_quat(axis, torch.atan2(delta[:, 1], delta[:, 0]))
            root_ang_vel = torch.zeros_like(root_ang_vel)
            dof_vel = (dof_pos - self._engine.get_dof_pos(obj_id)) / self._engine.get_timestep()
            joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        alive_ids = (state != CombatState.DOWN).nonzero(as_tuple=False).flatten()
        body_pos, body_rot = self._kin_char_model.forward_kinematics(pos, root_rot, joint_rot)
        if self._reference_control_enabled:
            # Hold the support foot during footfall. Release while the next
            # step advances, so the anchor cannot cancel forward travel.
            planted = mobile & (step_drive == 0)
            support = torch.where(step_phase < 0.5, 0, 1)
            rows = torch.arange(self.get_num_envs(), device=self._device)
            feet = body_pos[:, self._foot_ids, :]
            selected = feet[rows, support]
            new_contact = planted & (self._support_foot[:, side] != support)
            if self._motor_pose_enabled:
                actual_dof = self._engine.get_dof_pos(obj_id).clone()
                actual_body_pos, _ = self._kin_char_model.forward_kinematics(
                    pos, root_rot, self._kin_char_model.dof_to_rot(actual_dof))
                contact_foot = actual_body_pos[rows, self._foot_ids[support]]
            else:
                contact_foot = selected
            self._support_anchor[new_contact, side] = contact_foot[new_contact]
            self._support_anchor[new_contact, side, 2] = self._ring_floor_height + 0.045
            # Three-dimensional support-leg IK: hip roll/pitch and knee.
            # Root correction below handles motor lag left after the solve.
            active_ik = planted
            self._support_ik_error[:, side] = 0
            if torch.any(active_ik):
                for _ in range(2):
                    columns = []
                    for joint_offset in (0, 1, 2):
                        trial = dof_pos.clone()
                        for foot_index, (hip, knee, _) in enumerate(self._leg_dofs):
                            target_joint = (hip, hip + 1, knee)[joint_offset]
                            trial[support == foot_index, target_joint] += 0.01
                        trial_pos, _ = self._kin_char_model.forward_kinematics(
                            pos, root_rot, self._kin_char_model.dof_to_rot(trial))
                        columns.append((trial_pos[rows, self._foot_ids[support]]
                                        - selected) / 0.01)
                    jac = torch.stack(columns, dim=-1)
                    error = self._support_anchor[:, side] - selected
                    jt = jac.transpose(1, 2)
                    normal = jt @ jac + 0.01 * torch.eye(3, device=self._device)
                    update = torch.linalg.solve(normal, (jt @ error.unsqueeze(-1))).squeeze(-1)
                    update = update.clamp(-0.12, 0.12) * active_ik[:, None]
                    for foot_index, (hip, knee, _) in enumerate(self._leg_dofs):
                        mask = support == foot_index
                        dof_pos[mask, hip] = (dof_pos[mask, hip] + update[mask, 0]).clamp(-0.5, 0.5)
                        dof_pos[mask, hip + 1] += update[mask, 1]
                        dof_pos[mask, knee] = (dof_pos[mask, knee] + update[mask, 2]).clamp(0, 2.79)
                    joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
                    body_pos, body_rot = self._kin_char_model.forward_kinematics(pos, root_rot, joint_rot)
                    selected = body_pos[rows, self._foot_ids[support]]
                self._support_ik_error[:, side] = torch.where(
                    active_ik, torch.linalg.vector_norm(
                        selected - self._support_anchor[:, side], dim=-1), 0)
            self._support_raw_slip[:, side] = torch.where(
                planted, torch.linalg.vector_norm(
                    selected[:, :2] - self._support_anchor[:, side, :2], dim=-1), 0.0)
            correction = (self._support_anchor[:, side, :2] - selected[:, :2]).clamp(-0.025, 0.025)
            correction *= (planted & (not self._motor_pose_enabled)).unsqueeze(-1)
            pos[:, :2] += correction
            pos[:, 0].clamp_(xmin, xmax)
            pos[:, 1].clamp_(ymin, ymax)
            correction = pos[:, :2] - body_pos[:, 0, :2]
            body_pos[:, :, :2] += correction[:, None, :]
            self._combat_pos[:, side] = pos
            self._support_slip[:, side] = torch.where(
                planted, torch.linalg.vector_norm(
                    body_pos[rows, self._foot_ids[support], :2] -
                    self._support_anchor[:, side, :2], dim=-1), 0.0)
            self._support_foot[:, side] = torch.where(
                planted, support, torch.full_like(support, -1))
        if self._motor_pose_enabled:
            self._engine.set_cmd(obj_id, dof_pos)
            actual_dof = self._engine.get_dof_pos(obj_id).clone()
            self._motor_pose_error[:, side] = torch.amax(
                (actual_dof - dof_pos).abs(), dim=-1)
            joint_rot = self._kin_char_model.dof_to_rot(actual_dof)
            body_pos, body_rot = self._kin_char_model.forward_kinematics(
                pos, root_rot, joint_rot)
            if self._reference_control_enabled:
                actual_foot = body_pos[rows, self._foot_ids[support], :2]
                foot_correction = (self._support_anchor[:, side, :2] - actual_foot)
                foot_correction = foot_correction.clamp(-0.025, 0.025) * planted[:, None]
                pos[:, :2] += foot_correction
                pos[:, 0].clamp_(xmin, xmax)
                pos[:, 1].clamp_(ymin, ymax)
                body_pos, body_rot = self._kin_char_model.forward_kinematics(
                    pos, root_rot, joint_rot)
                actual_foot = body_pos[rows, self._foot_ids[support], :2]
                self._support_slip[:, side] = torch.where(
                    planted, torch.linalg.vector_norm(
                        actual_foot - self._support_anchor[:, side, :2], dim=-1), 0.0)
            physical_root = self._engine.get_root_pos(obj_id)
            self._motor_root_correction[:, side] = torch.linalg.vector_norm(
                pos - physical_root, dim=-1)
        setters = [(self._engine.set_root_pos, pos),
                   (self._engine.set_root_rot, root_rot),
                   (self._engine.set_root_vel, vel),
                   (self._engine.set_root_ang_vel, root_ang_vel)]
        if not self._motor_pose_enabled:
            setters.extend(((self._engine.set_dof_pos, dof_pos),
                            (self._engine.set_dof_vel, dof_vel)))
        setters.extend(((self._engine.set_body_pos, body_pos),
                        (self._engine.set_body_rot, body_rot)))
        for setter, value in setters:
            setter(alive_ids, obj_id, value[alive_ids])
        down_ids = (state == CombatState.DOWN).nonzero(as_tuple=False).flatten()
        if len(down_ids):
            self._apply_knockdown(down_ids, obj_id)

    def _receive_hit(self, env_ids, side):
        self._enter_state(side, torch.isin(torch.arange(self.get_num_envs(), device=self._device),
                                         env_ids), CombatState.HIT_STUN)
        self._attack_cooldown[env_ids, side] = self._stagger_duration + self._retreat_duration + 1.0
        active = self._attack_active_buf if side == 0 else self._opponent_attack_active_buf
        active[env_ids] = False
        health = self._fighter_health if side == 0 else self._opponent_health
        down_ids = env_ids[health[env_ids] <= 0]
        if len(down_ids):
            self._combat_state[down_ids, side] = CombatState.DOWN
            self._combat_state[down_ids, 1 - side] = CombatState.IDLE
            self._attack_active_buf[down_ids] = False
            self._opponent_attack_active_buf[down_ids] = False
            self._apply_knockdown(down_ids, self._get_char_id() if side == 0 else self._get_opponent_id())

    def _detect_punch_hit(self, side):
        """Shared geometry and impact gate for either attacking fighter."""
        dt = self._engine.get_timestep()
        attacker = self._get_char_id() if side == 0 else self._get_opponent_id()
        defender = self._get_opponent_id() if side == 0 else self._get_char_id()
        hand_ids = self._attack_body_ids if side == 0 else self._opponent_attack_body_ids
        target_ids = self._target_body_ids if side == 0 else self._opponent_target_body_ids
        previous = self._prev_attack_pos if side == 0 else self._opponent_prev_attack_pos
        cooldown = self._hit_cooldown_buf if side == 0 else self._opponent_hit_cooldown_buf
        active = self._attack_active_buf if side == 0 else self._opponent_attack_active_buf
        consumed = self._attack_hit_consumed if side == 0 else self._opponent_punch_hit_consumed
        speed_buf = self._attack_speed_buf if side == 0 else self._opponent_attack_speed_buf
        approach_buf = self._approach_speed_buf if side == 0 else self._opponent_approach_speed_buf
        cooldown.sub_(dt).clamp_min_(0.0)
        variant = torch.where(self._combo_second[:, side] &
                              (self._combat_state[:, side] == CombatState.ATTACK),
                              self._combo_followup_variant[:, side], self._attack_variant[:, side])
        rows = torch.arange(self.get_num_envs(), device=self._device)
        hand = self._boxing_reference.punch_hand(variant)
        attack_pos = self._engine.get_body_pos(attacker)[:, hand_ids, :]
        target_pos = self._engine.get_body_pos(defender)[:, target_ids, :]
        velocity = (attack_pos - previous) / dt
        previous[:] = attack_pos
        chosen_pos = attack_pos[rows, hand]
        chosen_vel = velocity[rows, hand]
        separation = target_pos - chosen_pos[:, None, :]
        distance = torch.linalg.vector_norm(separation, dim=-1).clamp_min(1e-6)
        approach = torch.sum(chosen_vel[:, None, :] * separation / distance[..., None], dim=-1)
        target_index = torch.arange(len(target_ids), device=self._device)[None, :]
        head_target = target_index == self._target_body_names.index("head")
        body_target = ((target_index == self._target_body_names.index("torso")) |
                       (target_index == self._target_body_names.index("pelvis")))
        head_punch = (variant == 3) | (variant == 4)
        target_allowed = ((~head_punch & (variant != 5))[:, None] |
                          (head_punch[:, None] & head_target) |
                          ((variant == 5)[:, None] & body_target))
        closest = distance.masked_fill(~target_allowed, float("inf"))
        nearest = torch.argmin(closest, dim=-1)
        nearest_distance = closest[rows, nearest]
        speed_buf[:] = torch.linalg.vector_norm(chosen_vel, dim=-1)
        approach_buf[:] = approach[rows, nearest]
        attack_force = torch.linalg.vector_norm(
            self._engine.get_contact_forces(attacker)[:, hand_ids, :], dim=-1)[rows, hand]
        target_force = torch.linalg.vector_norm(
            self._engine.get_contact_forces(defender)[:, target_ids, :], dim=-1)[rows, nearest]
        if side == 0:
            self._attack_contact_force_buf[:] = attack_force
            self._target_contact_force_buf[:] = target_force
        moving = (speed_buf >= self._min_attack_speed) & (approach_buf >= self._min_approach_speed)
        impact = moving | (attack_force >= self._contact_force_threshold)
        fighter_distance = torch.linalg.vector_norm(
            self._combat_pos[:, 1, :2] - self._combat_pos[:, 0, :2], dim=-1)
        in_range = ((variant != 3) & (variant != 4) & (variant != 5)) | (
            (variant == 3) & (fighter_distance >= 0.35) & (fighter_distance <= 0.65)) | (
            (variant == 4) & (fighter_distance >= 0.25) & (fighter_distance <= 0.65)) | (
            (variant == 5) & (fighter_distance >= 0.45) & (fighter_distance <= 0.65))
        return (nearest_distance <= self._hit_distance) & in_range & active & ~consumed & (cooldown <= 0) & impact

    def _update_hits(self):
        self._hit_buf[:] = self._detect_punch_hit(0)
        hit_ids = self._hit_buf.nonzero(as_tuple=False).flatten()
        if len(hit_ids) > 0:
            blocked = self._combat_state[hit_ids, 1] == CombatState.BLOCK
            damage = torch.where(blocked,
                                 torch.full_like(self._opponent_health[hit_ids], self._hit_damage * self._block_damage_scale),
                                 torch.full_like(self._opponent_health[hit_ids], self._hit_damage))
            self._opponent_health[hit_ids] -= damage
            self._hit_cooldown_buf[hit_ids] = self._hit_cooldown
            self._attack_hit_consumed[hit_ids] = True
            self._stagger_buf[hit_ids] = self._stagger_duration

            unblocked_ids = hit_ids[(~blocked) | (self._opponent_health[hit_ids] <= 0)]
            if len(unblocked_ids):
                self._receive_hit(unblocked_ids, 1)
            self._set_character_color(self._get_opponent_id(),
                                      [1.0, 0.85, 0.05],
                                      hit_ids[self._opponent_health[hit_ids] > 0])
            knocked_down = self._opponent_health[hit_ids] <= 0.0
            knocked_down_ids = hit_ids[knocked_down]
            if len(knocked_down_ids) > 0:
                self._knocked_down[knocked_down_ids] = True
                self._apply_knockdown(knocked_down_ids)
            Logger.print("HIT! opponent health: {:.1f}".format(
                float(self._opponent_health[hit_ids[0]].item())))
        return

    def _update_opponent_hits(self):
        if not self._opponent_attack_enabled:
            self._opponent_hit_buf[:] = False
            return

        self._opponent_hit_buf[:] = self._detect_punch_hit(1)
        hit_ids = self._opponent_hit_buf.nonzero(as_tuple=False).flatten()
        if len(hit_ids) > 0:
            blocked = self._combat_state[hit_ids, 0] == CombatState.BLOCK
            damage = torch.where(blocked,
                                 torch.full_like(self._fighter_health[hit_ids], self._hit_damage * self._block_damage_scale),
                                 torch.full_like(self._fighter_health[hit_ids], self._hit_damage))
            self._fighter_health[hit_ids] -= damage
            self._opponent_hit_cooldown_buf[hit_ids] = self._hit_cooldown
            self._opponent_punch_hit_consumed[hit_ids] = True
            self._fighter_stagger_buf[hit_ids] = self._stagger_duration
            unblocked_ids = hit_ids[(~blocked) | (self._fighter_health[hit_ids] <= 0)]
            if len(unblocked_ids):
                self._receive_hit(unblocked_ids, 0)
            self._set_character_color(self._get_char_id(),
                                      [1.0, 0.85, 0.05],
                                      hit_ids[self._fighter_health[hit_ids] > 0])
            Logger.print("FIGHTER HIT! fighter health: {:.1f}".format(
                float(self._fighter_health[hit_ids[0]].item())))
        return

    def _apply_knockdown(self, env_ids, opponent_id=None):
        if opponent_id is None:
            opponent_id = self._get_opponent_id()
        root_pos = self._engine.get_root_pos(opponent_id)[env_ids].clone()
        root_pos[:, 2] = self._ring_floor_height + 0.35
        axis = torch.zeros([len(env_ids), 3], device=self._device)
        axis[:, 1] = 1.0
        angle = torch.full(
            [len(env_ids)], np.pi * 0.5, device=self._device)
        root_rot = torch_util.axis_angle_to_quat(axis, angle)
        dof_pos = self._engine.get_dof_pos(opponent_id)[env_ids]
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        body_pos, body_rot = self._kin_char_model.forward_kinematics(
            root_pos, root_rot, joint_rot)

        self._engine.set_root_pos(env_ids, opponent_id, root_pos)
        self._engine.set_root_rot(env_ids, opponent_id, root_rot)
        self._engine.set_root_vel(env_ids, opponent_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, opponent_id, 0.0)
        self._engine.set_dof_vel(env_ids, opponent_id, 0.0)
        self._engine.set_body_vel(env_ids, opponent_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, opponent_id, 0.0)
        self._engine.set_body_pos(env_ids, opponent_id, body_pos)
        self._engine.set_body_rot(env_ids, opponent_id, body_rot)
        self._set_character_color(opponent_id, [0.35, 0.08, 0.08], env_ids)
        return

    def _get_char_color(self):
        if self._engine.get_name() == "newton":
            return np.array([0.35, 0.45, 0.7])
        return np.array([0.5, 0.65, 0.95])

    def _set_character_color(self, obj_id, color, env_ids=None):
        if not self._visualize:
            return
        if self._engine.get_name() != "newton":
            return

        num_bodies = self._engine.get_obj_num_bodies(obj_id)
        if env_ids is None:
            env_ids = range(self.get_num_envs())
        for env_id in env_ids:
            for body_id in range(num_bodies):
                self._engine.set_body_color(
                    int(env_id), obj_id, body_id, np.asarray(color, dtype=np.float32))
        return

    def _reset_opponent(self, env_ids):
        opponent_id = self._get_opponent_id()
        n = len(env_ids)
        root_pos = torch.as_tensor(self._opponent_start_pos,
                                   device=self._device).expand(n, -1)
        root_rot = torch.as_tensor(self._opponent_start_rot,
                                   device=self._device).expand(n, -1)
        dof_pos = self._init_dof_pos.expand(n, -1)

        self._engine.set_root_pos(env_ids, opponent_id, root_pos)
        self._engine.set_root_rot(env_ids, opponent_id, root_rot)
        self._engine.set_root_vel(env_ids, opponent_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, opponent_id, 0.0)
        self._engine.set_dof_pos(env_ids, opponent_id, dof_pos)
        self._engine.set_dof_vel(env_ids, opponent_id, 0.0)
        self._engine.set_body_vel(env_ids, opponent_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, opponent_id, 0.0)

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        body_pos, body_rot = self._kin_char_model.forward_kinematics(
            root_pos, root_rot, joint_rot)
        self._engine.set_body_pos(env_ids, opponent_id, body_pos)
        self._engine.set_body_rot(env_ids, opponent_id, body_rot)
        return

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)

        fighter_id = self._get_char_id()
        opponent_id = self._get_opponent_id()
        fighter_pos = self._engine.get_root_pos(fighter_id)
        fighter_vel = self._engine.get_root_vel(fighter_id)
        opponent_pos = self._engine.get_root_pos(opponent_id)
        opponent_vel = self._engine.get_root_vel(opponent_id)

        if env_ids is not None:
            fighter_pos = fighter_pos[env_ids]
            fighter_vel = fighter_vel[env_ids]
            opponent_pos = opponent_pos[env_ids]
            opponent_vel = opponent_vel[env_ids]

        relative_pos = opponent_pos - fighter_pos
        relative_vel = opponent_vel - fighter_vel
        combat_obs = torch.cat([relative_pos, relative_vel], dim=-1)
        return torch.cat([obs, combat_obs], dim=-1)

    def _update_reward(self):
        super()._update_reward()
        self._reward_buf += self._hit_reward * self._hit_buf.float()
        return

    def _update_done(self):
        super()._update_done()
        self._done_buf[self._opponent_health <= 0.0] = base_env.DoneFlags.SUCC.value
        self._done_buf[self._fighter_health <= 0.0] = base_env.DoneFlags.FAIL.value
        return

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)
        selection = slice(None) if env_ids is None else env_ids
        ref_intent, ref_direction, ref_distance, ref_guard = self._boxing_reference.compute(
            self._engine.get_root_pos(self._get_char_id()),
            self._engine.get_root_pos(self._get_opponent_id()),
            self._combat_state[:, 0], self._fighter_health, self._opponent_health)
        self._info["boxing_reference_intent"] = ref_intent[selection]
        self._info["boxing_reference_direction"] = ref_direction[selection]
        self._info["boxing_reference_distance"] = ref_distance[selection]
        self._info["boxing_reference_guard_target"] = ref_guard[selection]
        for side, prefix in enumerate(("fighter", "opponent")):
            self._info[prefix + "_state"] = self._combat_state[selection, side]
            self._info[prefix + "_attack_cooldown"] = self._attack_cooldown[selection, side]
            stun = self._fighter_stagger_buf if side == 0 else self._stagger_buf
            self._info[prefix + "_hit_stun"] = stun[selection]
        self._info["combat_state_names"] = tuple(state.name for state in CombatState)
        self._info["fighter_stamina"] = self._stamina[selection, 0]
        self._info["opponent_stamina"] = self._stamina[selection, 1]
        self._info["boxing_combo"] = self._combo[selection]
        active_punch = torch.where(
            self._combo_second & (self._combat_state == CombatState.ATTACK),
            self._combo_followup_variant, self._attack_variant)
        self._info["attack_variant_names"] = self._boxing_reference.PUNCH_NAMES
        self._info["fighter_attack_variant"] = active_punch[selection, 0]
        self._info["opponent_attack_variant"] = active_punch[selection, 1]
        self._info["support_foot_slip"] = self._support_slip[selection]
        self._info["support_raw_slip"] = self._support_raw_slip[selection]
        self._info["support_ik_error"] = self._support_ik_error[selection]
        self._info["motor_pose_error"] = self._motor_pose_error[selection]
        self._info["motor_root_correction"] = self._motor_root_correction[selection]
        self._info["foot_canvas_contact"] = self._foot_canvas_contact[selection]
        self._info["fighter_distance"] = torch.linalg.vector_norm(
            self._engine.get_root_pos(self._get_char_id())[:, :2] -
            self._engine.get_root_pos(self._get_opponent_id())[:, :2], dim=-1)[selection]
        if env_ids is None:
            self._info["combat_hit"] = self._hit_buf
            self._info["opponent_health"] = self._opponent_health
            self._info["fighter_health"] = self._fighter_health
            self._info["attack_active"] = self._attack_active_buf
            self._info["opponent_stagger"] = self._stagger_buf
            self._info["attack_speed"] = self._attack_speed_buf
            self._info["approach_speed"] = self._approach_speed_buf
            self._info["attack_contact_force"] = self._attack_contact_force_buf
            self._info["target_contact_force"] = self._target_contact_force_buf
            self._info["opponent_knocked_down"] = self._knocked_down
            self._info["fighter_hit"] = self._opponent_hit_buf
            self._info["opponent_attack_active"] = self._opponent_attack_active_buf
            self._info["opponent_attack_speed"] = self._opponent_attack_speed_buf
        else:
            self._info["combat_hit"] = self._hit_buf[env_ids]
            self._info["opponent_health"] = self._opponent_health[env_ids]
            self._info["fighter_health"] = self._fighter_health[env_ids]
            self._info["attack_active"] = self._attack_active_buf[env_ids]
            self._info["opponent_stagger"] = self._stagger_buf[env_ids]
            self._info["attack_speed"] = self._attack_speed_buf[env_ids]
            self._info["approach_speed"] = self._approach_speed_buf[env_ids]
            self._info["attack_contact_force"] = self._attack_contact_force_buf[env_ids]
            self._info["target_contact_force"] = self._target_contact_force_buf[env_ids]
            self._info["opponent_knocked_down"] = self._knocked_down[env_ids]
            self._info["fighter_hit"] = self._opponent_hit_buf[env_ids]
            self._info["opponent_attack_active"] = self._opponent_attack_active_buf[env_ids]
            self._info["opponent_attack_speed"] = self._opponent_attack_speed_buf[env_ids]
        return

