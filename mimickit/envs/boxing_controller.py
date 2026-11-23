"""Deterministic boxing reference controller for residual policies.

Controller supplies high-level intent only. Physics action remains owned by
CombatEnv, allowing PPO to learn residual corrections around reference motion.
"""

from enum import IntEnum

import torch


class BoxingIntent(IntEnum):
    GUARD = 0
    APPROACH = 1
    JAB = 2
    RECOVER = 3
    RETREAT = 4
    BLOCK = 5


class BoxingReferenceController:
    PUNCH_NAMES = ("JAB", "CROSS", "HOOK", "REAR_HOOK", "UPPERCUT", "BODY_JAB")

    @staticmethod
    def punch_stamina_cost(variant):
        costs = variant.new_tensor([0.18, 0.24, 0.24, 0.25, 0.25, 0.18], dtype=torch.float32)
        return costs[variant]

    @staticmethod
    def punch_recovery_adjustment(variant):
        seconds = variant.new_tensor([0.0, 0.08, 0.06, 0.15, 0.15, -0.10], dtype=torch.float32)
        return seconds[variant]

    @staticmethod
    def punch_hand(variant):
        """Index in configured [right_hand, left_hand] attack bodies."""
        return torch.where((variant == 1) | (variant == 3) | (variant == 4), 0, 1)

    @staticmethod
    def punch_duration(variant):
        """Animation length for each punch variant, in seconds."""
        return torch.where(variant == 0, 0.9,
                           torch.where(variant == 1, 1.2,
                                       torch.where(variant == 2, 1.0,
                                                   torch.where(variant == 3, 1.1,
                                                               torch.where(variant == 4, 1.05, 0.85)))))

    def boxing_pose(self, base, joint_ids, variant, time, attacking):
        """Guard plus six punch trajectories.

        Shoulder coordinates are exponential maps. Every punch returns to guard.
        """
        pose = base.clone()
        duration = self.punch_duration(variant)
        phase = (time / duration).clamp(0, 1)
        progress = torch.where(phase < 0.45, phase / 0.45,
                               (1 - phase) / 0.55).clamp(0, 1)
        progress = progress.square() * (3 - 2 * progress)
        progress *= attacking.to(base.dtype)
        for arm, (shoulder, _, elbow) in enumerate(joint_ids):
            sign = 1.0 if arm == 0 else -1.0
            guard = base.new_tensor([1.1302 * sign, -0.5409, 0.5800 * sign])
            straight = base.new_tensor([0.0, 0.0, 1.85 * sign])
            hook = base.new_tensor([0.20, 0.0, 1.60 * sign])
            is_hook = (variant == 2) | (variant == 3)
            extended = torch.where(is_hook[:, None], hook, straight)
            selected = (self.punch_hand(variant) == arm)
            blend = progress * selected
            pose[:, shoulder:shoulder + 3] = guard + blend[:, None] * (extended - guard)
            extension = torch.where(is_hook, 0.65, 0.15)
            pose[:, elbow] = sign * (2.1832 + (extension - 2.1832) * blend)
            if arm == 1:
                body_jab = base.new_tensor([0.0, 0.55, -1.85])
                body_mask = (variant == 5) & attacking
                pose[:, shoulder:shoulder + 3] = torch.where(
                    body_mask[:, None], guard + progress[:, None] * (body_jab - guard),
                    pose[:, shoulder:shoulder + 3])
                pose[:, elbow] = torch.where(
                    body_mask, -2.1832 + progress * (2.1832 - 0.20), pose[:, elbow])
            if arm == 0:
                windup = base.new_tensor([0.40, 0.45, 1.40])
                strike = base.new_tensor([0.40, -0.50, 1.40])
                down = (phase / 0.28).clamp(0, 1)
                rise = ((phase - 0.28) / 0.37).clamp(0, 1)
                return_guard = ((phase - 0.65) / 0.35).clamp(0, 1)
                upper = torch.where((phase < 0.28)[:, None],
                                    guard + down[:, None] * (windup - guard),
                                    torch.where((phase < 0.65)[:, None],
                                                windup + rise[:, None] * (strike - windup),
                                                strike + return_guard[:, None] * (guard - strike)))
                upper_elbow = torch.where(phase < 0.28,
                                          2.1832 + down * (1.10 - 2.1832),
                                          torch.where(phase < 0.65,
                                                      1.10 + rise * (0.90 - 1.10),
                                                      0.90 + return_guard * (2.1832 - 0.90)))
                upper_mask = (variant == 4) & attacking
                pose[:, shoulder:shoulder + 3] = torch.where(
                    upper_mask[:, None], upper, pose[:, shoulder:shoulder + 3])
                pose[:, elbow] = torch.where(upper_mask, upper_elbow, pose[:, elbow])
        active_start = torch.where(variant == 3, 0.30, 0.25)
        active_end = torch.where(variant == 3, 0.58, 0.60)
        active_start = torch.where(variant == 4, 0.42, active_start)
        active_end = torch.where(variant == 4, 0.67, active_end)
        active = attacking & (phase >= active_start) & (phase <= active_end)
        return pose, active

    def __init__(self, attack_range=1.15, preferred_distance=1.0,
                 retreat_distance=0.72, guard_height=1.05):
        self.attack_range = float(attack_range)
        self.preferred_distance = float(preferred_distance)
        self.retreat_distance = float(retreat_distance)
        self.guard_height = float(guard_height)

    def compute(self, fighter_pos, opponent_pos, fighter_state, fighter_health,
                opponent_health):
        """Return intent, planar direction, distance and guard target."""
        delta = opponent_pos[:, :2] - fighter_pos[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
        direction = delta / distance[:, None]
        live = (fighter_health > 0.0) & (opponent_health > 0.0)
        intent = torch.full_like(fighter_state, BoxingIntent.GUARD)
        intent[(fighter_state == 1) & live] = BoxingIntent.APPROACH
        intent[(fighter_state == 2) & live] = BoxingIntent.JAB
        intent[(fighter_state == 3) & live] = BoxingIntent.RECOVER
        intent[(fighter_state == 4) & live] = BoxingIntent.RETREAT
        intent[(distance > self.attack_range) & live & (fighter_state == 0)] = BoxingIntent.APPROACH
        retreat = (distance < self.retreat_distance) & live & (fighter_state == 0)
        intent[retreat] = BoxingIntent.RETREAT
        guard_target = torch.zeros_like(fighter_pos)
        guard_target[:, :2] = fighter_pos[:, :2] + direction * self.preferred_distance
        guard_target[:, 2] = self.guard_height
        return intent, direction, distance, guard_target

    def reference_action(self, init_dof, intent, side, joint_ids):
        """Build simple guard/punch targets; residual policy refines them."""
        action = init_dof.clone()
        shoulder_x, shoulder_y, elbow = joint_ids
        live = intent != BoxingIntent.GUARD
        zeros = torch.zeros_like(intent, dtype=action.dtype)
        jab = intent == BoxingIntent.JAB
        action[:, shoulder_x] += torch.where(jab, torch.full_like(zeros, -0.45), zeros)
        action[:, shoulder_y] += torch.where(
            jab, torch.full_like(zeros, 0.25 if side == 0 else -0.25), zeros)
        action[:, elbow] += torch.where(
            jab, torch.full_like(zeros, 0.55 if side == 0 else -0.55),
            torch.where(live, torch.full_like(zeros, 0.25 if side == 0 else -0.25), zeros))
        return action
