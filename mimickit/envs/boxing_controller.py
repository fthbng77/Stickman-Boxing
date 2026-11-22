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
    @staticmethod
    def punch_duration(variant):
        """Jab, cross and lead-hook animation lengths in seconds."""
        return torch.where(variant == 0, 0.9,
                           torch.where(variant == 1, 1.2, 1.0))

    def boxing_pose(self, base, joint_ids, variant, time, attacking):
        """Guard plus jab, cross and lead-hook joint trajectories.

        Shoulder coordinates are exponential maps. Variant 1 uses the right
        arm; variants 0 and 2 use the left. Every trajectory returns to guard.
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
            extended = torch.where((variant == 2)[:, None], hook, straight)
            selected = (variant == 1) if arm == 0 else (variant != 1)
            blend = progress * selected
            pose[:, shoulder:shoulder + 3] = guard + blend[:, None] * (extended - guard)
            extension = torch.where(variant == 2, 0.65, 0.15)
            pose[:, elbow] = sign * (2.1832 + (extension - 2.1832) * blend)
        active = attacking & (phase >= 0.25) & (phase <= 0.6)
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
