"""Headless targeted checks for selected hand, head target and impact gates."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))
import torch
from envs.env_builder import build_env
from envs.combat_env import CombatState

env = build_env("data/envs/combat_humanoid_env.yaml",
                "data/engines/newton_engine.yaml", 1, "cuda:0", False)
env.reset()
row = torch.tensor([0], device="cuda:0")
for side in (0, 1):
    attacker = env._get_char_id() if side == 0 else env._get_opponent_id()
    defender = env._get_opponent_id() if side == 0 else env._get_char_id()
    hand_ids = env._attack_body_ids if side == 0 else env._opponent_attack_body_ids
    target_ids = env._target_body_ids if side == 0 else env._opponent_target_body_ids
    previous = env._prev_attack_pos if side == 0 else env._opponent_prev_attack_pos
    active = env._attack_active_buf if side == 0 else env._opponent_attack_active_buf
    consumed = env._attack_hit_consumed if side == 0 else env._opponent_punch_hit_consumed
    cooldown = env._hit_cooldown_buf if side == 0 else env._opponent_hit_cooldown_buf
    env._combat_state[0, side] = CombatState.ATTACK
    env._attack_variant[0, side] = 3
    env._combo_second[0, side] = False
    active[0] = True
    consumed[0] = False
    cooldown[0] = 0
    env._combat_pos[0, 0, :2] = torch.tensor([0., 0.], device="cuda:0")
    env._combat_pos[0, 1, :2] = torch.tensor([0.5, 0.], device="cuda:0")
    attack_pos = env._engine.get_body_pos(attacker).clone()
    target_pos = env._engine.get_body_pos(defender).clone()
    attack_pos[:, hand_ids, :] = torch.tensor([2., 0., 1.], device="cuda:0")
    target_pos[:, target_ids, :] = torch.tensor([3., 0., 1.], device="cuda:0")
    head = env._target_body_names.index("head")
    torso = env._target_body_names.index("torso")
    target_pos[0, target_ids[head]] = torch.tensor([0., 0., 1.], device="cuda:0")
    attack_pos[0, hand_ids[0]] = torch.tensor([0.05, 0., 1.], device="cuda:0")

    def detect():
        env._engine.set_body_pos(row, attacker, attack_pos)
        env._engine.set_body_pos(row, defender, target_pos)
        previous[:] = attack_pos[:, hand_ids, :]
        chosen_hand = int(env._boxing_reference.punch_hand(env._attack_variant[0, side]))
        previous[0, chosen_hand, 0] += 0.3
        return bool(env._detect_punch_hit(side)[0])

    assert detect(), (side, "valid rear hook")
    attack_pos[0, hand_ids[0]] = torch.tensor([2., 0., 1.], device="cuda:0")
    attack_pos[0, hand_ids[1]] = torch.tensor([0.05, 0., 1.], device="cuda:0")
    assert not detect(), (side, "wrong hand")
    attack_pos[0, hand_ids[0]] = torch.tensor([0.05, 0., 1.], device="cuda:0")
    target_pos[0, target_ids[head]] = torch.tensor([3., 0., 1.], device="cuda:0")
    target_pos[0, target_ids[torso]] = torch.tensor([0., 0., 1.], device="cuda:0")
    assert not detect(), (side, "wrong target")
    target_pos[0, target_ids[head]] = torch.tensor([0., 0., 1.], device="cuda:0")
    env._combat_pos[0, 1, 0] = 0.8
    assert not detect(), (side, "out of range")
    env._combat_pos[0, 1, 0] = 0.5
    active[0] = False
    assert not detect(), (side, "outside active window")
    active[0] = True
    consumed[0] = True
    assert not detect(), (side, "second hit")
    consumed[0] = False
    cooldown[0] = 0.5
    assert not detect(), (side, "cooldown")
    cooldown[0] = 0
    env._attack_variant[0, side] = 4
    assert detect(), (side, "valid uppercut")
    target_pos[0, target_ids[head]] = torch.tensor([3., 0., 1.], device="cuda:0")
    assert not detect(), (side, "uppercut wrong target")
    target_pos[0, target_ids[head]] = torch.tensor([0., 0., 1.], device="cuda:0")
    env._combat_pos[0, 1, 0] = 0.7
    assert not detect(), (side, "uppercut out of range")
    env._attack_variant[0, side] = 5
    env._combat_pos[0, 1, 0] = 0.6
    attack_pos[0, hand_ids[0]] = torch.tensor([2., 0., 1.], device="cuda:0")
    attack_pos[0, hand_ids[1]] = torch.tensor([0.05, 0., 1.], device="cuda:0")
    target_pos[0, target_ids[head]] = torch.tensor([3., 0., 1.], device="cuda:0")
    target_pos[0, target_ids[torso]] = torch.tensor([0., 0., 1.], device="cuda:0")
    assert detect(), (side, "valid body jab")
    target_pos[0, target_ids[torso]] = torch.tensor([3., 0., 1.], device="cuda:0")
    target_pos[0, target_ids[head]] = torch.tensor([0., 0., 1.], device="cuda:0")
    assert not detect(), (side, "body jab wrong target")
    target_pos[0, target_ids[torso]] = torch.tensor([0., 0., 1.], device="cuda:0")
    attack_pos[0, hand_ids[0]] = torch.tensor([0.05, 0., 1.], device="cuda:0")
    attack_pos[0, hand_ids[1]] = torch.tensor([2., 0., 1.], device="cuda:0")
    assert not detect(), (side, "body jab wrong hand")
    attack_pos[0, hand_ids[1]] = torch.tensor([0.05, 0., 1.], device="cuda:0")
    env._combat_pos[0, 1, 0] = 0.7
    assert not detect(), (side, "body jab out of range")
    env.reset()
print("Rear hook, uppercut and body jab target gates passed on both sides")
