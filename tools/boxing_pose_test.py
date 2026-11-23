"""CPU checks for six punches and their guard endpoints."""
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))
import torch
from anim.mjcf_char_model import MJCFCharModel
from envs.boxing_controller import BoxingReferenceController

model = MJCFCharModel("cpu")
model.load("data/assets/humanoid/humanoid.xml")
def dof(name):
    return next(j.dof_idx for j in model._joints if j is not None and j.name == name)

ids = [(dof(s + "_shoulder"), dof(s + "_shoulder") + 1,
        dof(s + "_elbow")) for s in ("right", "left")]
controller = BoxingReferenceController()
base = torch.zeros(6, model.get_dof_size())
variant = torch.tensor([0, 1, 2, 3, 4, 5])
attacking = torch.ones(6, dtype=torch.bool)
duration = controller.punch_duration(variant)
assert torch.equal(controller.punch_stamina_cost(variant),
                   torch.tensor([0.18, 0.24, 0.24, 0.25, 0.25, 0.18]))
assert controller.punch_recovery_adjustment(variant)[3] > controller.punch_recovery_adjustment(variant)[5]
guard, _ = controller.boxing_pose(base, ids, variant, torch.zeros(6), attacking)
peak, active = controller.boxing_pose(base, ids, variant, duration * 0.45, attacking)
end, inactive = controller.boxing_pose(base, ids, variant, duration, attacking)
assert torch.equal(guard, end) and active.all() and not inactive.any()
for row, resting_arm in ((0, 0), (1, 1), (2, 0), (3, 1), (4, 1), (5, 0)):
    start, _, elbow = ids[resting_arm]
    assert torch.equal(guard[row, start:elbow + 1], peak[row, start:elbow + 1])
    start, _, elbow = ids[1 - resting_arm]
    assert not torch.equal(guard[row, start:elbow + 1], peak[row, start:elbow + 1])
pos, _ = model.forward_kinematics(torch.tensor([[0., 0., .882416]]).expand(6, -1),
                                 torch.tensor([[0., 0., 0., 1.]]).expand(6, -1),
                                 model.dof_to_rot(guard))
peak_pos, _ = model.forward_kinematics(torch.tensor([[0., 0., .882416]]).expand(6, -1),
                                      torch.tensor([[0., 0., 0., 1.]]).expand(6, -1),
                                      model.dof_to_rot(peak))
names = model.get_body_names()
hook_hand = peak_pos[2, names.index("left_hand")]
guard_hand = pos[2, names.index("left_hand")]
assert hook_hand[1] < guard_hand[1] - 0.1
assert abs(peak[2, ids[1][2]]) > abs(peak[0, ids[1][2]])
rear_hook_hand = peak_pos[3, names.index("right_hand")]
assert rear_hook_hand[1] > pos[3, names.index("right_hand")][1] + 0.1
assert abs(peak[3, ids[0][2]]) > abs(peak[1, ids[0][2]])
windup, _ = controller.boxing_pose(base, ids, variant, duration * 0.28, attacking)
strike, _ = controller.boxing_pose(base, ids, variant, duration * 0.65, attacking)
windup_pos, _ = model.forward_kinematics(
    torch.tensor([[0., 0., .882416]]).expand(6, -1),
    torch.tensor([[0., 0., 0., 1.]]).expand(6, -1), model.dof_to_rot(windup))
strike_pos, _ = model.forward_kinematics(
    torch.tensor([[0., 0., .882416]]).expand(6, -1),
    torch.tensor([[0., 0., 0., 1.]]).expand(6, -1), model.dof_to_rot(strike))
assert strike_pos[4, names.index("right_hand"), 2] > windup_pos[4, names.index("right_hand"), 2] + 0.25
assert peak_pos[5, names.index("left_hand"), 2] < peak_pos[0, names.index("left_hand"), 2] - 0.18
for side in ("right", "left"):
    hand = pos[:, names.index(side + "_hand")]
    elbow = pos[:, names.index(side + "_lower_arm")]
    assert (hand[:, 2] > elbow[:, 2] + .15).all()
    assert (hand[:, 0] > 0).all()
config = yaml.safe_load(Path('data/envs/combat_humanoid_env.yaml').read_text())
init_pose = torch.tensor(config['init_pose'], dtype=torch.float32)
stance_pos, _ = model.forward_kinematics(
    init_pose[:3][None], torch.tensor([[0., 0., 0., 1.]]),
    model.dof_to_rot(init_pose[6:][None]))
right_foot = stance_pos[0, names.index('right_foot')]
left_foot = stance_pos[0, names.index('left_foot')]
assert left_foot[0] - right_foot[0] > 0.2
for foot in (right_foot, left_foot):
    assert abs(float(foot[2]) - (config['ring_floor_height'] + 0.05)) < 0.015
print("Jab, cross, lead/rear hook, uppercut, body jab and guard return passed")
