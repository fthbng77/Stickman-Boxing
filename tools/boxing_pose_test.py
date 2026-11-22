"""CPU checks for three distinct punches and their guard endpoints."""
from pathlib import Path
import sys

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
base = torch.zeros(3, model.get_dof_size())
variant = torch.tensor([0, 1, 2])
attacking = torch.ones(3, dtype=torch.bool)
duration = controller.punch_duration(variant)
guard, _ = controller.boxing_pose(base, ids, variant, torch.zeros(3), attacking)
peak, active = controller.boxing_pose(base, ids, variant, duration * 0.45, attacking)
end, inactive = controller.boxing_pose(base, ids, variant, duration, attacking)
assert torch.equal(guard, end) and active.all() and not inactive.any()
for row, resting_arm in ((0, 0), (1, 1), (2, 0)):
    start, _, elbow = ids[resting_arm]
    assert torch.equal(guard[row, start:elbow + 1], peak[row, start:elbow + 1])
    start, _, elbow = ids[1 - resting_arm]
    assert not torch.equal(guard[row, start:elbow + 1], peak[row, start:elbow + 1])
pos, _ = model.forward_kinematics(torch.tensor([[0., 0., .882416]]).expand(3, -1),
                                 torch.tensor([[0., 0., 0., 1.]]).expand(3, -1),
                                 model.dof_to_rot(guard))
peak_pos, _ = model.forward_kinematics(torch.tensor([[0., 0., .882416]]).expand(3, -1),
                                      torch.tensor([[0., 0., 0., 1.]]).expand(3, -1),
                                      model.dof_to_rot(peak))
names = model.get_body_names()
hook_hand = peak_pos[2, names.index("left_hand")]
guard_hand = pos[2, names.index("left_hand")]
assert hook_hand[1] < guard_hand[1] - 0.1
assert abs(peak[2, ids[1][2]]) > abs(peak[0, ids[1][2]])
for side in ("right", "left"):
    hand = pos[:, names.index(side + "_hand")]
    elbow = pos[:, names.index(side + "_lower_arm")]
    assert (hand[:, 2] > elbow[:, 2] + .15).all()
    assert (hand[:, 0] > 0).all()
print("Jab, cross, hook, resting guard and return endpoints passed")
