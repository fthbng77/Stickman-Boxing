"""Run from repo root: .venv-py310/bin/python tools/combat_state_smoke.py [--visualize]."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mimickit'))
import torch
from envs.env_builder import build_env
from envs.combat_env import CombatState as S

parser = argparse.ArgumentParser()
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--motor-pose', action='store_true')
args = parser.parse_args()
if args.visualize:
    from util.display import ensure_virtual_display
    ensure_virtual_display()
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
    out = Path('output/combat_humanoid/state_machine.mp4')
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=30, codec='libx264', macro_block_size=2)
torch.manual_seed(7)
if args.motor_pose:
    from envs.combat_env import CombatEnv
    from envs.env_builder import load_configs
    env_config, engine_config = load_configs('data/envs/combat_humanoid_env.yaml',
                                             'data/engines/newton_engine.yaml')
    env_config['motor_pose'] = True
    env = CombatEnv(env_config, engine_config, 1, 'cuda:0', args.visualize)
else:
    env = build_env('data/envs/combat_humanoid_env.yaml',
                    'data/engines/newton_engine.yaml', 1, 'cuda:0', args.visualize)
env.reset()
action = torch.zeros((1, env.get_action_space().shape[0]), device='cuda:0')
seen = [set(), set()]
transitions = [set(), set()]
previous = [S.IDLE, S.IDLE]
hits = [0, 0]
hits_by_punch = [[0] * 6, [0] * 6]
footwork_travel = [0.0, 0.0]
previous_positions = [env._engine.get_root_pos(obj)[0, :2].clone()
                      for obj in (env._get_char_id(), env._get_opponent_id())]
orbit_directions = [set(), set()]
combo_seen = [False, False]
punches_seen = [set(), set()]
attack_starts = [[], []]
max_support_slip = 0.0
max_raw_slip = 0.0
support_samples = 0
support_contacts = 0
max_ik_error = 0.0
max_motor_error = 0.0
max_motor_root_correction = 0.0
root_height_range = [[float('inf'), float('-inf')] for _ in range(2)]
for step in range(900):
    obs, reward, done, info = env.step(action)
    assert torch.isfinite(obs).all()
    states = env._combat_state[0].tolist()
    assert states != [S.ATTACK, S.ATTACK]
    for side, state in enumerate(states):
        seen[side].add(state)
        if state != previous[side]:
            transitions[side].add((previous[side], state))
            print(f'{step}: {side}: {S(previous[side]).name} -> {S(state).name}', flush=True)
            if state == S.ATTACK:
                attack_starts[side].append(int(env._attack_variant[0, side]))
        previous[side] = state
    hits[0] += int(info['combat_hit'][0])
    hits[1] += int(info['fighter_hit'][0])
    for side, hit_key in enumerate(('combat_hit', 'fighter_hit')):
        if bool(info[hit_key][0]):
            variant_key = ('fighter', 'opponent')[side] + '_attack_variant'
            hits_by_punch[side][int(info[variant_key][0])] += 1
    for side in range(2):
        combo_seen[side] |= bool(env._combo_second[0, side])
        if states[side] == S.ATTACK:
            punches_seen[side].add(int(info[('fighter', 'opponent')[side] + '_attack_variant'][0]))
        obj = env._get_char_id() if side == 0 else env._get_opponent_id()
        height = float(env._engine.get_root_pos(obj)[0, 2])
        root_height_range[side][0] = min(root_height_range[side][0], height)
        root_height_range[side][1] = max(root_height_range[side][1], height)
        current_pos = env._engine.get_root_pos(obj)[0, :2]
        if states[side] in (S.IDLE, S.RECOVER) and float(env._attack_cooldown[0, side]) > 0:
            footwork_travel[side] += float(torch.linalg.vector_norm(current_pos - previous_positions[side]))
            orbit_directions[side].add(int(env._orbit_sign[0, side]))
        previous_positions[side] = current_pos.clone()
    max_support_slip = max(max_support_slip, float(info['support_foot_slip'].max()))
    max_raw_slip = max(max_raw_slip, float(info['support_raw_slip'].max()))
    support_samples += int((env._support_foot >= 0).sum())
    for side in range(2):
        foot = int(env._support_foot[0, side])
        if foot >= 0:
            support_contacts += int(info['foot_canvas_contact'][0, side, foot])
    max_ik_error = max(max_ik_error, float(info['support_ik_error'].max()))
    if step >= 60:
        max_motor_error = max(max_motor_error, float(info['motor_pose_error'].max()))
        max_motor_root_correction = max(
            max_motor_root_correction, float(info['motor_root_correction'].max()))
    for obj in (env._get_char_id(), env._get_opponent_id()):
        pos = env._engine.get_root_pos(obj)[0]
        xmin, xmax, ymin, ymax = env._ring_bounds
        assert xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax
    if args.visualize:
        frame = Image.fromarray(env._engine._viewer.get_frame(render_ui=False).numpy().copy())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((10, 10, 700, 82), fill=(18, 22, 30))
        for side, label in enumerate(('BLUE', 'RED')):
            hp = env._fighter_health if side == 0 else env._opponent_health
            punch = (info['attack_variant_names'][int(info[('fighter', 'opponent')[side] +
                                                          '_attack_variant'][0])]
                     if states[side] == S.ATTACK else '-')
            draw.text((20, 20 + side * 28),
                      f'{label}  HP {float(hp[0]):.1f}  {S(states[side]).name}  '
                      f'{punch}  '
                      f'Stamina {float(env._stamina[0, side]):.2f}  '
                      f'Combo {bool(env._combo[0, side])}',
                      fill=(120, 170, 255) if side == 0 else (255, 130, 130))
        frame = frame.crop((0, 0, frame.width - frame.width % 2,
                            frame.height - frame.height % 2))
        writer.append_data(np.asarray(frame))
        if step in (3, 60, 120, 180, 240):
            frame.save(str(out.with_name(f'state_{step // 3:02d}.png')))
if args.visualize:
    writer.close()
if args.motor_pose:
    print('Motor diagnostics:', hits, hits_by_punch, punches_seen,
          footwork_travel, f'contacts={support_contacts}/{support_samples}',
          f'slip={max_support_slip:.4f}', f'ik={max_ik_error:.4f}',
          f'pose_error={max_motor_error:.4f}',
          f'root_correction={max_motor_root_correction:.4f}',
          f'root_heights={root_height_range}', flush=True)
for side in range(2):
    assert {S.IDLE, S.APPROACH, S.ATTACK, S.RECOVER} <= seen[side], seen
    assert (S.ATTACK, S.RECOVER) in transitions[side]
assert any(S.BLOCK in states for states in seen), seen
assert any(S.SLIP in states for states in seen), seen
assert any(S.BACKSTEP in states for states in seen), seen
assert min(hits) > 0, hits
if not args.motor_pose:
    assert sum(punch_hits[2] for punch_hits in hits_by_punch) > 0, hits_by_punch
assert (any(combo_seen) if args.motor_pose else all(combo_seen)), combo_seen
if args.motor_pose:
    assert len(set.union(*punches_seen)) >= 4, punches_seen
else:
    assert set.union(*punches_seen) == {0, 1, 2, 3, 4, 5}, punches_seen
assert all(all(a != b for a, b in zip(starts, starts[1:])) for starts in attack_starts), attack_starts
assert min(footwork_travel) > 1.0, footwork_travel
assert all(signs == {-1, 1} for signs in orbit_directions), orbit_directions
assert support_samples > 50, support_samples
assert max_support_slip < (0.035 if args.motor_pose else 0.03), max_support_slip
assert support_contacts / support_samples > (0.8 if args.motor_pose else 0.7), (support_contacts, support_samples)
if args.motor_pose:
    assert max_motor_error < 1.0, max_motor_error
    assert all(lo > env._ring_floor_height + 0.65 and
               hi < env._ring_floor_height + 1.1 for lo, hi in root_height_range), root_height_range
else:
    assert max_ik_error < 0.02, max_ik_error
print('900 steps passed; hits:', hits, flush=True)
print('Hits by jab/cross/lead hook/rear hook/uppercut/body jab:', hits_by_punch, flush=True)
print('Punch variants:', punches_seen, '; attack starts:', attack_starts, flush=True)
print('Cooldown footwork travel:', footwork_travel, 'm; orbit directions:', orbit_directions, flush=True)
print(f'Support footfall samples: {support_samples}; raw/corrected max slip: {max_raw_slip:.4f}/{max_support_slip:.4f} m', flush=True)
print(f'Canvas contact: {support_contacts}/{support_samples}; max IK error: {max_ik_error:.4f} m', flush=True)
if args.motor_pose:
    print(f'Motor max pose error after warmup: {max_motor_error:.4f} rad', flush=True)
# Exact-boundary outward velocity, both axes and both characters.
ids = torch.tensor([0], device='cuda:0')
# Exercise stun independently of the defense decisions in the free fight.
for side in range(2):
    env.reset()
    obj = env._get_char_id() if side == 0 else env._get_opponent_id()
    hit_start = env._engine.get_root_pos(obj)[0, :2].clone()
    env._receive_hit(ids, side)
    assert env._combat_state[0, side] == S.HIT_STUN
    for _ in range(12):
        env.step(action)
    assert env._combat_state[0, side] == S.RETREAT
    stun_travel = torch.linalg.vector_norm(
        env._engine.get_root_pos(obj)[0, :2] - hit_start)
    assert 0.05 < stun_travel < 0.35, (side, float(stun_travel))
assert ((env._stamina >= 0) & (env._stamina <= 1)).all()
for obj in (env._get_char_id(), env._get_opponent_id()):
    pos = env._engine.get_root_pos(obj).clone()
    pos[0, :2] = pos.new_tensor([xmin, ymin])
    env._engine.set_root_pos(ids, obj, pos)
    vel = torch.zeros_like(pos)
    vel[0, :2] = -2
    env._engine.set_root_vel(ids, obj, vel)
    env._enforce_ring_bounds()
    assert (env._engine.get_root_vel(obj)[0, :2] == 0).all()
# Each side can be knocked down; winner stops, episode ends, reset clears state.
for side in range(2):
    env.reset()
    health = env._fighter_health if side == 0 else env._opponent_health
    health[0] = 0
    env._receive_hit(ids, side)
    env.step(action)
    assert env._combat_state[0, side] == S.DOWN
    assert env._combat_state[0, 1-side] == S.IDLE
    assert not env._attack_active_buf.any() and not env._opponent_attack_active_buf.any()
    assert env._done_buf[0] != 0
    obj = env._get_char_id() if side == 0 else env._get_opponent_id()
    assert (env._engine.get_root_vel(obj) == 0).all()
env.reset()
assert (env._combat_state == S.IDLE).all()
print('Ring, bilateral knockdown, reset passed', flush=True)
