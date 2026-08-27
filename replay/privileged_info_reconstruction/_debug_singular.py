import sys
sys.path.insert(0, "/nethome/chuang475/testnvme/projects/SafeManip/replay/privileged_info_reconstruction")
import json
import numpy as np
import reconstruct_video as rv

p = "/nethome/chuang475/testnvme/projects/SafeManip/results/evals/all_tasks_3_ckpt_50_rollouts/target_posttraining/evals/target/ArrangeBreadBasket/rollout_data/ArrangeBreadBasket--2026_05_05-00_22_16/privileged_information_2.json"
d = json.load(open(p))
si = d["privileged_static_info"]
episode_meta = si["task"]["episode_meta"]
seed = d["replay_summary"]["seed"]
dyn = d["privileged_dynamic_info"]

print("=== FRESH env, episode 2 ===")
env, raw = rv.make_env("ArrangeBreadBasket", seed, "target", episode_meta)
sim = raw.env.sim
root_body = dyn[0]["data"]["robot"]["root_body"]
forward0 = sim.data.get_joint_qpos(rv.MOBILE_BASE_JOINTS[0])
side0 = sim.data.get_joint_qpos(rv.MOBILE_BASE_JOINTS[1])
yaw0 = sim.data.get_joint_qpos(rv.MOBILE_BASE_JOINTS[2])
torso0 = sim.data.get_joint_qpos(rv.MOBILE_BASE_JOINTS[3])
print("forward0,side0,yaw0,torso0 =", forward0, side0, yaw0, torso0)
support_z0 = sim.data.get_body_xpos(rv.TORSO_SUPPORT_BODY)[2]
print("support_z0 =", support_z0, "torso range check: is torso0 near limit [0,0.34]?", torso0)

# manually replicate _measure_jacobian_inv but print the raw jacobian before inverting
calib_tmp = rv.MobileBaseCalibrator.__new__(rv.MobileBaseCalibrator)
from scipy.spatial.transform import Rotation as R
calib_tmp._R = R
calib_tmp._root_body = root_body

def measure(forward, side, yaw, torso):
    sim.data.set_joint_qpos(rv.MOBILE_BASE_JOINTS[0], forward)
    sim.data.set_joint_qpos(rv.MOBILE_BASE_JOINTS[1], side)
    sim.data.set_joint_qpos(rv.MOBILE_BASE_JOINTS[2], yaw)
    sim.data.set_joint_qpos(rv.MOBILE_BASE_JOINTS[3], torso)
    sim.forward()
    bid = sim.model.body_name2id(root_body)
    xy = sim.data.xpos[bid][:2].copy()
    body_yaw = R.from_quat(rv._wxyz_to_xyzw(sim.data.xquat[bid])).as_euler("xyz")[2]
    sz = sim.data.get_body_xpos(rv.TORSO_SUPPORT_BODY)[2]
    return np.array([xy[0], xy[1], body_yaw, sz])

base = measure(forward0, side0, yaw0, torso0)
print("base measurement:", base)
eps = 1e-3
for i, name in enumerate(["forward", "side", "yaw", "torso"]):
    perturbed = [forward0, side0, yaw0, torso0]
    perturbed[i] += eps
    m = measure(*perturbed)
    print(f"perturb {name}: delta = {(m - base)}")

print()
print("=== readback check ===")
sim.data.set_joint_qpos(rv.MOBILE_BASE_JOINTS[0], forward0 + 1.0)
sim.forward()
print("set forward+1.0, readback:", sim.data.get_joint_qpos(rv.MOBILE_BASE_JOINTS[0]))
bid = sim.model.body_name2id(root_body)
print("root_body name:", root_body, "-> resolved id:", bid, "-> body name from id:", sim.model.body_id2name(bid))
print("xpos at that body:", sim.data.xpos[bid])
print("get_body_xpos same call:", sim.data.get_body_xpos(root_body))

# is there possibly a second body also named similarly, or is root_body body actually static (no joint)?
jnt_adr = sim.model.body_jntadr[bid]
jnt_num = sim.model.body_jntnum[bid]
print("body_jntadr:", jnt_adr, "body_jntnum:", jnt_num)
for j in range(jnt_adr, jnt_adr + jnt_num):
    print("  joint", j, "name:", sim.model.joint_id2name(j))
