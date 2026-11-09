"""Read-only compiled-model comparison; does not train or edit either XML."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def constants(path, name):
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == name for x in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(name)


def model_info(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    assert len(free) == 1
    base = int(model.jnt_qposadr[free[0]])
    data.qpos[base:base + 7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    records = []
    for jid in range(model.njnt):
        if jid in free:
            continue
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
        adr = int(model.jnt_qposadr[jid])
        records.append({"name": model.joint(jid).name, "joint_id": jid, "qpos_address": adr,
                        "qvel_address": int(model.jnt_dofadr[jid]), "export_dof_column": adr - 7,
                        "axis_local": model.jnt_axis[jid].tolist(), "axis_world_at_zero": data.xaxis[jid].tolist(),
                        "anchor_world_at_zero": data.xanchor[jid].tolist(), "range_rad": model.jnt_range[jid].tolist(),
                        "qpos0_rad": float(model.qpos0[adr]), "limited": bool(model.jnt_limited[jid]),
                        "body_name": model.body(model.jnt_bodyid[jid]).name})
    return model, data, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-xml", type=Path, required=True)
    parser.add_argument("--runtime-xml", type=Path, required=True)
    parser.add_argument("--runtime-constants", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, sd, sr = model_info(args.source_xml)
    target, td, tr = model_info(args.runtime_xml)
    order = constants(args.runtime_constants, "G1_JOINT_NAMES")
    tracked = constants(args.runtime_constants, "TRACKED_BODIES")
    source_dict = {x["name"]: x for x in sr}
    target_dict = {x["name"]: x for x in tr}
    assert set(order) == set(source_dict) == set(target_dict)
    rows, permutation, signs, offsets = [], [], [], []
    for i, name in enumerate(order):
        s, t = source_dict[name], target_dict[name]
        dot = float(np.dot(s["axis_world_at_zero"], t["axis_world_at_zero"]))
        axis_match = abs(abs(dot) - 1) < 1e-8
        sign = 1 if dot >= 0 else -1
        offset = t["qpos0_rad"] - sign * s["qpos0_rad"]
        permutation.append(s["export_dof_column"])
        signs.append(sign)
        offsets.append(offset)
        rows.append({"csv_column": 7 + i, "csv_joint_index": i, "name": name,
                     "source_dof_column": s["export_dof_column"], "runtime_qpos_address": t["qpos_address"],
                     "source_axis_local": s["axis_local"], "runtime_axis_local": t["axis_local"],
                     "zero_world_axis_dot": dot, "axis_collinear": axis_match, "sign": sign, "offset_rad": offset,
                     "source_range_rad": s["range_rad"], "runtime_range_rad": t["range_rad"],
                     "range_equal": bool(np.allclose(s["range_rad"], t["range_rad"], atol=1e-8, rtol=0)),
                     "zero_anchor_error_m": float(np.linalg.norm(np.array(s["anchor_world_at_zero"]) - t["anchor_world_at_zero"]))})
    initial_s, initial_t = sd.qpos.copy(), td.qpos.copy()
    probes = []
    # Deterministic zero and individual +0.1-rad perturbations establish more
    # than name equality. No controller/dynamics rollout is performed.
    for name in [None] + order:
        sd.qpos[:] = initial_s
        td.qpos[:] = initial_t
        if name:
            i = order.index(name)
            sd.qpos[source_dict[name]["qpos_address"]] += .1
            td.qpos[target_dict[name]["qpos_address"]] += .1 * signs[i]
        mujoco.mj_forward(source, sd)
        mujoco.mj_forward(target, td)
        pos_errors, rot_errors = [], []
        for body in tracked:
            a, b = source.body(body).id, target.body(body).id
            pos_errors.append(float(np.linalg.norm(sd.xpos[a] - td.xpos[b])))
            rot_errors.append(float(np.linalg.norm(sd.xmat[a] - td.xmat[b])))
        probes.append({"joint_perturbed": name or "zero", "delta_source_rad": .1 if name else 0.,
                       "max_tracked_body_position_error_m": max(pos_errors), "max_rotation_matrix_frobenius_error": max(rot_errors)})
    fk_p = max(x["max_tracked_body_position_error_m"] for x in probes)
    fk_r = max(x["max_rotation_matrix_frobenius_error"] for x in probes)
    verified = all(x["axis_collinear"] for x in rows) and fk_p < 1e-6 and fk_r < 1e-6
    result = {
        "schema_version": "gmr-g1-compiled-mapping/1.0", "mujoco_version": mujoco.__version__,
        "source_xml": str(args.source_xml.resolve()), "source_xml_sha256": sha(args.source_xml),
        "runtime_xml": str(args.runtime_xml.resolve()), "runtime_xml_sha256": sha(args.runtime_xml),
        "runtime_constants": str(args.runtime_constants.resolve()), "runtime_constants_sha256": sha(args.runtime_constants),
        "source_model": {"nq": source.nq, "nv": source.nv, "njnt": source.njnt, "nu": source.nu},
        "runtime_model": {"nq": target.nq, "nv": target.nv, "njnt": target.njnt, "nu": target.nu},
        "source_joint_order": [x["name"] for x in sr], "runtime_qpos_joint_order": [x["name"] for x in tr],
        "target_csv_joint_order": order, "joint_rows": rows, "fk_probe_results": probes,
        "max_fk_position_error_m": fk_p, "max_fk_rotation_matrix_error": fk_r,
        "mapping": {"verified": verified, "source_joint_names": [x["name"] for x in sr], "target_joint_names": order,
                    "permutation": permutation, "sign": signs, "offset_rad": offsets,
                    "evidence": "MuJoCo compilation, qpos addresses, zero world axes/anchors and 30 FK probes in g1-mapping-audit.json"},
        "range_difference_joints": [x["name"] for x in rows if not x["range_equal"]],
        "limitations": ["Applies only to the two recorded XML hashes and runtime CSV contract.",
                       "FK alignment is not collision/inertia/actuator/dynamics equivalence or physical tracking success.",
                       "Exported references must satisfy the runtime joint ranges; no automatic clipping is performed."],
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"mapping_verified": verified, "source_order_equals_csv": permutation == list(range(29)),
                      "max_fk_position_error_m": fk_p, "max_fk_rotation_matrix_error": fk_r,
                      "range_difference_joints": result["range_difference_joints"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
