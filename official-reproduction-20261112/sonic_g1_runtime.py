"""Released G1-mode SONIC controller for the corrected-v2 common simulator.

CPU is the reference inference provider. CUDA inference is checked against the
original CPU model on identical inputs and profiled for actual CUDA execution.
SONIC's own gains/defaults/scales replace control parameters; common G1 armatures
and all reward/termination definitions remain unchanged. No training is involved.
"""
import dataclasses
import json
import re

import numpy as np

from prepare_sonic_onnx_batch import MODEL_HASHES, digest


def configure_control(cfg, contract):
    rows = contract["joints_in_policy_order"]
    robot = cfg.scene.entities["robot"]
    old_actuators = robot.articulation.actuators
    actuators, preserved = [], {}
    for row in rows:
        name = row["name"]
        matches = [a for a in old_actuators
                   if any(re.fullmatch(pattern, name) for pattern in a.target_names_expr)]
        assert len(matches) == 1
        original = matches[0]
        preserved[name] = original.armature
        actuators.append(dataclasses.replace(original, target_names_expr=(name,),
            stiffness=row["stiffness"], damping=row["damping"],
            effort_limit=row["effort_limit"]))
    robot.articulation = dataclasses.replace(robot.articulation, actuators=tuple(actuators))
    robot.init_state = dataclasses.replace(robot.init_state,
        joint_pos={row["name"]: row["default_joint_position"] for row in rows})
    assert len(cfg.actions) == 1
    action = next(iter(cfg.actions.values()))
    assert action.use_default_offset and tuple(action.actuator_names) == (".*",)
    action.scale = {row["name"]: row["action_scale"] for row in rows}
    return preserved


def rotations(quaternions):
    """Unit wxyz quaternion to active rotation, with arbitrary leading shape."""
    q = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    assert np.allclose(norms, 1., atol=1e-5)
    w, x, y, z = np.moveaxis(q / norms, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3,3))


def verify_compiled_control(model, contract, preserved):
    import mujoco
    records = []
    for row in contract["joints_in_policy_order"]:
        joint = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"robot/"+row["name"])
        assert joint >= 0
        actuator = np.flatnonzero(model.actuator_trnid[:,0] == joint).item()
        actual = [model.actuator_gainprm[actuator,0],-model.actuator_biasprm[actuator,1],
                  -model.actuator_biasprm[actuator,2],model.actuator_forcerange[actuator,1]]
        expected = [row["stiffness"],row["stiffness"],row["damping"],row["effort_limit"]]
        assert np.allclose(actual,expected,atol=1e-9,rtol=1e-9)
        armature = float(model.dof_armature[model.jnt_dofadr[joint]])
        assert np.isclose(armature,preserved[row["name"]],atol=1e-12)
        records.append(dict(name=row["name"],actuator=int(actuator),Kp=float(actual[0]),
                            Kd=float(actual[2]),effort=float(actual[3]),common_armature=armature))
    return records


class SonicG1Controller:
    def __init__(self, models, batched, contract_path, motions, native_names, *,
                 provider="CPUExecutionProvider", profile_dir=None):
        import onnxruntime as ort
        self.contract = json.loads(contract_path.read_bytes())
        self.rows = self.contract["joints_in_policy_order"]
        names = [r["name"] for r in self.rows]
        self.policy_to_native = np.asarray([native_names.index(n) for n in names])
        self.native_to_policy = np.argsort(self.policy_to_native)
        self.defaults = np.asarray([r["default_joint_position"] for r in self.rows])
        source_names = json.loads((motions/"joint_names.json").read_bytes())
        self.reference_order = [source_names.index(n) for n in names]
        self.reference_q = np.load(motions/"joint_pos.npy", mmap_mode="r")
        self.reference_qd = np.load(motions/"joint_vel.npy", mmap_mode="r")
        self.reference_quat = np.load(motions/"root_quat_w.npy", mmap_mode="r")
        self.ends = np.load(motions/"motion_ends.npy")
        assert all(digest(models/name) == expected for name,expected in MODEL_HASHES.items())
        validation = json.loads((batched/"validation.json").read_bytes())
        for record in validation["derived_files"]:
            assert digest(batched/record["file"]) == record["sha256"]
        self.provider = provider
        self.profile_records = {}
        cuda = provider == "CUDAExecutionProvider"
        if provider not in {"CUDAExecutionProvider", "CPUExecutionProvider"}:
            raise ValueError(f"Unsupported ONNX provider: {provider}")
        if cuda:
            if provider not in ort.get_available_providers() or profile_dir is None:
                raise RuntimeError("CUDA inference requires the GPU wheel and an execution profile path")
            profile_dir.mkdir(exist_ok=False)

        def session(path, *, role=None):
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            providers = ["CPUExecutionProvider"]
            if cuda and role:
                options.enable_profiling = True
                options.profile_file_prefix = str(profile_dir / role)
                providers = [("CUDAExecutionProvider", {"device_id": "0", "use_tf32": "0"}),
                             "CPUExecutionProvider"]
            instance = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            if cuda and role and instance.get_providers()[0] != provider:
                raise RuntimeError(f"The {role} fell back to CPU during session initialization")
            instance.disable_fallback()
            return instance

        self.encoder = session(batched/"g1-encoder-batched.onnx", role="encoder")
        self.decoder = session(batched/"decoder-batched.onnx", role="decoder")
        self.original_encoder = session(models/"model_encoder.onnx")
        self.original_decoder = session(models/"model_decoder.onnx")
        self.history = self.last_action = None
        self.calls = self.original_comparisons = 0
        self.maximum_token_difference = self.maximum_action_difference = 0.
        self.token_atol = 3e-5 if cuda else 1e-6
        self.action_atol = 2e-4 if cuda else 1e-4

    def verify_cuda_execution(self):
        from collections import Counter

        for label, session in (("encoder", self.encoder), ("decoder", self.decoder)):
            profile = Path(session.end_profiling())
            events = json.loads(profile.read_bytes())
            nodes = [e.get("args", {}) for e in events if e.get("cat") == "Node"]
            counts = Counter(node.get("provider") for node in nodes if node.get("provider"))
            math = Counter(node.get("op_name") for node in nodes
                           if node.get("provider") == "CUDAExecutionProvider"
                           and node.get("op_name") in {"MatMul", "Gemm", "FusedMatMul", "Attention", "Conv"})
            if not math:
                raise RuntimeError(f"{label} GPU profile contains no CUDA compute node")
            self.profile_records[label] = {"providers": dict(counts), "cuda_compute_ops": dict(math),
                                           "profile_sha256": digest(profile)}

    def reset(self):
        self.history = self.last_action = None

    def __call__(self, states, times):
        states = np.asarray(states)
        assert states.ndim == 2 and states.shape[1] == 71
        n = len(states)
        rotation = rotations(states[:,3:7])
        if self.last_action is None:
            self.last_action = np.zeros((n,29), dtype=np.float32)
        omega = np.einsum("nji,nj->ni", rotation, states[:,39:42])
        gravity = -rotation[:,2,:]
        sensor = np.concatenate((omega,states[:,7:36][:,self.policy_to_native]-self.defaults,
            states[:,42:71][:,self.policy_to_native],self.last_action,gravity), axis=-1).astype(np.float32)
        assert sensor.shape == (n,93) and np.isfinite(sensor).all()
        self.history = (np.repeat(sensor[:,None],10,axis=1) if self.history is None else
                        np.concatenate((self.history[:,1:],sensor[:,None]),axis=1))
        motion_indices = np.searchsorted(self.ends, times, side="right")
        assert np.all(motion_indices < len(self.ends))
        future = np.minimum(times[:,None]+np.arange(10)[None]*5,self.ends[motion_indices,None]-1)
        features = np.zeros((n,1762),dtype=np.float32)
        features[:,4:294] = self.reference_q[future][..., self.reference_order].reshape(n,290)
        features[:,294:584] = self.reference_qd[future][..., self.reference_order].reshape(n,290)
        relative = np.einsum("nji,nkjl->nkil",rotation,rotations(self.reference_quat[future]))
        features[:,601:661] = relative[...,:2].reshape(n,60)
        tokens = self.encoder.run(None,{"obs_dict":features})[0].reshape(n,64)
        h = self.history
        decoder_input = np.concatenate((tokens,h[:,:,:3].reshape(n,-1),h[:,:,3:32].reshape(n,-1),
            h[:,:,32:61].reshape(n,-1),h[:,:,61:90].reshape(n,-1),h[:,:,90:].reshape(n,-1)),axis=1)
        assert decoder_input.shape == (n,994)
        self.last_encoder_input = features
        self.last_decoder_input = decoder_input
        raw = self.decoder.run(None,{"obs_dict":decoder_input})[0]
        assert raw.shape == (n,29) and np.isfinite(raw).all()
        if self.calls == 0 and self.provider == "CUDAExecutionProvider":
            self.verify_cuda_execution()
        if self.calls % 1000 == 0:
            for i in sorted({0,n-1}):
                original_token = self.original_encoder.run(None,{"obs_dict":features[i:i+1]})[0].reshape(1,64)
                original_input = decoder_input[i:i+1].copy()
                original_input[:,:64] = original_token
                original_action = self.original_decoder.run(None,{"obs_dict":original_input})[0]
                self.maximum_token_difference = max(self.maximum_token_difference,
                    float(np.max(np.abs(tokens[i:i+1]-original_token))))
                self.maximum_action_difference = max(self.maximum_action_difference,
                    float(np.max(np.abs(raw[i:i+1]-original_action))))
                assert np.allclose(tokens[i:i+1],original_token,atol=self.token_atol,rtol=self.token_atol)
                assert np.allclose(raw[i:i+1],original_action,atol=self.action_atol,rtol=1e-4)
                self.original_comparisons += 1
        self.calls += 1
        self.last_action = raw.copy()
        return np.clip(raw,-20.,20.)[:,self.native_to_policy]
