"""Upgrade only our orchestration: preserve resume input and export fixed videos."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def save_checked(name, code):
    ast.parse(code)
    p=ROOT/name
    if p.exists():
        assert p.read_text()==code or hashlib.sha256(p.read_text().encode()).hexdigest()=="077f88208d300d6a7f8a7ed8ba336fd75ec7d9cb9fb0fdf3f498618f483516db",name
    p.write_text(code,newline="\n")

train=(ROOT/"train_audited.py").read_bytes()
assert hashlib.sha256(train).hexdigest()=="22e943f64743388e147c9b1ddc7d2f128b32dd46df0b8400a2bcc042a1970616"
train=train.decode()
needle='        return {"sha256_parameters": h.hexdigest(), "finite": finite}'
assert train.count(needle)==1
train=train.replace(needle,'''        state_hash=hashlib.sha256()
        for name,value in module.state_dict().items():
            x=value.detach().contiguous()
            finite &= bool(torch.isfinite(x).all())
            state_hash.update(name.encode())
            state_hash.update(x.cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
        return {"sha256_parameters": h.hexdigest(), "sha256_state":state_hash.hexdigest(), "finite": finite}''')
needle='    def train(self, resume=None):'
assert train.count(needle)==1
train=train.replace(needle,'''    def wm_update(self):
        if hasattr(self,"first_wm_update_seconds"): return super().wm_update()
        torch.cuda.synchronize()
        started=time.monotonic()
        result=super().wm_update()
        torch.cuda.synchronize()
        self.first_wm_update_seconds=time.monotonic()-started
        return result

    def policy_update(self):
        if hasattr(self,"first_policy_update_seconds"): return super().policy_update()
        torch.cuda.synchronize()
        started=time.monotonic()
        result=super().policy_update()
        torch.cuda.synchronize()
        self.first_policy_update_seconds=time.monotonic()-started
        return result

'''+needle)
train=train.replace('"amp": self.cfg.amp, "compile": self.cfg.compile,','''"amp": self.cfg.amp, "compile": self.cfg.compile,
                 "first_world_model_update_seconds":self.first_wm_update_seconds,
                 "first_policy_update_seconds":self.first_policy_update_seconds,
                 "first_update_timing_includes_compilation_if_enabled":True,''')
train=train.replace('        print("SUPERMIMIC_UPDATE_PROOF " + json.dumps(proof), flush=True)','''        for key,opt in (("wm","wm_optimizer"),("policy","policy_optimizer"),("critic","critic_optimizer")):
            print("SUPERMIMIC_REAL_UPDATE "+json.dumps({"model":key,"before":self.audit_before[key]["sha256_parameters"],
                  "after":after[key]["sha256_parameters"],"optimizer_before":self.audit_before[opt],"optimizer_after":after[opt],"finite":after[key]["finite"]}),flush=True)
        print("SUPERMIMIC_UPDATE_SUMMARY "+json.dumps({"passed":proof["passed"],"start_iteration":self.audit_before["iteration"],
              "end_iteration":after["iteration"],"first_wm_update_seconds":self.first_wm_update_seconds,
              "first_policy_update_seconds":self.first_policy_update_seconds,"includes_compile":self.cfg.compile,"entry_seconds":proof["entry_seconds"]}),flush=True)
        print("SUPERMIMIC_UPDATE_PROOF_SAVED "+str(destination),flush=True)''')
save_checked("train_audited_v2.py",train)
evaluation=(ROOT/"evaluate_audited.py").read_bytes()
assert hashlib.sha256(evaluation).hexdigest()=="c817c1f89a26e51956ac48464bd07348b7b7d2fb19f25f762efca3f651558287"
evaluation=evaluation.decode().replace('        print("SUPERMIMIC_EVALUATION " + json.dumps(result), flush=True)','''        print("SUPERMIMIC_EVAL_SUMMARY "+json.dumps({k:result[k] for k in ("norm_return","norm_return_framew","n_failed","wall_seconds","checkpoint_sha256")}),flush=True)
        for name,row in result["per_clip"].items():
            print("SUPERMIMIC_EVAL_CLIP "+json.dumps({"clip":name,**row}),flush=True)''')
save_checked("evaluate_audited_v2.py",evaluation)
original=(ROOT/"run_gpu.py").read_bytes()
assert hashlib.sha256(original).hexdigest()=="bfef4f7cd4ec306bb8f4cdb4c4fb61e79483fbdd823b5884b2e673a15968038d"
code=original.decode()
code=code.replace('ROOT/"train_audited.py"','ROOT/"train_audited_v2.py"')
code=code.replace('ROOT/"evaluate_audited.py"','ROOT/"evaluate_audited_v2.py"')
code=code.replace('    print("SUPERMIMIC_GPU_PREFLIGHT_PASS "+json.dumps(metadata),flush=True)',
                  '    print("SUPERMIMIC_GPU_PREFLIGHT_PASS "+json.dumps({k:v for k,v in metadata.items() if k!="resources"}),flush=True)\n    print("SUPERMIMIC_GPU_RESOURCE_START "+json.dumps(metadata["resources"]),flush=True)')
code=code.replace("import subprocess\n", "import subprocess\nimport shutil\n", 1)
needle='        run("resume",[str(ROOT/"train_audited_v2.py"),'
assert code.count(needle)==1
code=code.replace(needle,'        shutil.copy2(smoke/"last.pt",smoke/"resume-source.pt")\n'+needle)
code=code.replace('"algo.checkpoint=last.pt"','"algo.checkpoint=resume-source.pt"')
needle='    print("SUPERMIMIC_BASELINE_COMPLETE",flush=True)'
assert code.count(needle)==1
video='''    os.environ["MUJOCO_GL"]="osmesa"
    os.environ["PYOPENGL_PLATFORM"]="osmesa"
    evaluation=json.loads((ROOT/"runs/baseline-evaluation.json").read_text())
    selected=["jumps1_subject1","fallAndGetUp1_subject1"]
    selected += [n for n in evaluation["failed"][:1] if n not in selected]
    for index,clip in enumerate(selected):
        start=0.0 if index<2 else max(0.0,evaluation["per_clip"][clip]["steps"]/50.0-10.0)
        run("video-"+str(index),["-m","unisupermimic.tasks.motion_tracking.g1.supermimic.render",
             "--backend","mjwarp","--ckpt",str(baseline/"ckpt_012000.pt"),"--clip",clip,
             "--start",str(start),"--seconds","15","--global-ghost",
             "--output",str(ROOT/"runs/videos"/(clip+".mp4"))],deadline,360)
'''
code=code.replace(needle,video+needle)
save_checked("run_gpu_v2.py",code)
receipt={"source_sha256":hashlib.sha256(original).hexdigest(),
         "runtime_v2_sha256":hashlib.sha256(code.encode()).hexdigest(),
         "changes":["preserve full resume-source.pt before load","state-buffer identity and real parameter update checks","first-update timings include compilation when enabled","short per-model and per-clip log records","OSMesa global-reference videos plus first failure if any"],
         "train_audit_v2_sha256":hashlib.sha256(train.encode()).hexdigest(),
         "eval_audit_v2_sha256":hashlib.sha256(evaluation.encode()).hexdigest(),
         "upstream_training_algorithm_changed":False}
(ROOT/"evidence/runtime-v2.json").write_text(json.dumps(receipt,indent=2)+chr(10))
print("SUPERMIMIC_RUNTIME_V2_VERIFIED "+json.dumps(receipt),flush=True)
