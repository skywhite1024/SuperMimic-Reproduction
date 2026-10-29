"""Use the upstream CLI unchanged, adding real-update and resume evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

import torch
import unisupermimic.scripts.train_supermimic as entry


def snapshot(learner):
    def digest(module):
        h = hashlib.sha256()
        finite = True
        for name, value in module.named_parameters():
            x = value.detach().contiguous()
            finite &= bool(torch.isfinite(x).all())
            h.update(name.encode())
            h.update(x.cpu().view(torch.uint8).numpy().tobytes())
        state_hash=hashlib.sha256()
        for name,value in module.state_dict().items():
            x=value.detach().contiguous()
            finite &= bool(torch.isfinite(x).all())
            state_hash.update(name.encode())
            state_hash.update(x.cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
        return {"sha256_parameters": h.hexdigest(), "sha256_state":state_hash.hexdigest(), "finite": finite}

    def steps(optimizer):
        values = [int(s["step"].item()) for s in optimizer.state.values() if "step" in s]
        return {"min": min(values, default=0), "max": max(values, default=0), "states": len(values)}

    return {"iteration": learner.iteration, "sim_steps": learner.sim_steps,
            "wm": digest(learner.wm), "policy": digest(learner.pi), "critic": digest(learner.critic),
            "wm_optimizer": steps(learner.opt_wm), "policy_optimizer": steps(learner.opt_pi),
            "critic_optimizer": steps(learner.opt_v)}


class AuditedLearner(entry.SuperMimicLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_before = snapshot(self)
        self.audit_resume = None
        self.clip_transition_counts = torch.zeros(len(self.mdp.motion.names), dtype=torch.int64, device=self.device)

    def collect(self, steps):
        start = self.buffer.ptr
        result = super().collect(steps)
        if steps:
            assert steps <= self.buffer.T
            indices = (start + torch.arange(steps, device=self.device)) % self.buffer.T
            frames = self.buffer.data["f"][indices].reshape(-1)
            clip_ids = self.mdp.motion.clip_id[frames]
            self.clip_transition_counts += torch.bincount(clip_ids, minlength=len(self.mdp.motion.names))
        return result

    def load(self, path):
        super().load(path)
        self.audit_before = snapshot(self)
        self.audit_resume = {"checkpoint": str(path),
                             "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        print("SUPERMIMIC_RESUME_LOADED " + json.dumps(self.audit_before), flush=True)

    def wm_update(self):
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

    def train(self, resume=None):
        started = time.monotonic()
        summary = super().train(resume)
        after = snapshot(self)
        proof = {"before": self.audit_before, "after": after,
                 "resume": self.audit_resume, "summary": summary,
                 "entry_seconds": time.monotonic() - started,
                 "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                 "observations": self.mdp.obs_dim,
                 "clips": len(self.mdp.motion.names), "backend": self.backend,
                 "amp": self.cfg.amp, "compile": self.cfg.compile,
                 "first_world_model_update_seconds":self.first_wm_update_seconds,
                 "first_policy_update_seconds":self.first_policy_update_seconds,
                 "first_update_timing_includes_compilation_if_enabled":True,
                 "transitions_by_clip": dict(zip(self.mdp.motion.names, self.clip_transition_counts.cpu().tolist())),
                 "replay_in_checkpoint": False,
                 "resume_semantics": "optimizer/model restored; replay rewarmed; not bit-exact RNG continuation"}
        proof["all_models_finite"] = all(after[k]["finite"] for k in ("wm", "policy", "critic"))
        proof["all_models_updated"] = all(
            after[k]["sha256_parameters"] != self.audit_before[k]["sha256_parameters"]
            for k in ("wm", "policy", "critic"))
        proof["all_optimizers_advanced"] = all(
            after[k]["max"] > self.audit_before[k]["max"]
            for k in ("wm_optimizer", "policy_optimizer", "critic_optimizer"))
        proof["passed"] = (proof["all_models_finite"] and proof["all_models_updated"]
                           and proof["all_optimizers_advanced"]
                           and after["iteration"] > self.audit_before["iteration"]
                           and after["sim_steps"] > self.audit_before["sim_steps"])
        destination = self.log_dir / f"update-proof-{after['iteration']:06d}.json"
        destination.write_text(json.dumps(proof, indent=2) + "\n")
        for key,opt in (("wm","wm_optimizer"),("policy","policy_optimizer"),("critic","critic_optimizer")):
            print("SUPERMIMIC_REAL_UPDATE "+json.dumps({"model":key,"before":self.audit_before[key]["sha256_parameters"],
                  "after":after[key]["sha256_parameters"],"optimizer_before":self.audit_before[opt],"optimizer_after":after[opt],"finite":after[key]["finite"]}),flush=True)
        print("SUPERMIMIC_UPDATE_SUMMARY "+json.dumps({"passed":proof["passed"],"start_iteration":self.audit_before["iteration"],
              "end_iteration":after["iteration"],"first_wm_update_seconds":self.first_wm_update_seconds,
              "first_policy_update_seconds":self.first_policy_update_seconds,"includes_compile":self.cfg.compile,"entry_seconds":proof["entry_seconds"]}),flush=True)
        print("SUPERMIMIC_UPDATE_PROOF_SAVED "+str(destination),flush=True)
        if not proof["passed"]:
            raise RuntimeError("Training finished without verified world-model, BPTT-policy and critic updates")
        return summary


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one allocated CUDA GPU")
    torch.cuda.reset_peak_memory_stats()
    entry.SuperMimicLearner = AuditedLearner
    entry.hydra_main()
