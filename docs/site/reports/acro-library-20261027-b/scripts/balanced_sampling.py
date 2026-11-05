"""Fixed per-clip rehearsal mass, with native performance priority only inside each clip."""
import torch
from unisupermimic.tasks.motion_tracking.g1.supermimic import manager_terms
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import BIN_FRAMES


class BalancedSampling:
    def __init__(self,new_names,new_fraction=.6):
        self.new_names=set(new_names);self.new_fraction=new_fraction;self.counts=None;self.names=None

    def probabilities(self,motion,base):
        names=list(motion.names);new=torch.tensor([name.removesuffix('.csv') in self.new_names for name in names],device=base.device)
        assert bool(new.any()) and bool((~new).any())
        mass=torch.where(new,self.new_fraction/new.sum(),(1-self.new_fraction)/(~new).sum()).to(base.dtype)
        totals=torch.zeros(len(names),device=base.device,dtype=base.dtype).scatter_add_(0,motion.bin_clip,base)
        p=base*mass[motion.bin_clip]/totals[motion.bin_clip]
        assert torch.isfinite(p).all() and abs(float(p.sum())-1)<1e-5
        return p,mass

    def __call__(self,motion,bin_G,bin_w_avg,gamma,num_samples,rng):
        base=manager_terms.start_bin_probabilities(bin_G,bin_w_avg,gamma)
        probabilities,mass=self.probabilities(motion,base)
        bins=rng.choice(motion.num_bins,(num_samples,),p=probabilities)
        start=motion.bin_start[bins];end=torch.minimum(start+BIN_FRAMES,motion.clip_end[motion.bin_clip[bins]])
        frames=start+(rng.random((num_samples,))*(end-start).float()).long()
        if self.counts is None:self.counts=torch.zeros(len(motion.names),dtype=torch.long,device=base.device);self.names=list(motion.names);self.mass=mass
        self.counts.add_(torch.bincount(motion.bin_clip[bins],minlength=len(motion.names)))
        return frames,bins

    def receipt(self):
        return {'new_library_mass':self.new_fraction,'lafan_rehearsal_mass':1-self.new_fraction,
                'within_clip_sampling':'native past-return priority, normalized independently inside each clip',
                'clip_reset_samples':dict(zip(self.names,self.counts.cpu().tolist())) if self.counts is not None else {},
                'clip_prior_mass':dict(zip(self.names,self.mass.cpu().tolist())) if self.counts is not None else {}}
