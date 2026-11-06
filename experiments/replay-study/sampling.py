"""Independent collection schedule and uniform-within-group training replay."""
import torch
from balanced_replay import BalancedReplay


class StudyReplay(BalancedReplay):
    def __init__(self, buffer, mdp, horizon, new_names, old_fraction, engineering=False):
        if not 0 <= old_fraction <= 1:
            raise ValueError('Invalid old replay fraction')
        self.engineering = engineering
        super().__init__(buffer, mdp, horizon, new_names, {}, new_fraction=1-old_fraction)

    def probabilities(self, counts):
        active = (self.new & (self.new_fraction > 0)) | (~self.new & (self.new_fraction < 1))
        if not self.engineering and bool(((counts == 0) & active).any()):
            missing = [name for name, bad in zip(self.names, ((counts == 0) & active).tolist()) if bad]
            raise RuntimeError(f'Uniform replay requires eligible starts for every active clip: {missing}')
        return super().probabilities(counts)

    def receipt(self):
        result = super().receipt()
        result['missing_clips'] = ('engineering only: renormalize within available clips'
                                   if self.engineering else 'stop if any active clip lacks eligible starts')
        return result


class FixedCollection:
    def __init__(self, new_names, old_fraction=0.5):
        self.new_names = set(new_names)
        self.old_fraction = old_fraction
        self.calls = 0
        self.counts = None

    def __call__(self, motion, bin_G, bin_w_avg, gamma, num_samples, rng):
        new = torch.tensor([n.removesuffix('.csv') in self.new_names for n in motion.names], device=motion.device)
        mass = torch.where(new, (1-self.old_fraction)/new.sum(), self.old_fraction/(~new).sum()).float()
        clips = rng.choice(len(motion.names), (num_samples,), p=mass)
        bootstrap = 0
        if self.calls == 0 and num_samples >= 80:
            old_ids = (~new).nonzero().flatten()
            new_ids = new.nonzero().flatten()
            assert len(old_ids) == 40 and len(new_ids) == 15
            # Forty old and forty new starts; all 55 clips represented.
            extra = rng.choice(len(new_ids), (10,), p=torch.full((len(new_ids),), 1/len(new_ids), device=motion.device))
            initial = torch.cat((old_ids, new_ids.repeat(2), new_ids[extra]))
            order = rng.random((80,)).argsort()
            clips[:80] = initial[order]
            bootstrap = 80
        lengths = motion.clip_end[clips] - motion.clip_start[clips]
        frames = motion.clip_start[clips] + (rng.random((num_samples,))*lengths.float()).long()
        if bootstrap:
            frames[:bootstrap] = motion.clip_start[clips[:bootstrap]]
        bins = motion.frame_bin[frames]
        self.calls += 1
        if self.counts is None:
            self.counts = torch.zeros(len(motion.names), dtype=torch.long, device=motion.device)
            self.names = list(motion.names)
        self.counts.add_(torch.bincount(clips, minlength=len(motion.names)))
        return frames, bins

    def receipt(self):
        return {'old_group_mass': self.old_fraction, 'rule': 'uniform clips and nonterminal frames; balanced first-reset coverage',
                'calls': self.calls, 'clip_reset_samples': dict(zip(self.names, self.counts.cpu().tolist())) if self.counts is not None else {}}
