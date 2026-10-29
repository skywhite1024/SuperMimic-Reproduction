"""Balance actual replay starts without changing stored transitions or native masks."""
import torch


class BalancedReplay:
    def __init__(self, buffer, mdp, horizon, new_names, old_priority, new_fraction=.6):
        self.base, self.mdp, self.horizon = buffer, mdp, horizon
        self.names = list(mdp.motion.names)
        self.new = torch.tensor([n.removesuffix('.csv') in set(new_names) for n in self.names], device=buffer.device)
        if not bool(self.new.any()) or not bool((~self.new).any()):
            raise ValueError('Both reference groups are required')
        self.priority = torch.tensor([old_priority.get(n.removesuffix('.csv'), 1.) for n in self.names], device=buffer.device)
        if not bool((self.priority > 0).all()):
            raise ValueError('Replay priorities must be positive')
        self.new_fraction = new_fraction
        self.version, self.cache = 0, {}
        self.counts = {key: torch.zeros(len(self.names), dtype=torch.long, device=buffer.device)
                       for key in ['wm_starts', 'wm_unmasked_transitions', 'pi_proposed', 'pi_consumed']}
        self.pool_builds = 0
        self.min_available = torch.full((len(self.names),), 2**31-1, dtype=torch.long, device=buffer.device)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def add(self, transition):
        self.base.add(transition)
        self.version += 1
        self.cache.clear()

    def probabilities(self, counts):
        available = counts > 0
        new_weights = (available & self.new).float()
        old_weights = (available & ~self.new).float() * self.priority
        if not bool(new_weights.sum() > 0) or not bool(old_weights.sum() > 0):
            raise RuntimeError('Replay lacks eligible starts in one group; stop instead of silently changing group mass')
        return self.new_fraction * new_weights / new_weights.sum() + (1-self.new_fraction) * old_weights / old_weights.sum()

    def _pool(self, kind, k=1):
        key = (kind, k)
        if key in self.cache:
            return self.cache[key]
        count = self.size-k+1 if kind == 'wm' else self.size
        if count < 1:
            raise ValueError('Insufficient replay for requested window')
        ages = torch.arange(count, device=self.device)
        frames = self.data['f'][self._row(ages)]
        flat = torch.arange(frames.numel(), device=self.device)
        ids = self.mdp.motion.clip_id[frames].flatten()
        if kind == 'pi':
            valid = self.mdp.valid_imagination_start(frames, self.horizon).flatten()
            flat, ids = flat[valid], ids[valid]
        order = torch.argsort(ids)
        indices = flat[order]
        counts = torch.bincount(ids, minlength=len(self.names))
        offsets = counts.cumsum(0)-counts
        probabilities = self.probabilities(counts)
        pool = (indices, counts, offsets, probabilities)
        self.cache[key] = pool
        self.pool_builds += 1
        self.min_available = torch.minimum(self.min_available, counts)
        return pool

    def _draw(self, B, kind, k=1):
        indices, counts, offsets, probabilities = self._pool(kind, k)
        clips = torch.multinomial(probabilities, B, replacement=True)
        within = (torch.rand(B, device=self.device) * counts[clips]).long()
        flat = indices[offsets[clips]+within]
        return flat // self.N, flat % self.N, clips

    def _record(self, key, clips):
        self.counts[key].add_(torch.bincount(clips.flatten(), minlength=len(self.names)))

    def sample_windows(self, B, k):
        if not 1 <= k <= self.size:
            raise ValueError('Invalid replay window length')
        age, env, clips = self._draw(B, 'wm', k)
        rows = self._row(age[:, None]+torch.arange(k, device=self.device))
        envs = env[:, None].expand_as(rows)
        out = {key: self.data[key][rows, envs] for key in ['s', 's1', 'a', 'xi', 'done']}
        done = out['done'].float()
        out['mask'] = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1-done[:, :-1]), -1), -1)
        self._record('wm_starts', clips)
        transition_clips = self.mdp.motion.clip_id[self.data['f'][rows, envs]]
        self._record('wm_unmasked_transitions', transition_clips[out['mask'].bool()])
        return out

    def sample_starts(self, B, hist):
        if hist < 1:
            raise ValueError('History length must be positive')
        age, env, clips = self._draw(B, 'pi')
        row = self._row(age)
        ep_step = self.data['ep_step'][row, env].long()
        back = torch.arange(hist-1, -1, -1, device=self.device)[None, :]
        back = torch.minimum(torch.minimum(back, ep_step[:, None]), age[:, None])
        rows = self._row(age[:, None]-back)
        envs = env[:, None].expand_as(rows)
        keys = ['s', 'a_prev', 'xi', 'f']
        self._record('pi_proposed', clips)
        return ({key: self.data[key][row, env] for key in keys},
                {key: self.data[key][rows, envs] for key in keys})

    def install(self, learner):
        assert learner.buffer is self.base
        learner.buffer = self
        native = learner._sample_valid_starts
        def sampled(batch_size):
            start, history = native(batch_size)
            self._record('pi_consumed', self.mdp.motion.clip_id[start['f']])
            return start, history
        learner._sample_valid_starts = sampled

    def receipt(self):
        rows = {}
        for key, counts in self.counts.items():
            total = int(counts.sum())
            rows[key] = {'count': total, 'new_fraction': float(counts[self.new].sum())/total if total else None,
                         'by_clip': dict(zip(self.names, counts.cpu().tolist()))}
        return {'rule': 'group then clip then uniform eligible replay entry; native history padding and done masks',
                'new_group_mass': self.new_fraction, 'old_group_mass': 1-self.new_fraction,
                'old_clip_priority': dict(zip(self.names, self.priority.cpu().tolist())),
                'within_new_group': 'equal available clip mass', 'missing_clips': 'renormalize within group; stop if either group absent',
                'policy_eligibility': 'complete native imagination horizon', 'pool_builds': self.pool_builds,
                'minimum_available_starts_by_clip': dict(zip(self.names, self.min_available.cpu().tolist())),
                'observed_sampling': rows, 'gradient_fraction_inferred': False}
