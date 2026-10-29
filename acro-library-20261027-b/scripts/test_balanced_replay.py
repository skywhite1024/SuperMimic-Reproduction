"""Check probability, circular storage, episode boundaries, and consumed-start accounting."""
import json
from types import SimpleNamespace
import torch
from runtime import ROOT
from balanced_replay import BalancedReplay
from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner


def main():
    torch.manual_seed(27)
    names = ['new_short', 'new_long', 'old_failed', 'old_ok']
    motion = SimpleNamespace(names=names, clip_id=torch.arange(4).repeat_interleave(100), clip_end=torch.arange(1,5)*100)
    def eligible(f, horizon): return f+horizon <= motion.clip_end[motion.clip_id[f]]
    mdp = SimpleNamespace(motion=motion, valid_imagination_start=eligible)
    base = ReplayBuffer(8, 12, 'cpu')
    replay = BalancedReplay(base, mdp, 5, names[:2], {'old_failed':3})
    clip = torch.tensor([0,1,1,1,1,2,2,3,3,3,3,3])
    for t in range(11):
        entry = {key: torch.zeros_like(value[0]) for key, value in base.data.items()}
        entry['s'][:,0] = t
        entry['s'][:,1] = torch.arange(12)
        entry['s1'][:,0] = t+1
        entry['f'] = clip*100 + (t % 3)
        entry['ep_step'][:] = t % 3
        entry['done'][:] = t % 3 == 2
        replay.add(entry)
    snapshots = {key:value.clone() for key,value in base.data.items()}
    pool = replay._pool('pi')
    expected = torch.tensor([.3,.3,.3,.1])
    assert torch.allclose(pool[3],expected)
    counts = pool[1]
    assert not torch.allclose(counts/counts.sum(), expected)
    start, history = replay.sample_starts(100000,5)
    ids = motion.clip_id[start['f']]
    freq = torch.bincount(ids,minlength=4).float()/len(ids)
    assert torch.allclose(freq,expected,atol=.006),freq
    current_t = start['s'][:,0].long()
    ep = current_t % 3
    oldest_t = 3
    for j in range(5):
        back = torch.minimum(torch.minimum(torch.full_like(ep,4-j),ep),current_t-oldest_t)
        assert torch.equal(history['s'][:,j,0].long(), current_t-back)
        assert torch.equal(history['s'][:,j,1], start['s'][:,1])
    assert bool(eligible(start['f'],5).all())
    windows = replay.sample_windows(100000,4)
    wm_ids = clip[windows['s'][:,0,1].long()]
    wm_freq = torch.bincount(wm_ids,minlength=4).float()/len(wm_ids)
    assert torch.allclose(wm_freq,expected,atol=.006),wm_freq
    native_mask = torch.cumprod(torch.cat((torch.ones(100000,1),1-windows['done'][:,:-1].float()),-1),-1)
    assert torch.equal(windows['mask'],native_mask)
    assert bool((windows['s'][:,1:,0]-windows['s'][:,:-1,0]==1).all())
    assert all(torch.equal(value,snapshots[key]) for key,value in base.data.items())
    learner = SimpleNamespace(buffer=base,mdp=mdp,cfg=SimpleNamespace(hist=5,horizon=5))
    learner._sample_valid_starts = lambda n:SuperMimicLearner._sample_valid_starts(learner,n)
    replay.install(learner)
    consumed,_ = learner._sample_valid_starts(20000)
    assert len(consumed['f']) == 20000 and int(replay.counts['pi_consumed'].sum()) == 20000
    assert int(replay.counts['pi_proposed'].sum()) == 140000
    missing = replay.probabilities(torch.tensor([0,5,4,7]))
    assert torch.allclose(missing,torch.tensor([0.,.6,.3,.1]))
    try: replay.probabilities(torch.tensor([0,0,4,7]))
    except RuntimeError: pass
    else: raise AssertionError('An absent group must stop')
    # End-of-clip starts must be excluded before balancing, rather than rejected afterward.
    replay.add({key:value[-1].clone() for key,value in base.data.items()})
    base.data['f'][base._row(torch.arange(base.size)),0] = 99
    replay.cache.clear()
    start,_ = replay.sample_starts(10000,5)
    assert bool(eligible(start['f'],5).all()) and not bool((start['f']==99).any())
    result={'status':'passed','checks':['equal clip mass despite unequal occupancy','fixed old failure priority 3:1',
            '100000 actual PI and WM draws','circular histories and episode padding','native done masks',
            'storage unchanged by sampling','native valid-start consumption counted separately','missing clip renormalization',
            'absent group stops','clip-end horizon exclusion'], 'observed_pi_fraction':freq.tolist(),'observed_wm_fraction':wm_freq.tolist()}
    (ROOT/'evidence/balanced-replay-tests.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
