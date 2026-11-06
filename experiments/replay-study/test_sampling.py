"""Check group mass, missing-clip handling, and the collection bootstrap."""
import types
import unittest
import torch
import study_runtime
from sampling import FixedCollection, StudyReplay


class Rng:
    def __init__(self): self.g = torch.Generator().manual_seed(17)
    def choice(self, n, shape, *, p):
        return torch.multinomial(p, shape[0], replacement=True, generator=self.g)
    def random(self, shape): return torch.rand(shape, generator=self.g)


class SamplingTest(unittest.TestCase):
    def setUp(self):
        self.new = [f'new{i}' for i in range(15)]
        self.motion = types.SimpleNamespace(names=self.new+[f'old{i}' for i in range(40)], device=torch.device('cpu'))
        self.motion.clip_start = torch.arange(55)*100
        self.motion.clip_end = self.motion.clip_start+99
        self.motion.frame_bin = torch.arange(5500)//50
        self.mdp = types.SimpleNamespace(motion=self.motion)
        self.buffer = types.SimpleNamespace(device=torch.device('cpu'))

    def test_uniform_mass_ignores_available_entry_counts(self):
        for rho in [0., .25, .5, .75]:
            replay = StudyReplay(self.buffer, self.mdp, 32, self.new, rho)
            p = replay.probabilities(torch.arange(1,56))
            self.assertTrue(torch.allclose(p[:15], torch.full((15,), (1-rho)/15)))
            self.assertTrue(torch.allclose(p[15:], torch.full((40,), rho/40)))

    def test_missing_active_clip_is_not_silently_renormalized(self):
        replay = StudyReplay(self.buffer, self.mdp, 32, self.new, .5)
        counts = torch.ones(55, dtype=torch.long); counts[0] = 0
        with self.assertRaises(RuntimeError): replay.probabilities(counts)

    def test_first_collection_covers_all_clips_and_excludes_terminal_frame(self):
        sampler = FixedCollection(self.new)
        frames, bins = sampler(self.motion, None, None, .99, 128, Rng())
        clips = frames//100
        self.assertEqual(len(set(clips[:80].tolist())),55)
        self.assertEqual(int((clips[:80]<15).sum()),40)
        self.assertTrue(bool((frames < self.motion.clip_end[clips]).all()))
        self.assertTrue(bool((bins == self.motion.frame_bin[frames]).all()))


if __name__ == '__main__': unittest.main()
