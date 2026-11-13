"""Check benchmark timing boundaries and actual published receipt identities."""

import copy
import json
from pathlib import Path

import pytest

import benchmark_local_checkpoint as benchmark


def test_warmup_and_synchronization_are_outside_measured_intervals(monkeypatch):
    events = []
    clock = iter([100.,104.,200.,206.])
    def now():
        value = next(clock)
        events.append(('clock',value))
        return value
    monkeypatch.setattr(benchmark.time,'perf_counter',now)
    durations = benchmark.timed_repeats(lambda:events.append('step'),
        lambda:events.append('sync'),warmup=2,steps=3,repeats=2)
    assert durations == [4.,6.]
    assert events == ['step','step','sync','sync',('clock',100.),'step','step','step',
        'sync',('clock',104.),'sync',('clock',200.),'step','step','step','sync',('clock',206.)]


def test_real_lafan_receipts_bind_same_model_and_reject_mismatched_source():
    folder = Path(__file__).parent
    read = lambda name:json.loads((folder/name).read_text())
    train = read('lafan40-paper-16k-result.json')
    resolved = read('lafan40-paper-16k-resolved-config.json')
    protocol = read('manual-reset-v2-lafan40-protocol.json')
    expected = benchmark.validate_receipts(train,resolved,protocol)
    assert expected['root_quat_w.npy'] == '9934171fce73559ec04d63db2bafee95575cd46196fe39aea425d677eae5303e'
    changed = copy.deepcopy(protocol)
    changed['training']['checkpoints'][0]['sha256'] = '0'*64
    with pytest.raises(ValueError,match='same completed40clip model'):
        benchmark.validate_receipts(train,resolved,changed)
    changed = copy.deepcopy(protocol)
    changed['motion_files']['unexpected.npy'] = changed['motion_files'].pop('root_quat_w.npy')
    with pytest.raises(ValueError,match='same completed40clip model'):
        benchmark.validate_receipts(train,resolved,changed)
