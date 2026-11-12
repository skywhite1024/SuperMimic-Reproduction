"""Numerical adapter checks; CUDA graph compatibility needs a GPU trial."""

import copy
import math

import torch
import pytest

from paper_profile import build_adam_optimizers, cosine_lr, set_learning_rate_in_place


def test_all_parameters_follow_reference_adam_with_changing_lr():
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(),
                                torch.nn.Linear(5, 4), torch.nn.Tanh(),
                                torch.nn.Linear(4, 2)).double()
    reference = copy.deepcopy(model)
    # Include the same module twice to exercise shared-parameter handling.
    optimizers = build_adam_optimizers([model, model], 2e-4, betas=(0.9, 0.95))
    optimizer = optimizers[0]
    params = optimizer.param_groups[0]["params"]
    assert {id(p) for p in params} == {id(p) for p in model.parameters()}
    assert len(params) == len(list(model.parameters()))
    expected = torch.optim.Adam(reference.parameters(), lr=2e-4, betas=(0.9, 0.95))
    lr_tensor = optimizer.param_groups[0]["lr"]
    lr_pointer = lr_tensor.data_ptr()
    inputs = torch.randn(7, 3, dtype=torch.float64)
    targets = torch.randn(7, 2, dtype=torch.float64)
    for step in (0, 1, 125000, 250000, 320000):
        set_learning_rate_in_place(optimizers, cosine_lr(step))
        assert optimizer.param_groups[0]["lr"] is lr_tensor
        assert lr_tensor.data_ptr() == lr_pointer
        # Compare to standard scalar-lr Adam at the same float32-rounded LR.
        expected.param_groups[0]["lr"] = lr_tensor.item()
        for net, opt in ((model, optimizer), (reference, expected)):
            opt.zero_grad(set_to_none=True)
            (net(inputs) - targets).square().mean().backward()
            opt.step()
        for actual, wanted in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, wanted, rtol=1e-9, atol=1e-10)


def test_schedule_endpoints_and_monotonicity():
    assert math.isclose(cosine_lr(0), 2e-4)
    assert math.isclose(cosine_lr(125000), 1.03e-4)
    assert math.isclose(cosine_lr(250000), 6e-6)
    assert cosine_lr(320000) == cosine_lr(250000)
    values = [cosine_lr(step) for step in range(0, 320001, 1000)]
    assert all(a >= b for a, b in zip(values, values[1:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph needs a GPU")
def test_captured_adam_uses_changed_learning_rate():
    """A replay must consume the new LR, rather than the capture-time value."""
    model = torch.nn.Linear(2, 1, bias=False, device="cuda")
    with torch.no_grad():
        model.weight.fill_(0.5)
    reference = copy.deepcopy(model)
    optimizers = build_adam_optimizers([model], 2e-4, device="cuda")
    expected = build_adam_optimizers([reference], 2e-4, device="cuda")
    model.weight.grad = torch.ones_like(model.weight)
    reference.weight.grad = torch.ones_like(reference.weight)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            optimizers[0].step()
            expected[0].step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        optimizers[0].step()
    for lr in (2e-4, 6e-6, 1.03e-4):
        set_learning_rate_in_place(optimizers, lr)
        set_learning_rate_in_place(expected, lr)
        graph.replay()
        expected[0].step()
        torch.cuda.synchronize()
        torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=1e-7)
