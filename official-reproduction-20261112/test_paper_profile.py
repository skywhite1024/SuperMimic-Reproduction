"""Numerical adapter checks; CUDA graph compatibility needs a GPU trial."""

import copy
import math

import torch

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
