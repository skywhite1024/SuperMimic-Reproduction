"""Check actual native transition counting and a known physical error signal."""
from types import SimpleNamespace

import numpy as np
import torch
from supermimic.learners.wm_learner import WorldModelLearner
from supermimic.models import Dynamics
from supermimic.representations import GenCoordRepresentation

from evaluate_saved_wm_forecasts import forecast,numpy_groups,reduce_errors


def test_native_window_transition_contract():
    calls=[]
    def dynamics(rep,action,aux,context):
        calls.append(action.clone())
        return aux,rep+action[:,:1]
    obj=SimpleNamespace(dynamics=dynamics)
    for length in (8,9):
        calls.clear()
        actions=torch.arange(length,dtype=torch.float32).view(1,length,1)
        pred,aux=WorldModelLearner._rollout(obj,torch.zeros(1,1),actions,torch.zeros(1,1),torch.zeros(1,1))
        assert pred.shape[1]==len(calls)==length-1
        assert pred[0,-1,0].item()==sum(range(length-1))
        assert calls[-1].item()==length-2


def test_explicit_eight_steps_and_native_physical_metric(tmp_path):
    rep=GenCoordRepresentation(29,torch.zeros(29));rep.dt=.02
    dynamics=Dynamics(rep,29,33,torch.ones(29),torch.zeros(29),(8,)).eval()
    for parameter in dynamics.parameters():parameter.data.zero_()
    states=torch.zeros(2,9,71);states[:,:,2]=.8;states[:,:,3]=1
    states[:,1:,0]=.01
    with torch.no_grad():
        predicted,aux,mse=forecast(dynamics,rep,states,torch.zeros(2,8,29),torch.zeros(2,33))
    assert predicted.shape==(2,8,68)
    assert torch.all(aux[:,:,0]==0) # open-loop prediction, no target-state teacher forcing
    expected=np.zeros((2,8,4));expected[:,:,0]=.01**2/3
    np.testing.assert_allclose(mse.numpy(),expected,atol=1e-10,rtol=1e-6)
    full=rep.reconstruct_full_state(aux,predicted).numpy()
    np.testing.assert_allclose(numpy_groups(full,states[:,1:].numpy()),mse.numpy(),atol=1e-10)
    path=tmp_path/'raw.npy';np.save(path,mse.numpy())
    summary=reduce_errors(path,[.01,.02,.1,3.14],[dict(offset=0,windows=2,saved_steps=10)])
    np.testing.assert_allclose(summary['normalized_state_mse_by_horizon'],np.full(8,1/70),rtol=1e-6)
    native_loss,_=WorldModelLearner._compute_losses(
        SimpleNamespace(representation=rep,loss_group_weights={
            'root_pos':1/.01**2,'root_ori':1/.02**2,'joint_pos':1/.1**2,'joint_vel':1/3.14**2}),
        predicted,aux,rep.represent(states[:,1:]),rep.extract_aux_state(states[:,1:]))
    np.testing.assert_allclose(native_loss.item(),summary['mean_normalized_state_mse_over_horizons'],rtol=1e-6)
