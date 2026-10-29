"""Explicit new 101D policy variant; the original 99D baseline remains unchanged."""
import copy
import json
import os
import torch
from runtime import ROOT, CHECKPOINT
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP
from unisupermimic.rl.algos.supermimic.geometry import yaw_mat
from unisupermimic.rl.algos.supermimic.networks import Policy


class GlobalPositionMDP(MDP):
    obs_dim = 101
    max_step_reward = .13

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_noise = torch.cat([self.obs_noise, self.obs_noise.new_zeros(2)])
        self.obs_scale = torch.cat([self.obs_scale, self.obs_scale.new_ones(2)])

    def obs(self, state, previous_action, context, frames):
        basic = super().obs(state, previous_action, context, frames)
        error = (yaw_mat(state[1]).transpose(-1,-2) @ (self.motion.root_p[frames] - state[0])[...,None]).squeeze(-1)
        return torch.cat([basic,error[...,:2].clamp(-3,3)],-1)

    def reward(self, state, frames, action, previous_action, terms=False):
        base = super().reward(state, frames, action, previous_action, terms=terms)
        error = ((state[0][...,:2] - self.motion.root_p[frames][...,:2])**2).sum(-1)
        extra = 1.5 * torch.exp(-error / .75**2) * .02
        if terms:
            reward, components = base
            return reward + extra, {**components, 'fixed_world_xy':extra}
        return base + extra


def expand_decoder(value, hist=10, z=32):
    out = value.new_zeros(value.shape[0], z+hist*101)
    out[:,:z] = value[:,:z]
    for i in range(hist):
        out[:,z+i*101:z+i*101+99] = value[:,z+i*99:z+(i+1)*99]
    return out


def convert_checkpoint(parent, output):
    parent = torch.load(parent,map_location='cpu',weights_only=True)
    result = copy.deepcopy(parent)
    args = parent['args']
    old = Policy(99,730,hist=args['hist'],z_dim=args['z_dim'],hidden=tuple(args['policy_hidden']))
    old.load_state_dict(parent['pi']);old.eval()
    new_state=result['pi']
    new_state['dec.0.weight']=expand_decoder(new_state['dec.0.weight'],args['hist'],args['z_dim'])
    new_state['obs_norm.mean']=torch.cat([new_state['obs_norm.mean'],torch.zeros(2)])
    new_state['obs_norm.var']=torch.cat([new_state['obs_norm.var'],torch.ones(2)])
    names=[n for n,_ in old.named_parameters()]
    param_ids=parent['opt_pi']['param_groups'][0]['params']
    assert len(names)==len(param_ids)
    decoder_id=param_ids[names.index('dec.0.weight')]
    for key in ['exp_avg','exp_avg_sq']:
        result['opt_pi']['state'][decoder_id][key]=expand_decoder(result['opt_pi']['state'][decoder_id][key],args['hist'],args['z_dim'])
    new=Policy(101,730,hist=args['hist'],z_dim=args['z_dim'],hidden=tuple(args['policy_hidden']))
    new.load_state_dict(new_state);new.eval()
    torch.manual_seed(921)
    with torch.no_grad():
        obs=torch.randn(5,args['hist'],99)*.1;goals=torch.randn(5,730)*.1
        augmented=torch.cat([obs,torch.randn(5,args['hist'],2)],-1)
        before=old(obs,goals);after=new(augmented,goals)
        error=float((before-after).abs().max())
        assert error<2e-5,error
    result['variant']={'name':'global-position-101d','parent_observation_dim':99,'observation_dim':101,'new_observation':'fixed-world root XY error expressed in robot yaw frame, clipped to +/-3 m','new_weights_and_optimizer_moments_zero_initialized':True,'initial_policy_identity_max_abs_error':error,'old_optimizer_counters_preserved':True,'replay_and_simulator_bit_exact_restore':False}
    temporary=output.with_suffix('.tmp');output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,temporary);temporary.replace(output)
    receipt=output.with_suffix('.json');receipt.write_text(json.dumps(result['variant'],indent=2)+'\n')
    print('GLOBAL_POSITION_CHECKPOINT_IDENTITY_PASS '+json.dumps(result['variant']),flush=True)
    return output


if __name__=='__main__':
    convert_checkpoint(__import__('pathlib').Path(os.environ.get('SUPERMIMIC_FULL',str(ROOT.parent/'implementation/continuation-20261025-a/downloaded/policy/full_016000.pt'))),ROOT/'runs/global-position/initial-full.pt')
