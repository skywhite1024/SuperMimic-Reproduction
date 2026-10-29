"""Training-only frozen teacher anchors and an explicit consistent action domain."""
import hashlib
from types import MethodType
import torch
from unisupermimic.rl.algos.supermimic.networks import Policy


def module_digest(module):
    digest=hashlib.sha256()
    for key,value in sorted(module.state_dict().items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def install_action_bound(policy):
    """Keep state_dict/optimizer parameter identities; bound all collection/imagined uses."""
    assert not hasattr(policy,'_native_unbounded_forward')
    policy._native_unbounded_forward=policy.forward
    device=next(policy.parameters()).device
    policy._bound_counters={key:torch.zeros((),device=device,dtype=torch.long) for key in ['grad_calls','no_grad_calls','overbound_elements']}
    policy._bound_peak=torch.zeros((),device=device)
    # Save the caller's grad context before entering no_grad for the bookkeeping.
    def counted(self,history,goal):
        grad_enabled=torch.is_grad_enabled()
        raw=self._native_unbounded_forward(history,goal)
        with torch.no_grad():
            self._bound_counters['grad_calls' if grad_enabled else 'no_grad_calls'].add_(1)
            self._bound_counters['overbound_elements'].add_((raw.abs()>20).sum())
            self._bound_peak.copy_(torch.maximum(self._bound_peak,raw.detach().abs().amax().float()))
        return raw.clamp(-20,20)
    policy.forward=MethodType(counted,policy)


def load_teacher(checkpoint,device):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);args=saved['args']
    obs=saved['pi']['obs_norm.mean'].numel();goal=saved['pi']['goal_norm.mean'].numel()
    hidden=args.get('test_hidden_dims') or args['policy_hidden']
    policy=Policy(obs,goal,hist=int(args['hist']),z_dim=int(args['z_dim']),num_joints=29,hidden=tuple(hidden)).to(device)
    policy.load_state_dict(saved['pi']);policy.eval()
    for parameter in policy.parameters():parameter.requires_grad_(False)
    for norm in [policy.obs_norm,policy.goal_norm]:norm.max_count=0
    return policy


class RetentionMixin:
    def install_retention(self,old_checkpoint,new_checkpoint,new_names,protected_new,old_weight,new_weight):
        self.old_teacher=load_teacher(old_checkpoint,self.device)
        self.new_teacher=load_teacher(new_checkpoint,self.device)
        self.teacher_before={'old':module_digest(self.old_teacher),'new':module_digest(self.new_teacher)}
        names=list(self.mdp.motion.names);new_names=set(new_names);protected_new=set(protected_new)
        self.retention_old=torch.tensor([name.removesuffix('.csv') not in new_names for name in names],device=self.device)
        self.retention_new=torch.tensor([name.removesuffix('.csv') in protected_new for name in names],device=self.device)
        self.retention_weights=(old_weight,new_weight)
        self.retention_records=[]

    def retention_loss(self,start,history,goal):
        # Teachers are training-only. The deployed student remains one 101D policy.
        ids=self.mdp.motion.clip_id[start['f']]
        raw=self.pi._native_unbounded_forward(history,goal)
        total=raw.sum()*0
        record={}
        for name,mask,teacher,dimension,weight in [
            ('old',self.retention_old[ids],self.old_teacher,99,self.retention_weights[0]),
            ('protected_new',self.retention_new[ids],self.new_teacher,101,self.retention_weights[1])]:
            count=int(mask.sum())
            mse=raw.new_zeros(())
            if count:
                with torch.no_grad():target=teacher(history[mask,:,:dimension],goal[mask]).clamp(-20,20)
                mse=(raw[mask]-target).float().square().mean()
                total=total+weight*mse
            record[name+'_samples']=count
            record[name+'_MSE']=float(mse.detach())
        assert bool(torch.isfinite(total))
        record['weighted_retention_loss']=float(total.detach())
        self.retention_records.append(record)
        return total

    def retention_receipt(self):
        after={'old':module_digest(self.old_teacher),'new':module_digest(self.new_teacher)}
        assert after==self.teacher_before
        assert all(parameter.grad is None for teacher in [self.old_teacher,self.new_teacher] for parameter in teacher.parameters())
        return {'teacher_before':self.teacher_before,'teacher_after':after,'teacher_parameters_and_norms_unchanged':True,
                'teacher_gradients_absent':True,'records':self.retention_records,
                'action_bound_calls':{key:int(value) for key,value in self.pi._bound_counters.items()},
                'peak_student_raw_output':float(self.pi._bound_peak),'deployed_student_policies':1,'deployed_teachers':0}
