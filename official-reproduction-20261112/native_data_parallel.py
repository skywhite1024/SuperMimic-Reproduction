"""Explicit collectives for a later two-rank native first-order experiment.

These helpers do not launch jobs or establish NCCL/CUDA training. Native
captured updates must run eagerly when Python collectives are installed.
"""
import torch
import torch.distributed as dist


def gather_equal_rows(tensor):
    """Concatenate equal-sized rank shards in rank order, with shape checks."""
    tensor=tensor.contiguous()
    shape=torch.tensor(tensor.shape,device=tensor.device,dtype=torch.long)
    shapes=[torch.empty_like(shape) for _ in range(dist.get_world_size())]
    dist.all_gather(shapes,shape)
    if not all(torch.equal(shape,value) for value in shapes):
        raise ValueError('Native data-parallel shards must have equal shapes')
    shards=[torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(shards,tensor)
    return torch.cat(shards,dim=0)


def broadcast_module_state(module,source=0):
    """Share learned parameters and persistent inference/normalization buffers."""
    for value in module.state_dict().values():
        dist.broadcast(value,src=source)


def average_gradients(parameters):
    """Average local-mean gradients, treating rank-local unused weights as zero.

    All ranks must use identical ordered optimizer parameters and equal local
    batch weights. Clipping happens after reduction, never on each shard first.
    """
    parameters=list(parameters)
    if not parameters:
        return
    device=parameters[0].device
    present=torch.tensor([parameter.grad is not None for parameter in parameters],
                         dtype=torch.long,device=device)
    dist.all_reduce(present,op=dist.ReduceOp.SUM)
    world=dist.get_world_size()
    # None on every rank stays None, preserving native optimizer semantics.
    # Iterating all parameters identically also handles a locally unused branch.
    for parameter,count in zip(parameters,present.tolist(),strict=True):
        if count==0:
            continue
        if parameter.grad is None:
            parameter.grad=torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)


def optimize_mean_gradient(optimizers,loss,params=None,max_norm=None,clip_value=0.0):
    """Native optimizer order, adding gradient averaging before both clips."""
    selected=(list(params) if params is not None else
              [parameter for optimizer in optimizers for group in optimizer.param_groups
               for parameter in group['params']])
    # Preserve a generator through both clipping passes, including the critic.
    if len({id(parameter) for parameter in selected})!=len(selected):
        raise ValueError('Optimizer gradient list must not repeat parameters')
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss.backward()
    average_gradients(selected)
    grad_norm=None
    if params is not None and max_norm is not None:
        grad_norm=torch.nn.utils.clip_grad_norm_(selected,max_norm=max_norm)
        if clip_value>0:
            torch.nn.utils.clip_grad_value_(selected,clip_value)
    for optimizer in optimizers:
        optimizer.step()
    return grad_norm


class SharedReturnSampler:
    """Use one identical native return sampler over all ranks' real episodes."""
    def __init__(self,cfg,bin_count,frame_count,local_num_envs,max_step_reward,device):
        from supermimic.runners.reset_sampling import ReturnSampler
        self.native=ReturnSampler(cfg,bin_count,frame_count,
            local_num_envs*dist.get_world_size(),max_step_reward,device=device)
        self.local_num_envs=local_num_envs
        self.records=0
        self.distribution_updates=0

    def record(self,frames,rewards,terminated,clip_end,timed_out):
        inputs=[frames,rewards,terminated,clip_end,timed_out]
        if any(value.shape!=(self.local_num_envs,) for value in inputs):
            raise ValueError('Each sampler input must be one real vector per local environment')
        gathered=[gather_equal_rows(value) for value in inputs]
        result=self.native.record(*gathered)
        self.records+=1
        self.distribution_updates+=int(result is not None)
        return result


def install_native_runner_sync(runner):
    """Install process-local sync after constructing a native runner.

    Call after resolving sqrt(world_size) learning rates in the configuration,
    before collecting data. This does not provide resumable distributed state.
    """
    from rsl_rl.modules import EmpiricalNormalization
    import supermimic.learners.policy_learner as policy_module
    import supermimic.learners.wm_learner as wm_module
    if not dist.is_initialized() or dist.get_world_size()!=2:
        raise ValueError('This integration requires an initialized two-rank group')
    if runner._sampler is None:
        raise ValueError('Expected the native return reset sampler')
    actor=runner.policy_learner.actor
    modules=[runner._dynamics,actor]
    critic=getattr(runner.policy_learner,'critic',None)
    if critic is not None:modules.append(critic)
    seen=set()
    normalizers=[]
    for module in modules:
        broadcast_module_state(module)
        for child in module.modules():
            if isinstance(child,EmpiricalNormalization) and id(child) not in seen:
                # Installed RSL 5.5.2 already reduces count, mean and variance.
                # Gathering the samples here would count every rank twice.
                normalizers.append(child);seen.add(id(child))
    counts=dict(world_model_optimizer_steps=0,actor_optimizer_steps=0,critic_optimizer_steps=0)
    actor_optimizer_ids={id(optimizer) for optimizer in runner.policy_learner.actor_optimizers}
    def wm_step(*args,**kwargs):
        result=optimize_mean_gradient(*args,**kwargs)
        counts['world_model_optimizer_steps']+=1
        return result
    def policy_step(optimizers,*args,**kwargs):
        result=optimize_mean_gradient(optimizers,*args,**kwargs)
        key='actor_optimizer_steps' if id(optimizers[0]) in actor_optimizer_ids else 'critic_optimizer_steps'
        counts[key]+=1
        return result
    policy_module._optimize_step=policy_step
    wm_module._optimize_step=wm_step
    runner._native_dp_optimizer_counts=counts
    # Python collective dispatch cannot be omitted by CUDA graph replay.
    # Keep the native loss/rollout functions, bypassing only CapturedUpdate.
    for learner in (runner.world_model_learner,runner.policy_learner):
        learner._captured_update=learner._update_from
    runner._sampler=SharedReturnSampler(runner.cfg.return_sampler,
        runner.motion_cmd.bin_count,runner.motion.time_step_total,runner.env.num_envs,
        runner.wm_env.max_step_reward,runner.device)
    return dict(world_size=2,normalizers=len(normalizers),
        gradients='local means averaged before norm and element clipping',
        normalizer='installed RSL native distributed count/mean/variance; no extra sample gathering',
        sampler='identical native returns and reset distribution over all real ranks',
        captured_learner_updates=False,full_resume_available=False,
        exact_paper_execution_performance_confirmed=False)
