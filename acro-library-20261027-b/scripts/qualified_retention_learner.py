"""Training-only selective qualified teacher; native actor/critic update stays inherited."""
import hashlib,json
import torch
from runtime import ROOT
from retention_learner import RetentionLearner

class QualifiedRetentionLearner(RetentionLearner):
    def install_retention(self,old_checkpoint,new_checkpoint,new_names,protected_new,old_weight,new_weight):
        qualification=json.loads((ROOT/'evidence/specialist-teacher-8808-1030.json').read_text())
        assert qualification['qualified_for_8808_under_both_references']
        assert set(protected_new)=={'88_08_stageii_unitree_g1'}
        with new_checkpoint.open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==qualification['checkpoint_sha256']
        super().install_retention(old_checkpoint,new_checkpoint,new_names,protected_new,old_weight,new_weight)
        assert self.new_teacher.obs_norm.mean.numel()==99

    def retention_loss(self,start,history,goal):
        ids=self.mdp.motion.clip_id[start['f']]
        raw=self.pi._native_unbounded_forward(history,goal)
        total=raw.sum()*0;record={}
        for name,mask,teacher,weight in [
            ('old',self.retention_old[ids],self.old_teacher,self.retention_weights[0]),
            ('protected_new',self.retention_new[ids],self.new_teacher,self.retention_weights[1])]:
            count=int(mask.sum());mse=raw.new_zeros(())
            if count:
                dimension=teacher.obs_norm.mean.numel()
                with torch.no_grad():target=teacher(history[mask,:,:dimension],goal[mask]).clamp(-20,20)
                mse=(raw[mask]-target).float().square().mean();total=total+weight*mse
            record[name+'_samples']=count;record[name+'_MSE']=float(mse.detach())
        assert torch.isfinite(total)
        record['weighted_retention_loss']=float(total.detach())
        self.retention_records.append(record)
        return total
