"""Read the local compiled runtime's hand/wrist collision flags; perform no simulation."""
import hashlib,json
import mujoco
from runtime import ROOT,UNI,make_env,references


def main():
    env=make_env(references('new')[:1],observation_dim=99)
    try:
        model=env.backend.model;rows=[]
        floors=[i for i in range(model.ngeom) if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE and model.geom_bodyid[i]==0]
        for i in range(model.ngeom):
            body=model.body(int(model.geom_bodyid[i])).name
            if not any(word in body for word in ['wrist','rubber_hand']):continue
            compatible=any((int(model.geom_contype[i]) & int(model.geom_conaffinity[j])) or
                           (int(model.geom_contype[j]) & int(model.geom_conaffinity[i])) for j in floors)
            rows.append({'body':body,'geom':model.geom(i).name,'contype':int(model.geom_contype[i]),
                         'conaffinity':int(model.geom_conaffinity[i]),'floor_collision_mask_compatible':bool(compatible)})
        files=['scene_supermimic.xml','supermimic.xml']
        result={'status':'compiled_local_runtime_collision_inventory_complete','scope':'local CPU model; no force, rollout, or hardware validation',
                'simulation_steps':0,'model_modified':False,'robot_files':{name:hashlib.sha256((UNI/'src/unisupermimic/assets/robots/g1'/name).read_bytes()).hexdigest() for name in files},
                'geometries':rows,'interpretation':'Rubber hand visuals and wrist collision meshes must be distinguished; collision masks alone do not establish support forces or dynamic feasibility.'}
        out=ROOT/'evidence/runtime-hand-collision-inventory.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
        print('HAND_COLLISION_INVENTORY '+json.dumps(result),flush=True)
    finally:env.close();env.backend.close()


if __name__=='__main__':main()
