"""Derive a G1-only batched SONIC graph and compare actual released-model outputs.

Original weights stay unchanged. This checks graph inference on synthetic G1
features; it does not validate sensor construction, simulation or policy quality.
Models: NVIDIA GEAR-SONIC, revision6733128a3d8a523b1418b06bca3cdf61c8b0987f.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

MODEL_HASHES={
    'model_encoder.onnx':'013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3',
    'model_decoder.onnx':'c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed',
    'observation_config.yaml':'466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b',
}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dynamic_batch(model):
    import onnx
    from onnx import numpy_helper
    model=copy.deepcopy(model)
    constants={n.output[0]:n for n in model.graph.node if n.op_type=='Constant'}
    changes=[]
    for node in model.graph.node:
        if node.op_type!='Reshape' or node.input[1] not in constants:continue
        constant=constants[node.input[1]]
        value=next((a for a in constant.attribute if a.name=='value'),None)
        if value is None:continue
        shape=numpy_helper.to_array(value.t)
        if shape.ndim==1 and shape.size and shape[0]==1:
            assert not any(a.name=='allowzero' and a.i for a in node.attribute)
            updated=shape.copy();updated[0]=0  # ONNX: copy that input dimension.
            value.t.CopyFrom(numpy_helper.from_array(updated))
            changes.append(dict(node=node.name,old_shape=shape.tolist(),new_shape=updated.tolist()))
    for value in list(model.graph.input)+list(model.graph.output):
        dimension=value.type.tensor_type.shape.dim[0]
        dimension.ClearField('dim_value');dimension.dim_param='batch'
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    return model,changes


def synthetic_g1_inputs(count):
    """Valid-range synthetic tensors; no claim of an observed robot trajectory."""
    rng=np.random.default_rng(211115)
    encoder=np.zeros((count,1762),dtype=np.float32)
    phases=rng.uniform(-np.pi,np.pi,(count,1,29))
    frames=np.arange(10,dtype=np.float32)[None,:,None]*.1
    encoder[:,4:294]=(.2*np.sin(phases+frames)).reshape(count,290)
    encoder[:,294:584]=(.2*np.cos(phases+frames)).reshape(count,290)
    # Scalar encoder mode0 followed by three zeros, rather than a one-hot code.
    # Required G1 columns: q290,qd290 and sixty 6D relative orientation values.
    angles=rng.uniform(-.4,.4,(count,10))
    orientation=np.zeros((count,10,3,2),dtype=np.float32)
    orientation[:,:,0,0]=np.cos(angles);orientation[:,:,0,1]=-np.sin(angles)
    orientation[:,:,1,0]=np.sin(angles);orientation[:,:,1,1]=np.cos(angles)
    encoder[:,601:661]=orientation.reshape(count,60)
    proprio=rng.normal(0,.04,(count,930)).astype(np.float32)
    gravity=proprio[:,-30:].reshape(count,10,3)
    gravity[:,:,2]=-1
    gravity/=np.linalg.norm(gravity,axis=-1,keepdims=True)
    return encoder,proprio


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import onnx
    from onnx import utils
    import onnxruntime as ort
    models=args.models.resolve();output=args.output.resolve()
    assert all(digest(models/name)==value for name,value in MODEL_HASHES.items())
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    full_encoder=onnx.load(models/'model_encoder.onnx')
    full_decoder=onnx.load(models/'model_decoder.onnx')
    # The pinned mode0 selector returns this quantized G1 branch unchanged.
    # Mode routing, teleoperation and SMPL branches are deliberately not exported.
    intermediate=output/'g1-encoder-original-static.onnx'
    utils.extract_model(str(models/'model_encoder.onnx'),str(intermediate),
                        ['obs_dict'],['/quantizer/Reshape_1_output_0'])
    g1=onnx.load(intermediate)
    assert not any(n.op_type in ('ScatterND','ReduceSum') for n in g1.graph.node)
    originals={a.name:a.SerializeToString() for a in full_encoder.graph.initializer}
    assert all(originals[a.name]==a.SerializeToString() for a in g1.graph.initializer)
    batch_encoder,encoder_changes=dynamic_batch(g1)
    batch_decoder,decoder_changes=dynamic_batch(full_decoder)
    assert all(a.SerializeToString()==b.SerializeToString()
               for a,b in zip(g1.graph.initializer,batch_encoder.graph.initializer,strict=True))
    assert all(a.SerializeToString()==b.SerializeToString()
               for a,b in zip(full_decoder.graph.initializer,batch_decoder.graph.initializer,strict=True))
    encoder_path=output/'g1-encoder-batched.onnx'
    decoder_path=output/'decoder-batched.onnx'
    onnx.save(batch_encoder,encoder_path);onnx.save(batch_decoder,decoder_path)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=lambda path:ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    original_encoder=session(models/'model_encoder.onnx')
    original_decoder=session(models/'model_decoder.onnx')
    derived_encoder=session(encoder_path);derived_decoder=session(decoder_path)
    features,proprio=synthetic_g1_inputs(64)
    original_tokens=np.concatenate([original_encoder.run(None,{'obs_dict':x[None]})[0] for x in features])
    original_inputs=np.concatenate((original_tokens,proprio),axis=1)
    original_actions=np.concatenate([original_decoder.run(None,{'obs_dict':x[None]})[0] for x in original_inputs])
    comparisons=[]
    arrays=dict(encoder_features=features,proprio=proprio,
                original_tokens=original_tokens,original_actions=original_actions)
    for count in (1,2,18,64):
        tokens=derived_encoder.run(None,{'obs_dict':features[:count]})[0].reshape(count,64)
        decoder_input=np.concatenate((tokens,proprio[:count]),axis=1)
        actions=derived_decoder.run(None,{'obs_dict':decoder_input})[0]
        assert tokens.shape==(count,64) and actions.shape==(count,29)
        assert np.isfinite(tokens).all() and np.isfinite(actions).all()
        token_error=float(np.max(np.abs(tokens-original_tokens[:count])))
        action_error=float(np.max(np.abs(actions-original_actions[:count])))
        assert np.allclose(tokens,original_tokens[:count],atol=1e-6,rtol=1e-6),token_error
        assert np.allclose(actions,original_actions[:count],atol=1e-4,rtol=1e-4),action_error
        arrays[f'derived_tokens_{count}']=tokens;arrays[f'derived_actions_{count}']=actions
        comparisons.append(dict(batch=count,token_max_absolute_difference=token_error,
                                action_max_absolute_difference=action_error))
    trace=output/'synthetic-inference-comparison.npz'
    np.savez_compressed(trace,**arrays)
    with np.load(trace,allow_pickle=False) as saved:
        assert set(saved.files)==set(arrays)
        assert all(np.array_equal(saved[name],value) for name,value in arrays.items())
    assert all(digest(models/name)==value for name,value in MODEL_HASHES.items())
    report=dict(status='released_SONIC_G1_batched_ONNX_outputs_match_original_CPU_inference',
        HF_repository='nvidia/GEAR-SONIC',HF_revision='6733128a3d8a523b1418b06bca3cdf61c8b0987f',
        source_commit='1e1ec4f5f27e6c3e03048f693bd7c60cce04afda',
        original_files=MODEL_HASHES,original_files_unchanged=True,
        derived_graph_scope='Mode0 G1 quantized branch only; input/output batch dimensions and integer reshape shape constants',
        encoder_reshape_changes=encoder_changes,decoder_reshape_changes=decoder_changes,
        every_retained_parameter_byte_unchanged=True,
        execution_provider='CPUExecutionProvider',onnxruntime_version=ort.__version__,onnx_version=onnx.__version__,
        encoder_input_dimensions=1762,latent_dimensions=64,decoder_input_dimensions=994,action_dimensions=29,
        encoder_mode='scalar0 followed by three zeros; other-mode feature groups zeroed',
        comparisons=comparisons,derived_files=[dict(file=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in (encoder_path,decoder_path)],
        trace=dict(file=trace.name,bytes=trace.stat().st_size,sha256=digest(trace),readback_verified=True),
        elapsed_seconds=time.monotonic()-started,
        sensor_port_verified=False,training_environment_equivalence_verified=False,
        simulation_executed=False,GPU_execution_verified=False,policy_quality_evaluated=False,
        physical_robot_deployment=False,paper_results_reproduced=False)
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
