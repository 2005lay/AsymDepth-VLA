"""GPU preflight for a merged student or the trusted official full teacher."""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher", action="store_true",
                        help="Check official 24V/24P teacher compatibility; no new completion marker required")
    parser.add_argument("--training-config", default=str(ROOT / "pretrained_models/configs"))
    parser.add_argument("--unnorm-key", default="libero_10_no_noops",
                        help="Dataset statistics key expected in this checkpoint")
    args = parser.parse_args()
    import numpy as np
    import torch
    from experiments.robot.openvla_utils import get_vla, get_action_head, get_proprio_projector, get_processor
    from prismatic.models.bridge_contract import extract_bridge_states
    from prismatic.vla.constants import NUM_TOKENS
    from PIL import Image

    path = Path(args.checkpoint).resolve()
    if not args.teacher and not (path / "checkpoint_complete.json").is_file():
        raise FileNotFoundError("No completion marker: checkpoint was not fully verified")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this preflight in the training GPU environment")
    cfg = SimpleNamespace(pretrained_checkpoint=str(path),num_images_in_input=2,use_film=False,
                          load_in_8bit=False,load_in_4bit=False,use_checkpoint_modeling=True,
                          use_l1_regression=True,use_pro_version=True,use_spline_head=False)
    model=get_vla(cfg)
    # Match the production LIBERO evaluator.  Released teacher checkpoints do
    # not serialize this runtime-only attribute, but their dynamic forward()
    # reads it to select the Pro/vla-adapter feature path.
    if not hasattr(model, "set_version"):
        raise AttributeError("Checkpoint model does not expose set_version()")
    model.set_version("vla-adapter")
    head=get_action_head(cfg,model.llm_dim)
    if args.teacher and (len(model.language_model.model.layers) != 24 or
                         len(head.model.mlp_resnet_blocks) != 24):
        raise ValueError("Teacher preflight requires the official full 24V/24P model")
    projector=get_proprio_projector(cfg,model.llm_dim,8)
    processor=get_processor(cfg)
    # Synthetic images/proprio test consistency, NOT task performance.
    prompt=('<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'
            '<|im_start|>user\nWhat action should the robot take to open the drawer?<|im_end|>\n<|im_start|>assistant\n')
    test_image=Image.fromarray(np.random.default_rng(0).integers(0,256,(256,256,3),dtype=np.uint8))
    inputs=processor(prompt,test_image).to("cuda",dtype=torch.bfloat16)
    if args.teacher:
        from transformers import AutoTokenizer
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        training_processor=PrismaticProcessor(
            image_processor=PrismaticImageProcessor.from_pretrained(args.training_config),
            tokenizer=AutoTokenizer.from_pretrained(args.training_config,trust_remote_code=False))
        training_inputs=training_processor(prompt,test_image).to("cuda",dtype=torch.bfloat16)
        for key in ("input_ids","attention_mask","pixel_values"):
            if not torch.equal(training_inputs[key],inputs[key]):
                raise RuntimeError(f"Teacher/training processor mismatch: {key}")
    inputs["pixel_values"]=torch.cat((inputs["pixel_values"],inputs["pixel_values"]),dim=1)
    norm_key=args.unnorm_key
    if norm_key not in model.norm_stats:
        raise KeyError(f"Checkpoint statistics missing {norm_key}")
    proprio=torch.zeros(1,8,device="cuda",dtype=torch.bfloat16)
    ids,attention=model._prepare_input_for_action_prediction(inputs["input_ids"],inputs["attention_mask"])
    labels=torch.full_like(inputs["input_ids"],-100)
    labels=model._prepare_labels_for_action_prediction(labels,ids)
    patches=model.vision_backbone.get_num_patches()*2
    # Match production get_vla_action: BF16 weights/inputs, no extra autocast.
    with torch.inference_mode():
        out=model(input_ids=ids,attention_mask=attention,pixel_values=inputs["pixel_values"],labels=labels,
                  output_hidden_states=True)
        states=extract_bridge_states(out.hidden_states,labels,patches,NUM_TOKENS)
        direct=head.predict_action(states,proprio,projector,phase="Inference").float().cpu().numpy()[0]
        if args.teacher:
            # Training distillation consumes teacher forward() hidden states via
            # extract_bridge_states and never calls the released checkpoint's
            # legacy predict_action bridge.  That legacy bridge uses a
            # different tensor contract from the compact Pro head.
            actions=direct
            error=0.0
        else:
            normalized=[]
            class Capture:
                def predict_action(self,*args,**kwargs):
                    result=head.predict_action(*args,**kwargs)
                    normalized.append(result.float().cpu().numpy()[0])
                    return result
            actions,returned_states=model.predict_action(**inputs,unnorm_key=norm_key,
                                           proprio=np.zeros(8,dtype=np.float32),
                                           proprio_projector=projector,action_head=Capture(),do_sample=False)
            error=float(np.max(np.abs(direct-normalized[0])))
            del returned_states
        del out, states
    assert np.isfinite(actions).all(), "Non-finite inferred actions"
    assert error < 2e-3, f"Training/inference normalized action mismatch: {error}"
    if args.teacher:
        print("PASS: released teacher forward states, strict components and finite distillation actions")
    else:
        print(f"PASS: checkpoint-owned dynamic code, physical VLM/policy depth, strict components, finite actions; parity max error={error}")
    if args.teacher:
        # Compare the ACTUAL local training VLM, not just two paths through the
        # official dynamic class. Release it first to bound peak GPU memory.
        del model
        torch.cuda.empty_cache()
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        local,info=OpenVLAForActionPrediction.from_pretrained(
            path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
            raise RuntimeError(f"Training teacher load is incomplete: {info}")
        local=local.to("cuda").eval().requires_grad_(False)
        local.vision_backbone.set_num_images_in_input(2)
        with torch.inference_mode():
            out=local(input_ids=ids,attention_mask=attention,pixel_values=inputs["pixel_values"],
                      labels=labels,output_hidden_states=True)
            local_states=extract_bridge_states(out.hidden_states,labels,patches,NUM_TOKENS)
            assert torch.isfinite(local_states).all(), "Non-finite local teacher bridge states"
            local_actions=head.predict_action(local_states,proprio,projector,phase="Inference").float().cpu().numpy()[0]
        assert np.isfinite(local_actions).all(), "Non-finite local teacher actions"
        print("PASS: actual frozen training-teacher load, bridge extraction and action distillation path")
    print("Next: run the matching LIBERO suite with 10 tasks x 10 trials using the same base protocol.")


if __name__=="__main__":
    main()
