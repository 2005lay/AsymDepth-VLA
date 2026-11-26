"""CPU regression tests. No real checkpoint, simulator or downloads required."""
import argparse
import ast
import copy
import importlib.util
import json
import re
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def isolated(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def actual_function(relative, name, namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.returns = None
    node.decorator_list = []
    for arg in node.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    for file in ROOT.rglob("*.py"):
        ast.parse(file.read_text(encoding="utf-8-sig"), filename=str(file))
    contract = isolated("contract", "prismatic/models/bridge_contract.py")
    assert contract.policy_hidden_indices(16, 12) == (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15)
    assert contract.policy_hidden_indices(6, 8) == (1, 1, 2, 3, 4, 4, 5, 6)
    assert contract.policy_hidden_indices(24, 24) == tuple(range(1, 25))
    print("PASS: Python syntax and floor-reindex formula")
    runner=(ROOT/"scripts/run_libero_long_fixed.sh").read_text(encoding="utf-8")
    for file, classname, entry in (
        ("vla-scripts/finetune.py","FinetuneConfig",'"$PROJECT_ROOT/vla-scripts/finetune.py"'),
        ("experiments/robot/libero/run_libero_eval.py","GenerateConfig",
         '"$PROJECT_ROOT/experiments/robot/libero/run_libero_eval.py"'),
    ):
        tree=ast.parse((ROOT/file).read_text(encoding="utf-8-sig"))
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==classname)
        fields={node.target.id for node in cls.body if isinstance(node,ast.AnnAssign)}
        command=runner.split(entry,1)[1].split('2>&1 | tee',1)[0]
        flags=set(re.findall(r"--([a-z][a-z_0-9]*)",command))
        assert flags <= fields, f"Unsupported CLI arguments for {file}: {flags-fields}"
    selected_runner=(ROOT/"scripts/run_libero_no_teacher_selected_lr2e4.sh").read_text(encoding="utf-8")
    for file, classname, start, end, launcher_flags in (
        ("vla-scripts/finetune.py", "FinetuneConfig", "python -u -m torch.distributed.run",
         '2>&1 | tee "$suite_log/train.log"', {"standalone", "nnodes", "nproc_per_node"}),
        ("experiments/robot/libero/run_libero_eval.py", "GenerateConfig",
         'python -u "$PROJECT_ROOT/experiments/robot/libero/run_libero_eval.py"',
         '2>&1 | tee "$step_log/eval_10x10.log"', set()),
    ):
        tree=ast.parse((ROOT/file).read_text(encoding="utf-8-sig"))
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==classname)
        fields={node.target.id for node in cls.body if isinstance(node,ast.AnnAssign)}
        command=selected_runner.split(start,1)[1].split(end,1)[0]
        flags=set(re.findall(r"--([a-z][a-z_0-9]*)",command))-launcher_flags
        assert flags <= fields, f"Unsupported selected-workflow CLI arguments for {file}: {flags-fields}"
    print("PASS: delivered train/eval CLI arguments match actual config dataclasses")
    train_tree = ast.parse((ROOT/"vla-scripts/finetune.py").read_text(encoding="utf-8"))
    train_fn = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef) and n.name == "finetune")
    # Teacher tensors may initialize the student only in teacher_sliced mode;
    # teacher KD remains independently enabled by use_distill.
    for call_name in ("load_student_proprio_from_teacher", "load_student_action_head_from_teacher"):
        guarded = [n for n in ast.walk(train_fn) if isinstance(n, ast.If)
                   and "cfg.student_init_mode" in ast.unparse(n.test)
                   and "teacher_sliced" in ast.unparse(n.test)
                   and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                           and c.func.id == call_name for stmt in n.body for c in ast.walk(stmt))]
        assert guarded, call_name
    save_if = next(n for n in ast.walk(train_fn) if isinstance(n, ast.If)
                   and "log_step in requested_save_steps" in ast.unparse(n.test))
    predicate = compile(ast.Expression(save_if.test), "save_schedule", "eval")
    requested = {10,1000,2000,3000,4000,5000,7000,10000}
    scope = {"requested_save_steps": requested, "cfg": types.SimpleNamespace(save_freq=1000,max_steps=10000)}
    actual = {s for s in range(1,10001) if eval(predicate, {}, dict(scope, log_step=s))}
    assert actual == requested
    train_source=(ROOT/"vla-scripts/finetune.py").read_text(encoding="utf-8")
    assert "vla.requires_grad_(True)" in train_source
    assert "Random student training requires use_lora=False" in train_source
    assert 'cfg.student_init_mode == "random_selected"' in train_source
    assert 'cfg.student_init_mode == "base_vlm_random_policy"' in train_source
    assert "build_base_vlm_random_policy_state" in train_source
    assert "zero LIBERO teacher tensors copied into student" in train_source
    assert "vla.language_model.model.layers.requires_grad_(True)" in train_source
    assert "vla.action_queries.requires_grad_(True)" in train_source
    assert "no selected VLM layer weights copied" in train_source
    save_fn=next(n for n in train_tree.body if isinstance(n,ast.FunctionDef)
                 and n.name=="save_training_checkpoint")
    save_source=ast.unparse(save_fn)
    assert "inference_model.save_pretrained(checkpoint_dir" in save_source
    assert '"student_init_mode": init_mode' in train_source
    assert '"student_training_mode": training_mode' in train_source
    utils_tree=ast.parse((ROOT/"experiments/robot/openvla_utils.py").read_text(encoding="utf-8"))
    action_fn=next(n for n in utils_tree.body if isinstance(n,ast.FunctionDef) and n.name=="get_vla_action")
    action_source=ast.unparse(action_fn)
    assert "register_forward_hook(mark_language_model_end)" in action_source
    assert action_source.index("vlm_component_start") < action_source.index("prepare_images_for_vla")
    assert "timed_action_head_predict" in action_source
    assert "policy_action_head_end_times.append" in action_source
    assert "policy_action_head_end_times[0] - language_model_end" in action_source
    assert "policy_component_end - language_model_end" not in action_source
    assert "elapsed - action_head_elapsed" not in action_source
    print("PASS: random-selected/full-random guards, complete save path and exact optimizer-step schedule")
    print("PASS: table latency boundary uses LM output -> bridge preparation -> normalized action-head output")
    if args.static_only:
        print("Tensor/forward/checkpoint tests NOT RUN (--static-only)")
        return

    import torch
    import numpy as np
    from torch import nn
    from types import SimpleNamespace as NS
    from PIL import Image
    torch.set_num_threads(2)
    for name in ("prismatic", "prismatic.models", "prismatic.vla", "prismatic.util", "prismatic.training"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    sys.modules["prismatic.models.bridge_contract"] = contract
    constants = isolated("prismatic.vla.constants", "prismatic/vla/constants.py")
    pruning = isolated("prismatic.models.pruning", "prismatic/models/pruning.py")
    heads = isolated("prismatic.models.action_heads", "prismatic/models/action_heads.py")
    projectors = isolated("projectors", "prismatic/models/projectors.py")
    train_utils = isolated("train_utils", "prismatic/training/train_utils.py")

    # Exact no-teacher-selected-weights initialization contract: compact VLM
    # blocks stay random while all non-layer tensors remain teacher-initialized.
    init_ns = {"nn": nn, "slice_state_dict": pruning.slice_state_dict}
    is_random_selected = actual_function(
        "vla-scripts/finetune.py", "is_random_selected_vlm_parameter", init_ns
    )
    init_ns["is_random_selected_vlm_parameter"] = is_random_selected
    expand_aliases = actual_function(
        "vla-scripts/finetune.py", "expand_tied_parameter_aliases", init_ns
    )
    init_ns["expand_tied_parameter_aliases"] = expand_aliases
    build_random_selected = actual_function(
        "vla-scripts/finetune.py", "build_random_selected_vlm_state", init_ns
    )
    build_base_random_policy = actual_function(
        "vla-scripts/finetune.py", "build_base_vlm_random_policy_state", init_ns
    )

    class ToyLanguage(nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(depth)])
            self.model.norm = nn.LayerNorm(2)

    class ToyVLM(nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.vision_backbone = nn.Linear(2, 2, bias=False)
            self.projector = nn.Linear(2, 2, bias=False)
            self.language_model = ToyLanguage(depth)
            self.action_queries = nn.Embedding(2, 2)

    generic = ToyVLM(4)
    compact = ToyVLM(2)
    generic_state = {key: torch.full_like(value, 7) for key, value in generic.state_dict().items()}
    compact_random = {key: value.clone() for key, value in compact.state_dict().items()}
    mixed = build_random_selected(compact, generic_state, (1, 4))
    for key, value in mixed.items():
        if is_random_selected(key):
            assert torch.equal(value, compact_random[key]), key
        else:
            assert torch.equal(value, torch.full_like(value, 7)), key
    assert is_random_selected("language_model.model.layers.0.weight")
    assert not is_random_selected("action_queries.weight")
    assert not is_random_selected("vision_backbone.weight")
    assert torch.equal(mixed["action_queries.weight"], torch.full_like(mixed["action_queries.weight"], 7))
    print("PASS: selected VLM blocks random; teacher non-layer/bridge tensors retained")

    generic_state["language_model.model.layers.0.weight"] = torch.full_like(
        generic_state["language_model.model.layers.0.weight"], 10
    )
    generic_state["language_model.model.layers.3.weight"] = torch.full_like(
        generic_state["language_model.model.layers.3.weight"], 40
    )
    compact_from_base = ToyVLM(2)
    random_action_queries = compact_from_base.action_queries.weight.detach().clone()
    base_mixed = build_base_random_policy(compact_from_base, generic_state, (1, 4))
    assert torch.equal(
        base_mixed["language_model.model.layers.0.weight"],
        torch.full_like(base_mixed["language_model.model.layers.0.weight"], 10),
    )
    assert torch.equal(
        base_mixed["language_model.model.layers.1.weight"],
        torch.full_like(base_mixed["language_model.model.layers.1.weight"], 40),
    )
    assert torch.equal(base_mixed["action_queries.weight"], random_action_queries)
    assert torch.equal(base_mixed["vision_backbone.weight"], torch.full_like(base_mixed["vision_backbone.weight"], 7))
    print("PASS: generic base VLM slicing + random action queries; no teacher state accepted")

    local_loader_ns = {"Path": Path, "torch": torch}
    load_local_generic = actual_function(
        "vla-scripts/finetune.py", "load_local_prismatic_vlm_state", local_loader_ns
    )
    with tempfile.TemporaryDirectory() as temp:
        generic_dir = Path(temp)
        (generic_dir / "checkpoints").mkdir()
        nested = {
            "vision_backbone": {"dino.weight": torch.ones(1)},
            "projector": {"projector.0.weight": torch.ones(1) * 2},
            "llm_backbone": {"llm.model.layers.0.weight": torch.ones(1) * 3},
        }
        torch.save({"model": nested}, generic_dir / "checkpoints" / "step-020792-epoch-01-loss=0.5268.pt")
        flat = load_local_generic(generic_dir)
        assert set(flat) == {
            "vision_backbone.dino.weight",
            "projector.projector.0.weight",
            "llm_backbone.llm.model.layers.0.weight",
        }
    print("PASS: local aggregate Prismatic checkpoint loads packaged tensors without Hub access")

    # Mixed prompt lengths and right padding; exact official predecessor slice.
    p, q, d = 4, 64, 16
    lengths = (3, 7, 11)
    labels = torch.full((3, max(lengths) + q + 1), -100)
    for b, length in enumerate(lengths):
        labels[b, length:length+q] = constants.ACTION_TOKEN_BEGIN_IDX + 1
        labels[b, length+q] = constants.STOP_INDEX
    state = torch.arange(3*(labels.shape[1]+p)*d).reshape(3,-1,d).float().requires_grad_()
    features = contract.extract_bridge_states((state, state+1, state+2), labels, p, q)
    for b, length in enumerate(lengths):
        torch.testing.assert_close(features[b,-1,p:], state[b,p+length-1:p+length-1+q]+2)
    features.sum().backward()
    assert state.grad is not None
    print("PASS: mixed prompt lengths, padding, exact official features and autograd")

    # Actual compact Pro head and selected teacher-block inheritance.
    teacher_head = heads.L1RegressionActionHead(d,d,7,num_task_tokens=p,use_pro_version=True)
    head = heads.L1RegressionActionHead(d,d,7,num_task_tokens=p,use_pro_version=True,
                                      vlm_kept_layers="2,24",policy_kept_layers="4,24")
    sliced = pruning.slice_state_dict(teacher_head.state_dict(),head.state_dict(),(4,24))
    pruning.load_complete_state_dict(head,sliced,"policy slicing test")
    assert len(head.model.mlp_resnet_blocks) == 2
    for compact, original in enumerate((4,24)):
        torch.testing.assert_close(head.model.mlp_resnet_blocks[compact].q_proj.weight,
                                   teacher_head.model.mlp_resnet_blocks[original-1].q_proj.weight,rtol=0,atol=0)
    random_head = heads.L1RegressionActionHead(
        d,d,7,num_task_tokens=p,use_pro_version=True,
        vlm_kept_layers="2,24",policy_kept_layers="4,24"
    )
    random_blocks = {
        key: value.clone() for key, value in random_head.state_dict().items()
        if key.startswith("model.mlp_resnet_blocks.")
    }
    policy_ns = {
        "load_component_from_dir": lambda *_: teacher_head.state_dict(),
        "parse_kept_layers": pruning.parse_kept_layers,
        "slice_state_dict": pruning.slice_state_dict,
        "load_complete_state_dict": pruning.load_complete_state_dict,
    }
    load_random_policy = actual_function(
        "vla-scripts/finetune.py", "load_student_action_head_with_random_selected_blocks", policy_ns
    )
    load_random_policy(random_head, NS(teacher_ckpt="teacher", policy_kept_layers="4,24"))
    for key, value in random_blocks.items():
        torch.testing.assert_close(random_head.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(random_head.model.fc1.weight, teacher_head.model.fc1.weight, rtol=0, atol=0)
    torch.testing.assert_close(random_head.model.fc2.weight, teacher_head.model.fc2.weight, rtol=0, atol=0)
    projector = projectors.ProprioProjector(d,8)
    pred = head(features.detach().div(1000),torch.randn(3,8),projector)
    assert pred.shape == (3,8,7) and torch.isfinite(pred).all()
    pred.sum().backward()
    assert projector.fc1.weight.grad is not None

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.a=nn.Linear(2,2,bias=False); self.b=nn.Linear(2,2,bias=False)
            self.b.weight=self.a.weight
    pruning.load_complete_state_dict(Tied(),{"a.weight":torch.ones(2,2)})
    try:
        pruning.load_complete_state_dict(nn.Linear(2,2),{"weight":torch.ones(2,2)})
        raise AssertionError("Missing bias accepted")
    except RuntimeError:
        pass
    print("PASS: physical policy slicing, actual Pro forward/backward, strict tied loads")

    # Execute the ACTUAL distillation function, with identical tiny models.
    class TinyVLM(nn.Module):
        def __init__(self):
            super().__init__(); self.queries=nn.Parameter(torch.randn(1,1,d).bfloat16())
        def forward(self,input_ids,**kwargs):
            x=(torch.arange(input_ids.shape[1]+p).reshape(1,-1,1)/100).bfloat16()+self.queries
            x=x.expand(input_ids.shape[0],-1,-1)
            return NS(hidden_states=(x,x*1.1,x*1.2))
    model=TinyVLM()
    teacher=copy.deepcopy(model).eval().requires_grad_(False)
    head=head.bfloat16().eval(); projector=projector.bfloat16().eval()
    cfg=NS(phase="Inference", use_distill=True,teacher_model=teacher,
           teacher_action_head=copy.deepcopy(head).eval().requires_grad_(False),
           teacher_proprio_projector=copy.deepcopy(projector).eval().requires_grad_(False),
           lambda_action=1.,lambda_rep=.5,enable_gate_pruning=False)
    namespace=dict(torch=torch,F=torch.nn.functional,NUM_TOKENS=q,
                   extract_bridge_states=contract.extract_bridge_states,
                   get_current_action_mask=train_utils.get_current_action_mask,
                   get_next_actions_mask=train_utils.get_next_actions_mask)
    forward=actual_function("vla-scripts/finetune.py","run_forward_pass",namespace)
    batch=dict(input_ids=labels.clamp(min=0).long(),labels=labels,attention_mask=torch.ones_like(labels),
               pixel_values=torch.zeros(3,12,8,8),proprio=torch.randn(3,8),actions=torch.zeros(3,8,7))
    loss,metrics=forward(model,head,projector,batch,None,torch.device("cpu"),True,False,True,False,p,cfg=cfg)
    assert metrics["distill_action_loss"] == metrics["distill_rep_loss"] == 0,metrics
    loss.backward()
    assert model.queries.grad is not None and model.queries.grad.count_nonzero()
    assert all(param.grad is None for module in (teacher,cfg.teacher_action_head,cfg.teacher_proprio_projector)
               for param in module.parameters())
    print("PASS: ACTUAL KD function, identical models => zero KD, student gradients, frozen teacher")

    # Validation must use deterministic inference and restore all student modes.
    import time
    modules = [nn.Linear(2,2) for _ in range(3)]
    modules[1].eval()
    original_modes = [module.training for module in modules]
    def validation_forward(**kwargs):
        assert kwargs["cfg"].phase == "Inference"
        assert all(not module.training for module in modules)
        return torch.tensor(0.), {"loss_value": 0.}
    ns = dict(time=time,copy=copy,torch=torch,run_forward_pass=validation_forward)
    validate = actual_function("vla-scripts/finetune.py","run_validation",ns)
    validation_cfg=NS(phase="Training",use_l1_regression=True,use_spline_head=False,
                      use_proprio=True,use_film=False,use_pro_version=True)
    validate(*modules,[{}],None,"cpu",validation_cfg,p,1,NS(is_main_process=False),1)
    assert [module.training for module in modules] == original_modes
    assert validation_cfg.phase == "Training"
    try:
        validate(*modules,[],None,"cpu",validation_cfg,p,1,NS(is_main_process=False),1)
        raise AssertionError("Empty validation accepted")
    except RuntimeError:
        pass
    assert [module.training for module in modules] == original_modes
    print("PASS: ACTUAL validation inference phase and mode restoration")

    ns=dict(np=np)
    summarize=actual_function("experiments/robot/libero/run_libero_eval.py","summarize_runtime_metrics",ns)
    metrics=summarize(NS(model_query_times=[.08,.12],actions_generated=16,episode_wall_times=[3.,4.],
                         vlm_times=[.05,.06],policy_times=[.01,.02]))
    assert abs(metrics["avg_vlm_latency_ms_per_chunk"]-55.) < 1e-9
    assert abs(metrics["avg_policy_latency_ms_per_chunk"]-15.) < 1e-9
    assert metrics["component_timed_queries"] == 2
    assert abs(metrics["avg_total_latency_ms_per_chunk"]-100.) < 1e-9
    assert abs(metrics["amortized_latency_ms_per_returned_action"]-12.5) < 1e-9
    assert abs(metrics["total_model_query_time_sec"]-.2) < 1e-9
    print("PASS: ACTUAL latency summary separates per-chunk/per-action/cumulative time")

    # Execute the actual evaluation-side component timer.  The artificial
    # sleeps make it observable that preprocessing belongs to VLM, bridge work
    # and the action-head call belong to Policy, and post-head work is excluded.
    class Batch(dict):
        def to(self,*args,**kwargs):
            return self
    class Processor:
        def __call__(self,*args,**kwargs):
            time.sleep(.002)
            return Batch(pixel_values=torch.zeros(1,3,2,2),input_ids=torch.ones(1,2,dtype=torch.long),
                         attention_mask=torch.ones(1,2,dtype=torch.long))
    class TimedLM(nn.Module):
        def forward(self,x):
            time.sleep(.004)
            return x
    class TimedVLA(nn.Module):
        def __init__(self):
            super().__init__(); self.language_model=TimedLM()
            self.norm_stats={"key":{"proprio":{}}}
        def predict_action(self,**kwargs):
            self.language_model(torch.ones(1))
            time.sleep(.006)
            action=kwargs["action_head"].predict_action(torch.ones(1))
            # Deliberately much longer than the head call: this post-head work
            # must not leak into the reported Policy latency.
            time.sleep(.100)
            return action,None
    class TimedHead(nn.Module):
        def predict_action(self,*args,**kwargs):
            time.sleep(.006)
            return np.zeros((8,7),dtype=np.float32)
    action_namespace=dict(torch=torch,time=time,np=np,DEVICE=torch.device("cpu"),
                          prepare_images_for_vla=lambda images,cfg:images,
                          normalize_proprio=lambda value,stats:value)
    timed_action=actual_function("experiments/robot/openvla_utils.py","get_vla_action",action_namespace)
    timed_vla=TimedVLA()
    timed_cfg=NS(num_images_in_input=2,use_proprio=True,unnorm_key="key",num_open_loop_steps=8)
    returned=timed_action(timed_cfg,timed_vla,Processor(),
                         {"full_image":np.zeros((2,2,3),dtype=np.uint8),
                          "wrist_image":np.zeros((2,2,3),dtype=np.uint8),
                          "state":np.zeros(8,dtype=np.float32)},
                         "task",action_head=TimedHead(),proprio_projector=object(),use_minivlm=True)
    assert len(returned)==8
    assert timed_vla.last_vlm_time_sec >= .007, timed_vla.last_vlm_time_sec
    assert timed_vla.last_action_head_time_sec >= .010, timed_vla.last_action_head_time_sec
    # Windows can round each 6 ms sleep to roughly 16 ms.  The ceiling admits
    # both bridge + head sleeps while still excluding the 100 ms post-head gap.
    assert timed_vla.last_action_head_time_sec < .080, timed_vla.last_action_head_time_sec
    print("PASS: ACTUAL Policy timer includes bridge preparation + action head and excludes post-head work")

    # Actual inference method, replacing only its expensive language decoder.
    class Decoder(nn.Module):
        def forward(self,inputs_embeds,**kwargs):
            return NS(hidden_states=(inputs_embeds,inputs_embeds*1.1,inputs_embeds*1.2))
    class Recorder:
        def predict_action(self,states,**kwargs):
            self.seen=states
            return torch.zeros(states.shape[0],8,7)
    prompt=lengths[0]
    inference_labels=labels[:1,:prompt+q+1]
    x=torch.randn(1,prompt+q+1,d)
    vision=torch.randn(1,p,d)
    combined=torch.cat((x[:,:1],vision,x[:,1:]),dim=1)
    fake=NS(action_queries=nn.Embedding(q,d),language_model=Decoder(),
            _replace_input_embeddings=lambda a,*_:a,
            _build_multimodal_attention=lambda *a:(combined,None))
    tree=ast.parse((ROOT/"prismatic/extern/hf/modeling_prismatic.py").read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="OpenVLAForActionPrediction")
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_regression_or_discrete_prediction")
    import time
    ns=dict(torch=torch,time=time,NUM_TOKENS=q,NUM_ACTIONS_CHUNK=8,ACTION_DIM=7,
            extract_bridge_states=contract.extract_bridge_states)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),"actual_inference","exec"),ns)
    recorder=Recorder()
    ns[fn.name](fake,x,inference_labels>constants.ACTION_TOKEN_BEGIN_IDX,vision,None,inference_labels,p,prompt-1,
                action_head=recorder)
    expected=contract.extract_bridge_states((combined,combined*1.1,combined*1.2),inference_labels,p,q)
    torch.testing.assert_close(recorder.seen,expected,rtol=0,atol=0)
    print("PASS: ACTUAL inference method reads exactly the training features")

    # Real tokenizer: training prompt must exactly equal eval's generation prefix.
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/"pretrained_models/configs",trust_remote_code=False)
    prompt_base=isolated("prismatic.models.backbones.llm.prompting.base_prompter",
                         "prismatic/models/backbones/llm/prompting/base_prompter.py")
    prompter=isolated("qwen_prompt","prismatic/models/backbones/llm/prompting/qwen_prompter.py")
    from dataclasses import dataclass
    from typing import Any, Dict, Type
    class Actions:
        def __call__(self,actions,use_minivlm):
            return np.full(np.asarray(actions).shape,constants.ACTION_TOKEN_BEGIN_IDX+1).tolist()
    dataset_tree=ast.parse((ROOT/"prismatic/vla/datasets/datasets.py").read_text(encoding="utf-8"))
    node=next(n for n in dataset_tree.body if isinstance(n,ast.ClassDef) and n.name=="RLDSBatchTransform")
    ns=dict(dataclass=dataclass,Any=Any,Dict=Dict,Type=Type,torch=torch,Image=Image,
            ActionTokenizer=Any,PreTrainedTokenizerBase=Any,ImageTransform=Any,
            PromptBuilder=Any,QwenPromptBuilder=prompter.QwenPromptBuilder,NUM_TOKENS=q,
            IGNORE_INDEX=-100,STOP_INDEX=constants.STOP_INDEX)
    exec(compile(ast.Module(body=[node],type_ignores=[]),"actual_transform","exec"),ns)
    transform=ns["RLDSBatchTransform"](Actions(),tokenizer,lambda _:torch.zeros(6,8,8),
                                    prompter.QwenPromptBuilder,use_minivlm=True)
    for instruction in ("open drawer","move both the cream cheese box and butter into the basket"):
        sample=transform(dict(dataset_name="libero_10_no_noops",action=np.zeros((8,7)),
                              task={"language_instruction":instruction.encode()},
                              observation={"image_primary":np.zeros((1,8,8,3),dtype=np.uint8)}))
        eval_prompt=(f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'
                     f'<|im_start|>user\nWhat action should the robot take to {instruction}?<|im_end|>\n<|im_start|>assistant\n')
        assert sample["input_ids"][:-65].tolist()==tokenizer(eval_prompt).input_ids
        assert (sample["labels"][:-65]==-100).all()
        assert (sample["labels"][-65:-1]>constants.ACTION_TOKEN_BEGIN_IDX).all()
        assert sample["labels"][-1]==constants.STOP_INDEX
    utils=isolated("data_utils","prismatic/util/data_utils.py")
    collated=utils.PaddedCollatorForActionPrediction(2048,0)([dict(sample,proprio=np.zeros((1,8)))])
    assert collated["proprio"].shape==(1,8)
    print("PASS: ACTUAL data transform with REAL Qwen tokenizer; exact eval prompt, 64 slots, STOP, batch=1")

    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)
        torch.save({"language_model.model.layers.0.weight":torch.ones(2,2)},path/"pytorch_model-00001.bin")
        (path/"pytorch_model.bin.index.json").write_text(json.dumps({"weight_map":{
            "language_model.model.layers.0.weight":"pytorch_model-00001.bin"}}))
        assert len(pruning.load_checkpoint_state_dict(path))==1
    print("PASS: sharded checkpoint reader")

    # ACTUAL save function + REAL HF/PEFT, with tiny BF16 weights. This is not
    # a replacement for the real checkpoint GPU preflight.
    from transformers import PreTrainedModel, PretrainedConfig
    from peft import LoraConfig, PeftModel, get_peft_model
    import shutil
    class TinyHF(PreTrainedModel):
        config_class=PretrainedConfig
        def __init__(self,config):
            super().__init__(config)
            self.linear=nn.Linear(8,8)
            self.action_queries=nn.Embedding(2,8)
        def forward(self):
            return self.linear(self.action_queries.weight)
    cfg=NS(vlm_kept_layers="1,24",policy_kept_layers="1,24",num_images_in_input=2,use_pro_version=True,
           enable_gate_pruning=False,lambda_gate=1e-4,use_distill=True,teacher_ckpt="unit-test-teacher",
           lambda_action=1.,lambda_rep=.5,verify_checkpoint=True,inherit_teacher_weights=False,
           use_lora=True)
    base=TinyHF(PretrainedConfig()).bfloat16()
    inherited={k:v.detach().clone() for k,v in base.state_dict().items()}
    live=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=["linear"]))
    live.get_base_model().action_queries.weight.requires_grad_(True)
    optimizer=torch.optim.AdamW([p for p in live.parameters() if p.requires_grad],lr=.01)
    class Processor:
        def save_pretrained(self,path):
            (Path(path)/"preprocessor_config.json").write_text("{}")
    def save_stats(stats,path):
        (Path(path)/"dataset_statistics.json").write_text(json.dumps(stats))
    ns=dict(torch=torch,Path=Path,copy=copy,json=json,shutil=shutil,
            OpenVLAForActionPrediction=TinyHF,PeftModel=PeftModel,
            FEATURE_CONTRACT=contract.FEATURE_CONTRACT,
            load_complete_state_dict=pruning.load_complete_state_dict,
            load_checkpoint_state_dict=pruning.load_checkpoint_state_dict,
            save_dataset_statistics=save_stats,dist=NS(barrier=lambda:None),
            __file__=str(ROOT/"vla-scripts/finetune.py"))
    actual_function("vla-scripts/finetune.py","remove_ddp_in_checkpoint",ns)
    actual_function("vla-scripts/finetune.py","load_component_from_dir",ns)
    save=actual_function("vla-scripts/finetune.py","save_training_checkpoint",ns)
    with tempfile.TemporaryDirectory() as temp:
        wrapped=NS(module=live)
        wrapped_head=NS(module=nn.Linear(2,2))
        wrapped_projector=NS(module=nn.Linear(2,2))
        stats={"libero_10_no_noops":{"action":{},"proprio":{}}}
        run=Path(temp)/"test"
        rng_before = torch.get_rng_state().clone()
        save(cfg,run,0,wrapped,Processor(),wrapped_projector,None,wrapped_head,
             NS(dataset_statistics=stats),NS(is_main_process=True),inherited)
        zero = Path(str(run)+"--0_chkpt")
        assert json.loads((zero/"config.json").read_text())["student_initialization"] == "random"
        zero_state = pruning.load_checkpoint_state_dict(zero)
        assert all(torch.equal(v, zero_state[k]) for k,v in inherited.items())
        assert torch.equal(rng_before, torch.get_rng_state())
        live().float().square().mean().backward(); optimizer.step()
        snapshot={k:v.detach().clone() for k,v in live.state_dict().items()}
        save(cfg,run,1,wrapped,Processor(),wrapped_projector,None,wrapped_head,
             NS(dataset_statistics=stats),NS(is_main_process=True),inherited)
        checkpoint=Path(str(run)+"--1_chkpt")
        assert (checkpoint/"checkpoint_complete.json").is_file()
        disk=pruning.load_checkpoint_state_dict(checkpoint)
        assert not any("lora_" in key for key in disk)
        torch.testing.assert_close(disk["action_queries.weight"],live.get_base_model().action_queries.weight)
        assert all(torch.equal(value,snapshot[key]) for key,value in live.state_dict().items())
        # Full-parameter random training saves the complete live student without
        # a PEFT wrapper and remains directly reloadable/trainable.
        cfg_full=copy.copy(cfg); cfg_full.use_lora=False
        full_live=TinyHF(PretrainedConfig()).bfloat16().requires_grad_(True)
        full_initial={k:v.detach().clone() for k,v in full_live.state_dict().items()}
        full_optimizer=torch.optim.AdamW(full_live.parameters(),lr=.01)
        full_live().float().square().mean().backward()
        assert all(p.grad is not None and p.grad.count_nonzero() for p in full_live.parameters())
        full_optimizer.step()
        full_before={k:v.detach().clone() for k,v in full_live.state_dict().items()}
        assert all(not torch.equal(v,full_initial[k]) for k,v in full_before.items())
        full_run=Path(temp)/"full"
        save(cfg_full,full_run,1,NS(module=full_live),Processor(),wrapped_projector,None,wrapped_head,
             NS(dataset_statistics=stats),NS(is_main_process=True),{})
        full_checkpoint=Path(str(full_run)+"--1_chkpt")
        full_disk=pruning.load_checkpoint_state_dict(full_checkpoint)
        assert all(torch.equal(v,full_disk[k]) for k,v in full_before.items())
        full_meta=json.loads((full_checkpoint/"config.json").read_text())
        assert full_meta["student_training_mode"] == "full_parameter"
        completion=json.loads((full_checkpoint/"checkpoint_complete.json").read_text())
        assert completion["full_trainable_student_weights"] is True
        assert not (full_checkpoint/"lora_adapter").exists()
    print("PASS: ACTUAL complete checkpoint saver for LoRA-merge and full-parameter students; readback passed")
    print("ALL CPU LOGIC TESTS PASSED. Real checkpoint/GPU/simulator preflight is still required.")


if __name__=="__main__":
    main()
