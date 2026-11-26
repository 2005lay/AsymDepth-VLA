"""Warm the exact DINOv2/SigLIP foundation cache used by the generic VLM."""

import gc
import os

# These must be set before importing timm/huggingface_hub.
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{os.environ['HF_HOME']}/hub")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

import timm
import torch


MODELS = (
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_so400m_patch14_siglip_224",
)


def main() -> None:
    print(
        "Vision cache configuration:",
        f"HF_HOME={os.environ['HF_HOME']}",
        f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}",
        f"HF_HUB_DISABLE_XET={os.environ['HF_HUB_DISABLE_XET']}",
        flush=True,
    )
    for model_id in MODELS:
        print(f"Caching {model_id} ...", flush=True)
        model = timm.create_model(model_id, pretrained=True, num_classes=0, img_size=224)
        state = model.state_dict()
        if not state or not all(torch.isfinite(value).all().item() for value in state.values() if torch.is_tensor(value)):
            raise RuntimeError(f"Invalid cached foundation weights: {model_id}")
        print(f"PASS: cached {model_id} ({len(state)} tensors)", flush=True)
        del state, model
        gc.collect()
    print("PASS: both generic foundation vision backbones are cached", flush=True)


if __name__ == "__main__":
    main()
