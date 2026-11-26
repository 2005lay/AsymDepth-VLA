"""Shared bridge contract, retaining the released teacher's feature convention.

The official checkpoint uses causal predecessor states for its 64 action slots
and hidden[:, :P] for task conditioning.  The latter includes text[0] and omits
the last patch; it is NOT exactly the pure-image slice hidden[:, 1:P+1].  Keep
this convention for teacher compatibility, and use it identically at train and
test time. Hidden state 0 is the embedding output; decoder outputs start at 1.
"""

FEATURE_CONTRACT = "official-predecessor-v1"


def policy_hidden_indices(num_vlm_layers, num_policy_layers):
    if num_vlm_layers < 1 or num_policy_layers < 1:
        raise ValueError("Both networks must contain at least one layer")
    return tuple(1 + min(i * num_vlm_layers // num_policy_layers, num_vlm_layers - 1)
                 for i in range(num_policy_layers))


def extract_bridge_states(hidden_states, action_labels, num_patches, num_tokens):
    """Padding-safe official predecessor features, shared by all forward paths."""
    import torch
    from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX

    mask = action_labels[:, 1:] > ACTION_TOKEN_BEGIN_IDX
    counts = mask.sum(dim=1)
    if not torch.all(counts == num_tokens).item():
        raise ValueError(f"Expected {num_tokens} action slots per sample, got {counts.tolist()}")
    collected = []
    for item in hidden_states:
        if item.shape[:2] != (mask.shape[0], mask.shape[1] + 1 + num_patches):
            raise ValueError(f"Hidden/label shape mismatch: {item.shape}, {action_labels.shape}")
        text = item[:, num_patches:-1]
        actions = text[mask].reshape(item.shape[0], num_tokens, item.shape[-1])
        collected.append(torch.cat((item[:, :num_patches], actions), dim=1))
    return torch.stack(collected, dim=1)
