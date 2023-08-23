import torch

from transformer_lens import HookedTransformer



def compute_grads(model: HookedTransformer, text, position, layer):

    cache = dict()
    def save_hook(value, hook):
        value.retain_grad()
        cache[hook.name] = value
        return value

    hooks = []
    for layer_idx in range(model.cfg.n_layers):
        hooks += [(f'blocks.{layer_idx}.hook_resid_post', save_hook)]

    with model.hooks(fwd_hooks=hooks):
        model(text)

    h = cache[f'blocks.{layer}.hook_resid_post'][0, position, :]

    # Compute gradients of h wrt every state that affects h.
    h.abs().sum().backward()

    grads_h_wrt_others = dict()
    for layer_idx in range(layer): # 0, 1, ..., layer-1
        grads_h_wrt_others[layer_idx] = cache[f'blocks.{layer_idx}.hook_resid_post'].grad
        print(grads_h_wrt_others[layer_idx].shape)
        print(grads_h_wrt_others[layer_idx])


model = HookedTransformer.from_pretrained('gelu-4l', device='cpu')
text = "The cat sat on the mat."


compute_grads(model, text, position=1, layer=1)
    
