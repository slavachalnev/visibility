import torch

from transformer_lens import HookedTransformer



def compute_grads(model: HookedTransformer, text, position, layer):

    def require_grads(value, hook):
        value = torch.tensor(value, requires_grad=True)
        return value
    
    cache = dict()
    def save_hook(value, hook):
        cache[hook.name] = value
        return value

    hooks = [(f'hook_embed', require_grads)]
    for layer in range(model.cfg.n_layers):
        hooks += [(f'blocks.{layer}.hook_resid_post', save_hook)]

    with model.hooks(fwd_hooks=hooks):
        model(text)

    print(cache.keys())
    h = cache[f'blocks.{layer}.hook_resid_post'][0, position, :]
    print(h.shape)

    # Compute gradients of h wrt every state that affects h.
    h.sum().backward()

    grads_h_wrt_others = dict()




model = HookedTransformer.from_pretrained('gelu-4l', device='cpu')
text = "The cat sat on the mat."


compute_grads(model, text, position=1, layer=1)
    












