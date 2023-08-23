import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    input_length = cache['blocks.0.hook_resid_post'].shape[1]

    h = cache[f'blocks.{layer}.hook_resid_post'][0, position, :]

    # Compute gradients of h wrt every state that affects h.
    h.abs().sum().backward()

    # grads_h_wrt_others = {i: torch.zeros(input_length) for i in range(model.cfg.n_layers)}
    grads_h_wrt_others = [torch.zeros(input_length) for _ in range(model.cfg.n_layers)]
    for layer_idx in range(layer): # 0, 1, ..., layer-1
        grads = cache[f'blocks.{layer_idx}.hook_resid_post'].grad
        # compute magnitude of grads per position
        grads_h_wrt_others[layer_idx] = grads[0].abs().sum(dim=-1)
        print(grads_h_wrt_others[layer_idx])
    
    return grads_h_wrt_others


model = HookedTransformer.from_pretrained('gelu-4l', device='cpu')
text = "The cat sat on the mat."


h_wrt_others = compute_grads(model, text, position=2, layer=1)
    
data = np.array([grad.numpy() for grad in reversed(h_wrt_others)])
positions = range(data.shape[1])
layers = range(len(h_wrt_others) - 1, -1, -1) # Reversed order of layers
# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data, xticklabels=positions, yticklabels=layers, cmap='YlGnBu')
plt.xlabel('Position')
plt.ylabel('Layer')
plt.title('Gradients of h with respect to Other States')
plt.show()