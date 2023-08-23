import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchviz import make_dot

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
    h.abs().sum().backward(retain_graph=True)

    grads_h_wrt_others = [torch.zeros(input_length) for _ in range(model.cfg.n_layers)]
    for layer_idx in range(layer): # 0, 1, ..., layer-1
        grads = cache[f'blocks.{layer_idx}.hook_resid_post'].grad
        # compute magnitude of grads per position
        grads_h_wrt_others[layer_idx] = grads[0].abs().sum(dim=-1)
    
    return grads_h_wrt_others

def compute_wrt_h(model: HookedTransformer, text, position, layer):
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
    res = torch.zeros(model.cfg.n_layers, input_length)

    for layer_idx in range(layer+1, model.cfg.n_layers): # layer+1, ..., n_layers-1
        for pos_idx in range(input_length):
            final = cache[f'blocks.{layer_idx}.hook_resid_post'][0, pos_idx, :].abs().sum()
            final.backward(retain_graph=True)

            g = cache[f'blocks.{layer}.hook_resid_post'].grad[0, position, :].abs().sum()
            res[layer_idx, pos_idx] = g

            model.zero_grad()
            for c in cache.values():
                if c.grad is not None:
                    c.grad.zero_()
    return res.numpy()


model = HookedTransformer.from_pretrained('gelu-4l', device='cpu')
text = "The cat sat on the mat."

others_wrt_h = compute_wrt_h(model, text, position=2, layer=1)
# reverse 0th dim of others_wrt_h to align with the order of layers
others_wrt_h = others_wrt_h[::-1]

h_wrt_others = compute_grads(model, text, position=2, layer=1)


data = np.array([grad.numpy() for grad in reversed(h_wrt_others)])
data += others_wrt_h
positions = range(data.shape[1])
layers = range(len(h_wrt_others) - 1, -1, -1) # Reversed order of layers
# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data, xticklabels=positions, yticklabels=layers, cmap='YlGnBu')
plt.xlabel('Position')
plt.ylabel('Layer')
plt.title('Gradients through h')
plt.show()

"""
# Concatenating the two gradient data
full_data = np.array([grad.numpy() for grad in reversed(h_wrt_others + others_wrt_h)])
positions = range(full_data.shape[1])
# Reversed order of layers to align with the previous part
full_layers = range(len(h_wrt_others + others_wrt_h) - 1, -1, -1) 

# Plotting the full heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(full_data, xticklabels=positions, yticklabels=full_layers, cmap='YlGnBu')
plt.xlabel('Position')
plt.ylabel('Layer')
plt.title('Gradients of h with respect to Other States & Gradients of Other States with respect to h')
plt.show()
"""