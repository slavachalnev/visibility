# %%
import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

import json
import time

from datasets import load_dataset

from transformer_lens import HookedTransformer
import transformer_lens.utils as utils


def get_cache(model: HookedTransformer, toks):
    cache = dict()

    def save_hook(value, hook):
        value.retain_grad()
        cache[hook.name] = value
        return value

    hooks = [(f'blocks.{layer_idx}.hook_resid_post', save_hook) for layer_idx in range(model.cfg.n_layers)]
    model.run_with_hooks(toks, fwd_hooks=hooks)
    return cache


def zero_grads(model: HookedTransformer, cache):
    model.zero_grad()
    for c in cache.values():
        if c.grad is not None:
            c.grad.zero_()


def compute_grads(model: HookedTransformer, toks, position, layer):
    cache = get_cache(model, toks)
    zero_grads(model, cache)
    input_length = cache['blocks.0.hook_resid_post'].shape[1]

    h = cache[f'blocks.{layer}.hook_resid_post'][0, position, :]
    h = h / torch.norm(h)

    # Compute gradients of h wrt every state that affects h.
    h.abs().sum().backward(retain_graph=True)

    grads_h_wrt_others = [torch.zeros(input_length) for _ in range(model.cfg.n_layers)]
    for layer_idx in range(layer): # 0, 1, ..., layer-1
        grads = cache[f'blocks.{layer_idx}.hook_resid_post'].grad
        # compute magnitude of grads per position
        grads_h_wrt_others[layer_idx] = grads[0].abs().sum(dim=-1)
    
    data = np.array([grad.numpy() for grad in reversed(grads_h_wrt_others)])
    return data

def compute_wrt_h(model: HookedTransformer, toks, position, layer):

    cache = get_cache(model, toks)
    zero_grads(model, cache)
    input_length = cache['blocks.0.hook_resid_post'].shape[1]

    res = torch.zeros(model.cfg.n_layers, input_length)
    for layer_idx in range(layer+1, model.cfg.n_layers): # layer+1, ..., n_layers-1
        for pos_idx in range(input_length):
            final = cache[f'blocks.{layer_idx}.hook_resid_post'][0, pos_idx, :]
            final = final / torch.norm(final)
            final = final.abs().sum()
            final.backward(retain_graph=True)

            g = cache[f'blocks.{layer}.hook_resid_post'].grad[0, position, :].abs().sum()
            res[layer_idx, pos_idx] = g
            zero_grads(model, cache)

    return res.numpy()


def compute_heatmap(model: HookedTransformer,
                    dataset: Dataset,
                    position: int,
                    layer: int,
                    average_over: int = 1,
                    toks_len: int = 20):
    n_layers = model.cfg.n_layers
    heatmap = np.zeros((n_layers, toks_len))
    
    for i, d in enumerate(dataset):
        if i == average_over:
            break

        # Truncate tokens to the specified length
        toks = d['tokens'][:toks_len]

        others_wrt_h = compute_wrt_h(model, toks, position=position, layer=layer)[::-1]
        h_wrt_others = compute_grads(model, toks, position=position, layer=layer)

        # Accumulate gradients
        heatmap += others_wrt_h + h_wrt_others

    return heatmap



model = HookedTransformer.from_pretrained('gelu-4l', device='cpu')
# model = HookedTransformer.from_pretrained('gpt2-small', device='cpu')

pile_data = load_dataset("NeelNanda/pile-10k", split="train")
dataset = utils.tokenize_and_concatenate(pile_data, model.tokenizer)

n_layers = model.cfg.n_layers
n_positions = 20

# %%
# Compute the heatmap for every layer, position
result = np.zeros((n_layers, n_positions, n_layers, n_positions))
for layer in range(n_layers):
    for position in range(n_positions):
        t0 = time.time()
        heatmap = compute_heatmap(model, dataset, position=position, layer=layer, average_over=1, toks_len=n_positions)
        print(f'Layer {layer}, position {position} took {time.time() - t0:.2f} seconds')
        result[layer, position] = heatmap

# save the result as json
with open('page/heatmaps.json', 'w') as f:
    result = np.flip(result, axis=2)
    json.dump(result.tolist(), f)



# %%
layer = 5
position = 10
data = compute_heatmap(model, dataset, position=position, layer=layer, average_over=1, toks_len=n_positions)

# %%
# Plot the heatmap
mask = np.zeros_like(data, dtype=bool)
mask[layer, position] = True
mask = np.flip(mask, axis=0)

cmap = sns.color_palette("YlGnBu", as_cmap=True)
custom_cmap = ListedColormap(cmap(np.arange(cmap.N)))
custom_cmap.set_bad(color='lightgrey')

positions = range(data.shape[1])
layers = range(n_layers - 1, -1, -1) # Reversed order of layers
plt.figure(figsize=(10, 6))
sns.heatmap(data, xticklabels=positions, yticklabels=layers, cmap=custom_cmap, mask=mask)

plt.xlabel('Position')
plt.ylabel('Layer')
plt.title('Gradients through h')
plt.show()

# %%
