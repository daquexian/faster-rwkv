#!/usr/bin/env python3

import sys

import torch
import numpy as np
import msgpack


w = torch.load(sys.argv[1], map_location=torch.device('cpu'))

d = {'embd_weights': [], 'weights': {}}

id_map = {}

# for k, v in w.items():
#     if k.endswith('ffn.value.weight') or k.endswith('ffn.receptance.weight'):
#         continue
#     if k.endswith('ffn.key.weight'):
#         pf = k[:-len('key.weight')]
#         kw = v
#         vw = w[pf + 'value.weight']
#         rw = w[pf + 'receptance.weight']
#         w[pf + 'kvr.weight'] = torch.stack([kw, vw, rw])
#         del w[pf + 'key.weight']
#         del w[pf + 'value.weight']
#         del w[pf + 'receptance.weight']

for k, v in w.items():
    if not isinstance(v, torch.Tensor):
        continue
    id_map[id(v)] = k
    if k == 'emb.weight':
        for x in v:
            d['embd_weights'].append(x.half())
    else:
        d['weights'][k] = v

n_layer = 0
version = 4
for x in w.keys():
    layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
    n_layer = max(n_layer, layer_id+1)
    if 'ln_x' in x:
        version = max(5, version)
    if 'gate.weight' in x:
        version = max(5.1, version)
    if int(version) == 5 and 'att.time_decay' in x:
        n_head = w[x].shape[0]
        d['n_head'] = n_head


d['version'] = str(version)
d['n_layer'] = n_layer
d['n_embd'] = w['emb.weight'].shape[1]
# weight has been transposed in chatrwkv conversion
d['n_att'] = w['blocks.0.att.key.weight'].shape[1]
d['n_ffn'] = w['blocks.0.ffn.key.weight'].shape[1]

def pack(x):
    if isinstance(x, torch.Tensor):
        return {'dtype': x.dtype, 'data': x.numpy().tobytes(), 'shape': x.shape}
    elif isinstance(x, torch.dtype):
        return str(x)
    return x

msgpack.pack(d, open(sys.argv[2], 'wb'), default=pack)
