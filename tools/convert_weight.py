#!/usr/bin/env python3

import sys

import torch
import msgpack


def convert_to_fr(input_path, output_path):
    w = torch.load(input_path, map_location=torch.device('cpu'))

    d = {'embd_weights': [], 'weights': {}}

    id_map = {}

    for k, v in w.items():
        if not isinstance(v, torch.Tensor):
            continue
        id_map[id(v)] = k
        if k == 'emb.weight':
            for x in v:
                d['embd_weights'].append(x)
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
            if(len(w[x].shape) > 1 and w[x].shape[1] > 1):
                version = max(5.2, version)

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

    msgpack.pack(d, open(output_path, 'wb'), default=pack)


if __name__ == '__main__':
    convert_to_fr(sys.argv[1], sys.argv[2])
