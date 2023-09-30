########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, types, json, math, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
with open("/data/user/cangshui/tianchao/repos/ChatRWKV/misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

########################################################################################################

# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230213-8019'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20230109-ctx4096'
# MODEL_NAME = '/data/user/cangshui/tianchao/pth_models/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096.pth'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023'

PAD_SEQ = []

########################################################################################################

print(f'\nLoading ChatRWKV https://github.com/BlinkDL/ChatRWKV')
import torch

# Tune these below (test True/False for all of them) to find the fastest setting:
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

from torch.nn import functional as F

MODEL_NAME = '/data/user/cangshui/tianchao/repos/fr-models/RWKV-4-World-0.1B-v1-20230520-ctx4096/native-fr/RWKV-4-World-0.1B-v1-20230520-ctx4096.fr'
TOKENIZER_NAME = '/data/user/cangshui/tianchao/repos/fr-models/RWKV-4-World-0.1B-v1-20230520-ctx4096/native-fr/'
print(f'Loading model - {MODEL_NAME}')
import fasterrwkv as fr
model = fr.Model(MODEL_NAME, 'cuda fp16')
sampler = fr.Tokenizer(TOKENIZER_NAME)

def forward_with_full_output(model, input_ids):
    outputs = []
    for i in range(len(input_ids)):
        output = model(input_ids[i])
        outputs.append(output)
    return torch.tensor(np.array(outputs))


print('Check LAMBADA...')
xsum = 0
xcnt = 0
xacc = 0
for d in todo:
    model.reset_states()
    src = PAD_SEQ + sampler.encode(d[0])
    dst = sampler.encode(d[1])

    logits = 0
    correct = True
    out = forward_with_full_output(model, src+dst)
    for i in range(len(dst)):
        probs = F.softmax(out[len(src)-1+i,:], dim=-1)
        logits += math.log(probs[dst[i]])
        if torch.argmax(probs).item() != dst[i]:
            correct = False

    xcnt += 1
    xsum += logits
    xacc += 1 if correct else 0
    if xcnt % 10 == 0 or xcnt == len(todo):
        print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2))
