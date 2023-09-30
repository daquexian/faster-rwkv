########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json
import math
import os
import sys
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
with open(os.path.join(script_directory, "lambada_test.jsonl"), "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

todo = todo[:int(os.getenv("FR_LAMBADA_SIZE", 99999999999))]

TOKENIZER_NAME = sys.argv[1]
MODEL_NAME = sys.argv[2]

import fasterrwkv as fr
model = fr.Model(MODEL_NAME, 'cuda fp16')
sampler = fr.Tokenizer(TOKENIZER_NAME)

def forward_with_full_output(model, input_ids):
    outputs = []
    for i in range(len(input_ids)):
        output = model(input_ids[i])
        outputs.append(output)
    return np.array(outputs)


def softmax(x):
    assert len(x.shape) == 1
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


print('Check LAMBADA...')
xsum = 0
xcnt = 0
xacc = 0
for d in todo:
    model.reset_states()
    src = sampler.encode(d[0])
    dst = sampler.encode(d[1])

    logits = 0
    correct = True
    out = forward_with_full_output(model, src+dst)
    for i in range(len(dst)):
        probs = softmax(out[len(src)-1+i])
        logits += math.log(probs[dst[i]])
        if np.argmax(probs).item() != dst[i]:
            correct = False

    xcnt += 1
    xsum += logits
    xacc += 1 if correct else 0
    if xcnt % 10 == 0 or xcnt == len(todo):
        print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2))
