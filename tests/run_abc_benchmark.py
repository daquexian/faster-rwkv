import math
import sys
import subprocess
import os
import numpy as np

import msgpack

from datasets import load_dataset

# copy from ChatRWKV
raw_data = load_dataset("sander-wood/irishman")["validation"]
data = []
for item in raw_data:
    data.append(item)
eval_set = [item["control code"]+item["abc notation"][item["abc notation"].index('\n')+1:] for item in data]
msgpack.pack(eval_set, open('eval_set', 'wb'))

env = os.environ.copy()
env["LIMIT"] = "10"

subprocess.check_call([sys.argv[1], sys.argv[2], "ncnn fp32", "eval_set"], env=env)

probs_vec = msgpack.unpack(open("abc_probs", 'rb'))

cnt = 0
sum = 0
xxx = 0

for probs in probs_vec:
    for prob in probs:
        sum -= math.log(prob)
        cnt += 1
    xxx += 1
    print(f'{xxx}_{round(sum/cnt, 8)}')

assert np.allclose(sum/cnt, 0.2698, atol=1e-4)

