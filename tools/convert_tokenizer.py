import msgpack
import sys

idx2token = {}

with open(sys.argv[1], "r", encoding="utf-8") as f:
    lines = f.readlines()

for l in lines:
    idx = int(l[:l.index(' ')])
    x = eval(l[l.index(' '):l.rindex(' ')])
    x = x.encode("utf-8") if isinstance(x, str) else x
    assert isinstance(x, bytes)
    assert len(x) == int(l[l.rindex(' '):])
    idx2token[idx] = x

with open(sys.argv[2], "wb") as f:
    msgpack.pack(idx2token, f)

