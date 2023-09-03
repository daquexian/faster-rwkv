# WIP, do not use it.
import msgpack
import sys
import json

idx2token = {}

with open(sys.argv[1], "r", encoding="utf-8") as f:
    j = json.load(f)
    token2idx = j["model"]["vocab"]

idx2token = {v: k for k, v in token2idx.items()}

with open(sys.argv[2], "wb") as f:
    msgpack.pack(idx2token, f)
