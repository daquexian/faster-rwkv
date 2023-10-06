import msgpack
import sys
import json

idx2token = {}

with open(sys.argv[1], "r", encoding="utf-8") as f:
    j = json.load(f)
    token2idx = j["model"]["vocab"]

idx2token = {v: k for k, v in token2idx.items()}

d = {'idx2word': idx2token, 'normalizer': j["normalizer"]["type"],
     "pre_tokenizer": j["pre_tokenizer"]["type"], "version": "1", "type": "NormalTokenizer"}

with open(sys.argv[2], "wb") as f:
    msgpack.pack(d, f)
