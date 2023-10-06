import msgpack
import sys

d = {"version": "1", "type": "SimpleABCTokenizer"}

with open(sys.argv[1], "wb") as f:
    msgpack.pack(d, f)
