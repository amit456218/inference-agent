#!/usr/bin/env python3
"""Tiny GGUF metadata reader for debugging. Usage:
   gguf_meta.py model.gguf key            -> print a metadata value
   gguf_meta.py model.gguf token ID...    -> print vocab text/type for token ids
   gguf_meta.py model.gguf find TEXT      -> find token ids whose text equals TEXT"""
import struct, sys

def read(path):
    f = open(path, "rb")
    def u32(): return struct.unpack("<I", f.read(4))[0]
    def u64(): return struct.unpack("<Q", f.read(8))[0]
    def s():  return f.read(u64()).decode("utf-8", "replace")
    def scalar(t):
        fmt = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<B",10:"<Q",11:"<q",12:"<d"}
        if t == 8: return s()
        return struct.unpack(fmt[t], f.read(struct.calcsize(fmt[t])))[0]
    assert f.read(4) == b"GGUF"
    ver = u32(); nt = u64(); nkv = u64()
    kv = {}
    for _ in range(nkv):
        k = s(); t = u32()
        if t == 9:
            et = u32(); n = u64()
            kv[k] = [scalar(et) for _ in range(n)]
        else:
            kv[k] = scalar(t)
    return kv

kv = read(sys.argv[1])
cmd = sys.argv[2]
if cmd == "key":
    print(kv[sys.argv[3]])
elif cmd == "token":
    toks, types = kv["tokenizer.ggml.tokens"], kv["tokenizer.ggml.token_type"]
    for a in sys.argv[3:]:
        i = int(a); print(i, repr(toks[i]), "type", types[i])
elif cmd == "find":
    toks = kv["tokenizer.ggml.tokens"]
    print([i for i, t in enumerate(toks) if t == sys.argv[3]])
