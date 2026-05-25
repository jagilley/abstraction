"""Inspect original vs denoised token sequences."""

import sys
import numpy as np
import tiktoken

def inspect(original_path: str, denoised_path: str, start: int = 0, length: int = 500):
    enc = tiktoken.get_encoding("gpt2")

    orig = np.load(original_path)
    den = np.load(denoised_path)

    chunk_o = orig[start:start + length]
    chunk_d = den[start:start + length]

    diff_mask = chunk_o != chunk_d
    n_diff = diff_mask.sum()

    print(f"=== ORIGINAL (tokens {start}..{start+length}) ===")
    print(enc.decode(chunk_o.tolist()))
    print(f"\n=== DENOISED (tau=0.3, {n_diff}/{length} changed) ===")
    print(enc.decode(chunk_d.tolist()))

    print(f"\n=== DIFF POSITIONS ===")
    for i in np.where(diff_mask)[0][:20]:
        o_tok = enc.decode([int(chunk_o[i])])
        d_tok = enc.decode([int(chunk_d[i])])
        print(f"  pos {start+i}: '{o_tok}' -> '{d_tok}'")
    if n_diff > 20:
        print(f"  ... and {n_diff - 20} more")

if __name__ == "__main__":
    orig = sys.argv[1] if len(sys.argv) > 1 else "/data/tokens/shard_00000.npy"
    den = sys.argv[2] if len(sys.argv) > 2 else "/data/denoised/tau_0.300/shard_00000.npy"
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    inspect(orig, den, start)
