"""FineWeb-Edu tokenization into the shared `{DATA_DIR}/tokens` corpus.

`experiments/pipeline/stages.py::tokenize` imports `tokenize_fineweb` from this
module, but the module itself was missing from the repo, so the stage could not
run. This restores it against the on-disk format every consumer already expects
(a2a_forward/stages.py:57, language_ratchet.py:134, analyze.py:60, ...):

    {tokens_dir}/shard_{idx:05d}.npy   uint16 token ids, `shard_size` each
    {tokens_dir}/meta.npy              dict: n_shards, shard_size, total_tokens,
                                             encoding, vocab_size

Encoding is tiktoken's GPT-2 BPE (vocab 50257), which is what makes the
canonical 4L/4H/256D language model come out at ~28.9M params as reported in
a2a_forward/LANGUAGE_RATCHET_README.md.

uint16 holds ids up to 65535, which covers GPT-2's 50257; the assert below keeps
that assumption honest if the encoding is ever swapped.
"""

import os

import numpy as np

__all__ = ["tokenize_fineweb"]


def tokenize_fineweb(tokens_dir, dataset_name, dataset_config, n_tokens,
                     shard_size, encoding_name="gpt2", eot_between_docs=True):
    """Stream a dataset, BPE-encode it, and write packed uint16 shards.

    Documents are concatenated with an end-of-text delimiter between them and
    then packed into fixed-size shards, so no padding is wasted and shard
    boundaries carry no meaning.
    """
    import tiktoken
    from datasets import load_dataset

    os.makedirs(tokens_dir, exist_ok=True)
    enc = tiktoken.get_encoding(encoding_name)
    vocab_size = enc.n_vocab
    assert vocab_size <= np.iinfo(np.uint16).max + 1, (
        f"encoding {encoding_name} has vocab {vocab_size}, too wide for uint16")
    eot = enc.eot_token if eot_between_docs else None

    print(f"=== Tokenizing {dataset_name}/{dataset_config} ===")
    print(f"  encoding={encoding_name} (vocab {vocab_size}), "
          f"target {n_tokens:,} tokens, shard_size {shard_size:,}")

    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    buf, shard_idx, total = [], 0, 0
    written_shard_sizes = []

    def flush(final=False):
        nonlocal buf, shard_idx
        while len(buf) >= shard_size or (final and buf):
            take = buf[:shard_size] if len(buf) >= shard_size else buf
            arr = np.asarray(take, dtype=np.uint16)
            path = os.path.join(tokens_dir, f"shard_{shard_idx:05d}.npy")
            np.save(path, arr)
            written_shard_sizes.append(int(arr.size))
            print(f"  shard {shard_idx:05d}: {arr.size:,} tokens "
                  f"({total:,} total)")
            del buf[:arr.size]
            shard_idx += 1
            if not final:
                break

    BATCH = 512
    texts = []
    for rec in ds:
        texts.append(rec["text"])
        if len(texts) < BATCH:
            continue
        for ids in enc.encode_ordinary_batch(texts):
            buf.extend(ids)
            if eot is not None:
                buf.append(eot)
            total += len(ids) + (1 if eot is not None else 0)
        texts = []
        flush()
        if total >= n_tokens:
            break

    if total < n_tokens and texts:
        for ids in enc.encode_ordinary_batch(texts):
            buf.extend(ids)
            if eot is not None:
                buf.append(eot)
            total += len(ids) + (1 if eot is not None else 0)

    flush(final=True)

    meta = {
        "n_shards": shard_idx,
        "shard_size": shard_size,
        "total_tokens": int(sum(written_shard_sizes)),
        "encoding": encoding_name,
        "vocab_size": vocab_size,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
    }
    np.save(os.path.join(tokens_dir, "meta.npy"), meta)
    print(f"  wrote {shard_idx} shards, {meta['total_tokens']:,} tokens")
    return meta
