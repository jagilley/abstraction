"""Cache Llama 3.2 1B activations for A2A forward model training.

Stage 1 of the Llama-scale A2A experiment. Runs Llama inference on FineWeb-Edu,
extracts residual-stream activations at two adjacent layers, and saves them as
shards to the Modal volume. Stage 2 (training forward models of various sizes
on these cached activations) can then iterate quickly without re-running Llama.

Storage: ~8 GB per 1M tokens (two d=2048 layers at float16).
  10M tokens  ≈   80 GB
  100M tokens ≈  800 GB

Compute: ~30-60 min on A100 for 100M tokens (dominated by Llama inference).
"""

import modal
from a2a_forward.shared import app, volume, DATA_DIR

llama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "torch==2.7.0",
        "transformers",
        "huggingface-hub",
        "datasets",
        "accelerate",
    )
    .add_local_python_source("a2a_forward")
)


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L40S",
    timeout=14400,
    image=llama_image,
)
def a2a_cache_llama_acts(
    n_tokens: int = 100_000_000,
    model_name: str = "unsloth/Llama-3.2-1B",
    source_layer: int = 7,
    target_layer: int = 8,
    seq_len: int = 2048,
    batch_size: int = 16,
    shard_size: int = 1_000_000,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_subset: str = "sample-10BT",
):
    """Cache (source_layer, target_layer) activation pairs from Llama.

    Packs FineWeb-Edu text into fixed-length chunks (no padding waste), runs
    Llama inference with output_hidden_states=True, and saves shards of
    (source, target, token_ids) to the volume. Each shard is an npz with:
      source: float16 (shard_size, d_model) - post_block_{source_layer}
      target: float16 (shard_size, d_model) - post_block_{target_layer}
      token_ids: int32 (shard_size,) - for downstream analysis
    """
    import os
    import time
    import json
    import threading
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = "cuda"
    out_dir = f"{DATA_DIR}/a2a_llama/acts_L{source_layer}_L{target_layer}"
    os.makedirs(out_dir, exist_ok=True)

    storage_gb = n_tokens * 2048 * 2 * 2 / 1e9
    print(f"=== Caching Llama activations ===")
    print(f"  Model: {model_name}")
    print(f"  Layers: post_block_{source_layer} -> post_block_{target_layer}")
    print(f"  Target: {n_tokens:,} tokens (~{storage_gb:.0f} GB storage)")
    print(f"  seq_len={seq_len}, batch_size={batch_size}, shard_size={shard_size:,}")

    # --- Load model ---
    t0 = time.time()
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.2f}B params, {n_layers} layers, d={d_model}")
    print(f"  Load time: {time.time() - t0:.1f}s")

    assert source_layer < n_layers, f"source_layer {source_layer} >= n_layers {n_layers}"
    assert target_layer < n_layers, f"target_layer {target_layer} >= n_layers {n_layers}"
    assert target_layer > source_layer, "target_layer must be > source_layer"

    # --- Stream and tokenize dataset ---
    print(f"\nStreaming {dataset_name}/{dataset_subset}...")
    ds = load_dataset(dataset_name, dataset_subset, split="train", streaming=True)
    ds_iter = iter(ds)
    dataset_exhausted = False
    token_stream = []
    PREFETCH_DOCS = 256

    def refill_token_stream():
        """Batch-tokenize many documents at once into token_stream."""
        nonlocal dataset_exhausted
        if dataset_exhausted:
            return
        texts = []
        for _ in range(PREFETCH_DOCS):
            try:
                texts.append(next(ds_iter)["text"])
            except StopIteration:
                dataset_exhausted = True
                break
        if texts:
            encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
            for ids in encoded:
                token_stream.extend(ids)

    def next_chunk():
        """Pack tokens into a fixed-length chunk of seq_len."""
        while len(token_stream) < seq_len:
            if dataset_exhausted:
                return None
            refill_token_stream()
            if not token_stream and dataset_exhausted:
                return None
        chunk = token_stream[:seq_len]
        del token_stream[:seq_len]
        return chunk

    # --- Inference loop ---
    source_buf = []
    target_buf = []
    ids_buf = []
    buf_tokens = 0
    total_tokens = 0
    shard_idx = 0
    save_threads = []
    t_infer = time.time()

    def save_shard_bg(path, source, target, ids, idx):
        """Save a single shard to disk (runs in background thread)."""
        np.savez(path, source=source, target=target, token_ids=ids)
        volume.commit()
        elapsed = time.time() - t_infer
        rate = total_tokens / elapsed if elapsed > 0 else 0
        print(f"  Shard {idx:4d} saved | "
              f"{total_tokens:>12,} / {n_tokens:,} tokens | "
              f"{rate:,.0f} tok/s")

    def flush_shards():
        """Save complete shards from the buffer, keep the remainder."""
        nonlocal source_buf, target_buf, ids_buf, buf_tokens, shard_idx

        if buf_tokens == 0:
            return

        source_all = np.concatenate(source_buf, axis=0)
        target_all = np.concatenate(target_buf, axis=0)
        ids_all = np.concatenate(ids_buf, axis=0)

        while len(source_all) >= shard_size:
            path = os.path.join(out_dir, f"shard_{shard_idx:04d}.npz")
            s = source_all[:shard_size].astype(np.float16)
            t = target_all[:shard_size].astype(np.float16)
            i = ids_all[:shard_size].astype(np.int32)
            source_all = source_all[shard_size:]
            target_all = target_all[shard_size:]
            ids_all = ids_all[shard_size:]

            thread = threading.Thread(
                target=save_shard_bg, args=(path, s, t, i, shard_idx))
            thread.start()
            save_threads.append(thread)
            shard_idx += 1

        # Keep remainder
        if len(source_all) > 0:
            source_buf[:] = [source_all]
            target_buf[:] = [target_all]
            ids_buf[:] = [ids_all]
            buf_tokens = len(source_all)
        else:
            source_buf.clear()
            target_buf.clear()
            ids_buf.clear()
            buf_tokens = 0

    print("\nRunning inference...")
    with torch.no_grad():
        while total_tokens < n_tokens:
            # Build a batch of chunks
            batch_chunks = []
            for _ in range(batch_size):
                chunk = next_chunk()
                if chunk is None:
                    break
                batch_chunks.append(chunk)

            if not batch_chunks:
                print("Dataset exhausted.")
                break

            input_ids = torch.tensor(batch_chunks, dtype=torch.long, device=device)

            # hidden_states[0] = embedding, hidden_states[i+1] = post layer i
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            source_h = outputs.hidden_states[source_layer + 1]
            target_h = outputs.hidden_states[target_layer + 1]

            B, T, D = source_h.shape
            source_flat = source_h.reshape(-1, D).cpu().float().numpy()
            target_flat = target_h.reshape(-1, D).cpu().float().numpy()
            ids_flat = input_ids.reshape(-1).cpu().numpy()
            del outputs, source_h, target_h

            n_new = len(source_flat)
            source_buf.append(source_flat)
            target_buf.append(target_flat)
            ids_buf.append(ids_flat)
            buf_tokens += n_new
            total_tokens += n_new

            if buf_tokens >= shard_size:
                flush_shards()

    # Save any remaining tokens as a final (possibly smaller) shard
    if buf_tokens > 0:
        source_all = np.concatenate(source_buf, axis=0)
        target_all = np.concatenate(target_buf, axis=0)
        ids_all = np.concatenate(ids_buf, axis=0)
        path = os.path.join(out_dir, f"shard_{shard_idx:04d}.npz")
        np.savez(
            path,
            source=source_all.astype(np.float16),
            target=target_all.astype(np.float16),
            token_ids=ids_all.astype(np.int32),
        )
        elapsed = time.time() - t_infer
        rate = total_tokens / elapsed if elapsed > 0 else 0
        print(f"  Shard {shard_idx:4d} saved (final, {len(source_all):,} tokens) | "
              f"{total_tokens:>12,} total | {rate:,.0f} tok/s")
        shard_idx += 1
        volume.commit()

    # Wait for all background saves to complete
    for t in save_threads:
        t.join()

    # --- Save metadata ---
    elapsed_total = time.time() - t0
    elapsed_infer = time.time() - t_infer
    meta = {
        "model_name": model_name,
        "n_model_params": n_params,
        "source_layer": source_layer,
        "target_layer": target_layer,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_tokens_cached": total_tokens,
        "n_shards": shard_idx,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "shard_size": shard_size,
        "dataset": dataset_name,
        "dataset_subset": dataset_subset,
        "storage_gb_approx": total_tokens * d_model * 2 * 2 / 1e9,
        "inference_seconds": elapsed_infer,
        "total_seconds": elapsed_total,
        "tokens_per_sec": total_tokens / elapsed_infer if elapsed_infer > 0 else 0,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    volume.commit()

    print(f"\n=== Done ===")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Shards: {shard_idx}")
    print(f"  Storage: ~{meta['storage_gb_approx']:.0f} GB")
    print(f"  Inference: {elapsed_infer:.0f}s ({meta['tokens_per_sec']:,.0f} tok/s)")
    print(f"  Total: {elapsed_total:.0f}s")
    print(f"  Output: {out_dir}")

    return meta


@app.local_entrypoint()
def main(
    n_tokens: int = 100_000_000,
    llama_model: str = "unsloth/Llama-3.2-1B",
    source_layer: int = 7,
    target_layer: int = 8,
    seq_len: int = 2048,
    shard_size: int = 1_000_000,
):
    result = a2a_cache_llama_acts.remote(
        n_tokens=n_tokens,
        model_name=llama_model,
        source_layer=source_layer,
        target_layer=target_layer,
        seq_len=seq_len,
        shard_size=shard_size,
    )
    print(f"Llama activation caching complete:")
    print(f"  Tokens: {result['n_tokens_cached']:,}")
    print(f"  Shards: {result['n_shards']}")
    print(f"  Storage: ~{result['storage_gb_approx']:.0f} GB")
    print(f"  Throughput: {result['tokens_per_sec']:,.0f} tok/s")
