import hashlib
import json
import time

import numpy as np

MODEL = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
IMAGE_MODEL = "google/siglip-base-patch16-224"


def text_identity(texts):
    description = [MODEL, texts, "plain-text-no-prompt-v1"]
    return hashlib.sha256(json.dumps(description).encode()).hexdigest()


def load_encoder():
    from mlx_embeddings import load

    model, tokenizer = load(MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def encode(texts, model, tokenizer):
    import mlx.core as mx

    inputs = tokenizer(texts, padding=True, return_tensors="np")
    ids = mx.array(inputs["input_ids"])
    mask = mx.array(inputs["attention_mask"])
    output = model(ids, attention_mask=mask).text_embeds
    mx.eval(output)
    X = np.array(output.astype(mx.float32))
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite text embedding")
    return X


def embed_text(folder, max_parts=None):
    corpus = json.loads((folder / "corpus.json").read_text())
    texts = [p["text"] for p in corpus["passages"]]
    digest = text_identity(texts)
    target = folder / "embeddings.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as old:
            if old["digest"].item() == digest:
                print("Using verified complete embedding cache")
                return target

    import mlx.core as mx

    parts = folder / "embedding-parts-0.6b"
    parts.mkdir(exist_ok=True)
    model, tokenizer = load_encoder()
    lengths = [len(tokenizer(text)["input_ids"]) for text in texts]
    if max(lengths) > 1024:
        raise ValueError("Split the long passages before embedding; do not truncate them")

    # the same passage should have the same vector alone and in a padded batch
    single = encode([texts[0]], model, tokenizer)[0]
    batched = encode(texts[:2], model, tokenizer)[0]
    consistency = float(
        np.dot(single, batched) / (np.linalg.norm(single) * np.linalg.norm(batched))
    )
    if consistency < 0.995:
        raise ValueError(f"Embedding batch inconsistency: {consistency}")

    started = time.monotonic()
    new_parts = 0
    for start in range(0, len(texts), 128):
        path = parts / f"{start:05d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if old["digest"].item() == digest:
                    continue
        end = min(start + 128, len(texts))
        batches = []
        for offset in range(start, end, 8):
            batches.append(encode(texts[offset : min(offset + 8, end)], model, tokenizer))
            mx.clear_cache()
        X = np.concatenate(batches)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        np.savez(path, x=X, digest=digest)
        new_parts += 1
        print(f"Checkpoint {end}/{len(texts)}; {time.monotonic() - started:.0f}s", flush=True)
        if max_parts is not None and new_parts >= max_parts:
            return

    batches = []
    for start in range(0, len(texts), 128):
        with np.load(parts / f"{start:05d}.npz", allow_pickle=False) as saved:
            if saved["digest"].item() != digest:
                raise ValueError("Mismatched checkpoint")
            batches.append(saved["x"])
    X = np.concatenate(batches).astype(float)
    if len(X) != len(texts) or not np.isfinite(X).all():
        raise ValueError("Incomplete or invalid embeddings")
    info = {
        "model": MODEL,
        "base_model": "Qwen/Qwen3-Embedding-0.6B",
        "quantization": "community 4-bit DWQ conversion",
        "dimensions": X.shape[1],
        "passages": len(texts),
        "longest_tokens": max(lengths),
        "truncated": 0,
        "prompt": None,
        "batch_consistency_cosine": consistency,
        "new_pass_seconds": time.monotonic() - started,
    }
    np.savez(target, x=X, digest=digest, info=json.dumps(info))
    return target


def embed_examples(pairs, path):
    texts = [text for examples in pairs.values() for pair in examples for text in pair]
    digest = hashlib.sha256(json.dumps([MODEL, texts]).encode()).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if saved["digest"].item() == digest:
                return saved["x"]
    model, tokenizer = load_encoder()
    batches = []
    for start in range(0, len(texts), 8):
        batches.append(encode(texts[start : start + 8], model, tokenizer))
    X = np.concatenate(batches).astype(float)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    np.savez(path, x=X, digest=digest)
    return X


def embed_images(files, cache, identity, shape, device=None):
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if saved["identity"].item() == identity:
                return saved["x"], saved["revision"].item()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, SiglipVisionModel

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL)
    model = SiglipVisionModel.from_pretrained(IMAGE_MODEL).eval().to(device)
    revision = getattr(model.config, "_commit_hash", None) or "unknown"
    batches = []
    for start in range(0, len(files), 12):
        images = []
        for path in files[start : start + 12]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            batches.append(model(**inputs).pooler_output.cpu().float().numpy())
        print(f"Embedded {min(start + 12, len(files))}/{len(files)} images", flush=True)
    X = np.concatenate(batches).reshape(*shape, -1).astype(float)
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite image embeddings")
    X /= np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)
    np.savez(cache, x=X, identity=identity, revision=revision)
    return X, revision
