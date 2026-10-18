import json

import numpy as np

from .components import ComponentSpace
from .embeddings import IMAGE_MODEL, text_identity
from .fonts import PHRASES


def write_results(folder, space, X, items, fit_rows, encoder):
    if len(X) != len(items):
        raise ValueError("Each embedding needs one matching metadata record")
    folder.mkdir(parents=True, exist_ok=True)
    scores = space.transform(X)
    space.save(folder / "model.npz")
    np.savez(folder / "scores.npz", scores=scores, fit_rows=fit_rows)
    (folder / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")

    with (folder / "profiles.jsonl").open("w") as target:
        for i, values in enumerate(scores):
            record = {"id": items[i]["id"], "components": values.tolist()}
            target.write(json.dumps(record, ensure_ascii=False) + "\n")

    examples = []
    for axis in range(space.dimensions):
        order = np.argsort(scores[:, axis])
        low = [{"id": items[i]["id"], "score": float(scores[i, axis])} for i in order[:5]]
        high = [{"id": items[i]["id"], "score": float(scores[i, axis])} for i in order[-5:][::-1]]
        examples.append({"component": axis + 1, "low": low, "high": high})
    (folder / "component_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    summary = {
        "encoder": encoder,
        "items": len(items),
        "fitting_items": len(fit_rows),
        "components": space.dimensions,
        "embedding_dimensions": X.shape[1],
        "warnings": space.messages,
    }
    (folder / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(items)} profiles with {space.dimensions} component scores to {folder}")
    return scores


def fit_text(data, output, dimensions=32):
    corpus = json.loads((data / "corpus.json").read_text())
    items = corpus["passages"]
    with np.load(data / "embeddings.npz", allow_pickle=False) as saved:
        if saved["digest"].item() != text_identity([p["text"] for p in items]):
            raise ValueError("The text changed; embed the prepared corpus again")
        X = saved["x"]
        encoder = json.loads(saved["info"].item())
    fit_rows = np.array(
        [i for i, item in enumerate(items) if item.get("split", "discovery") == "discovery"]
    )
    space = ComponentSpace(dimensions=dimensions, pca_dimensions=max(64, dimensions))
    space.fit(X, fit_rows)
    return write_results(output, space, X, items, fit_rows, encoder)


def fit_fonts(data, output, dimensions=16):
    metadata = json.loads((data / "font-metadata.json").read_text())
    fonts = metadata["fonts"]
    expected = [IMAGE_MODEL, PHRASES, fonts, "render-v1"]
    import hashlib

    identity = hashlib.sha256(json.dumps(expected).encode()).hexdigest()
    with np.load(data / "embeddings.npz", allow_pickle=False) as saved:
        if saved["identity"].item() != identity:
            raise ValueError("The font collection changed; embed it again")
        tensor = saved["x"]
        revision = saved["revision"].item()
    if tensor.shape[:2] != (len(fonts), len(PHRASES)):
        raise ValueError("Unexpected font embedding layout")

    # use the original discovery families and three phrases; every image still gets a score
    order = np.random.default_rng(2026).permutation(len(fonts))
    fit_families = order[: int(len(fonts) * 0.7)]
    fit_rows = np.array(
        [family * len(PHRASES) + phrase for family in fit_families for phrase in range(3)]
    )
    X = tensor.reshape(-1, tensor.shape[-1])
    space = ComponentSpace(
        dimensions=dimensions, pca_dimensions=dimensions, solver="full", seed=7, max_iter=2000
    )
    raw = space.fit_transform(X, fit_rows).reshape(len(fonts), len(PHRASES), dimensions)
    space.score_scale = np.maximum(raw[fit_families, :3].mean(axis=1).std(axis=0), 1e-12)
    items = []
    for family, font in enumerate(fonts):
        for phrase, text in enumerate(PHRASES):
            index = family * len(PHRASES) + phrase
            items.append(
                {
                    "id": index,
                    "font": font["name"],
                    "font_id": font["slug"],
                    "text": text,
                    "phrase": phrase,
                    "image": str(data / "images" / f"{font['slug']}-{phrase}.png"),
                    "split": "discovery" if index in fit_rows else "held-out",
                }
            )
    encoder = {"model": IMAGE_MODEL, "revision": revision}
    return write_results(output, space, X, items, fit_rows, encoder)
