from collections import Counter
import json

import numpy as np

from .components import ComponentSpace, percentiles
from .embeddings import embed_examples


def calibrate(space, coordinates, examples, properties, fitting_pairs=4):
    projected = space.transform(examples)
    selected, ranks = {}, {}
    offset = 0
    for name, pairs in properties.items():
        count = len(pairs)
        if count < fitting_pairs:
            raise ValueError(f"{name} needs at least {fitting_pairs} example pairs")
        paired = projected[offset : offset + 2 * count].reshape(count, 2, space.dimensions)
        changes = paired[:, 0] - paired[:, 1]
        effects = changes[:fitting_pairs].mean(axis=0)
        axis = int(np.argmax(abs(effects)))
        sign = float(np.sign(effects[axis]))
        ranks[name] = percentiles(coordinates[:, axis] * sign)
        selected[name] = {
            "component": axis + 1,
            "sign": sign,
            "check_pair_effects": (changes[fitting_pairs:, axis] * sign).tolist(),
        }
        offset += 2 * count
    return selected, ranks


def retrieve(items, properties, ranks, query):
    first, second = query["first"], query["second"]
    collision = properties[first]["component"] == properties[second]["component"]
    a = ranks[first] if query.get("first_sign", 1) == 1 else 1 - ranks[first]
    b = ranks[second] if query.get("second_sign", 1) == 1 else 1 - ranks[second]
    mode = query.get("mode", "intersection")
    if mode == "difference":
        # the original tension request: high tension and action in its lower half
        eligible = np.flatnonzero((a >= 0.8) & (b >= 0.5))
        combined = a + b - 1
        ordered = sorted(eligible, key=lambda i: combined[i], reverse=True)
    elif mode == "intersection":
        threshold = query.get("threshold", 0.7)
        eligible = np.flatnonzero((a >= threshold) & (b >= threshold))
        combined = np.minimum(a, b)
        ordered = sorted(eligible, key=lambda i: (combined[i], (a[i] + b[i]) / 2), reverse=True)
    else:
        raise ValueError(f"Unknown combination mode: {mode}")

    chosen, chapters, books = [], set(), Counter()
    if not collision:
        for index in ordered:
            item = items[index]
            chapter = item.get("chapter_id", item.get("source_id", item["id"]))
            book = item.get("book", 0)
            cap = query.get("per_book")
            if chapter in chapters or (cap is not None and books[book] >= cap):
                continue
            chapters.add(chapter)
            books[book] += 1
            chosen.append(int(index))
            if len(chosen) == query.get("count", 5):
                break
    records = [
        {
            **items[i],
            "score": float(combined[i]),
            "first_percentile": float(ranks[first][i]),
            "second_percentile": float(ranks[second][i]),
        }
        for i in chosen
    ]
    return {
        "name": query["name"],
        "first": properties[first],
        "second": properties[second],
        "same_component": collision,
        "eligible": len(eligible),
        "items": records,
    }


def run(folder, examples_path, cache, preset="all"):
    specification = json.loads(examples_path.read_text())
    properties = specification["properties"]
    for name, pairs in properties.items():
        if not pairs or any(
            len(pair) != 2 or any(not isinstance(t, str) for t in pair) for pair in pairs
        ):
            raise ValueError(f"Expected high/low description pairs for {name}")
    items = json.loads((folder / "items.json").read_text())
    space = ComponentSpace.load(folder / "model.npz")
    with np.load(folder / "scores.npz", allow_pickle=False) as saved:
        coordinates = saved["scores"]
    if len(items) != len(coordinates):
        raise ValueError("Scores do not match the item records")
    examples = embed_examples(properties, cache)
    selected, ranks = calibrate(
        space, coordinates, examples, properties, specification.get("fitting_pairs", 4)
    )
    queries = [q for q in specification["combinations"] if preset == "all" or q["name"] == preset]
    if not queries:
        raise ValueError(f"Unknown preset: {preset}")
    output = folder / "combinations"
    output.mkdir(exist_ok=True)
    for query in queries:
        result = retrieve(items, selected, ranks, query)
        (output / f"{query['name']}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        print(query["name"], [item["id"] for item in result["items"]])
    (output / "properties.json").write_text(json.dumps(selected, indent=2) + "\n")
