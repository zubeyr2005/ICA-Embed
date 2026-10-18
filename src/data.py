import hashlib
import json
from pathlib import PurePosixPath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np


BOOKS = [
    ("Harry Potter and the Sorcerer’s Stone", "THE BOY WHO LIVED", 17),
    ("Harry Potter and the Chamber of Secrets", "THE WORST BIRTHDAY", 18),
    ("Harry Potter and the Prisoner of Azkaban", "OWL POST", 22),
    ("Harry Potter and the Goblet of Fire", "THE RIDDLE HOUSE", 37),
    ("Harry Potter and the Order of the Phoenix", "DUDLEY DEMENTED", 38),
    ("Harry Potter and the Half-Blood Prince", "THE OTHER MINISTER", 30),
    ("Harry Potter and the Deathly Hallows", "THE DARK LORD ASCENDING", 37),
]
BLOCK_TAGS = {
    "p",
    "div",
    "li",
    "td",
    "th",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "section",
    "article",
    "ul",
    "ol",
    "table",
    "tr",
}


def tag_name(node):
    return node.tag.split("}")[-1]


def text_blocks(body):
    paragraphs = []
    first_heading = True

    def append(text):
        text = " ".join(text.split())
        chapter_number = re.fullmatch(r"CHAPTER [A-Z -]+", text.strip("—–- "))
        if text and not chapter_number:
            paragraphs.append(text)

    def walk(node):
        nonlocal first_heading
        tag = tag_name(node)
        if tag in {"script", "style", "nav"}:
            return
        if tag in {"h1", "h2", "h3"} and first_heading:
            first_heading = False
            return

        # read a whole paragraph/list item so inline formatting does not split words
        descendants = list(node.iter())[1:]
        nested_blocks = any(tag_name(child) in BLOCK_TAGS for child in descendants)
        if tag in BLOCK_TAGS and not nested_blocks:
            append("".join(node.itertext()))
            return

        append(node.text or "")
        for child in node:
            walk(child)
            append(child.tail or "")

    walk(body)
    return paragraphs


def epub_documents(source):
    with ZipFile(source) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        rootfile = next(node for node in container.iter() if tag_name(node) == "rootfile")
        opf_path = rootfile.attrib["full-path"]
        opf = ET.fromstring(archive.read(opf_path))
        metadata = {}
        for name in ("title", "creator", "language"):
            metadata[name] = [node.text for node in opf.iter() if tag_name(node) == name]
        manifest = {
            node.attrib["id"]: node.attrib for node in opf.iter() if tag_name(node) == "item"
        }

        documents = []
        for item in opf.iter():
            if tag_name(item) != "itemref":
                continue
            entry = manifest[item.attrib["idref"]]
            if not entry.get("media-type", "").endswith("xhtml+xml"):
                continue
            path = str(PurePosixPath(opf_path).parent / entry["href"])
            tree = ET.fromstring(archive.read(path))
            headings = []
            for node in tree.iter():
                if tag_name(node) in {"h1", "h2", "h3"}:
                    headings.append(" ".join("".join(node.itertext()).split()))
            body = next(node for node in tree.iter() if tag_name(node) == "body")
            documents.append({"file": path, "headings": headings, "paragraphs": text_blocks(body)})
    return metadata, documents


def inspect(source):
    metadata, documents = epub_documents(source)
    print(json.dumps(metadata, indent=2))
    for document in documents:
        words = sum(len(p.split()) for p in document["paragraphs"])
        print(document["file"], document["headings"][:1], words)


def chunk_paragraphs(paragraphs):
    units = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        if len(paragraph.split()) <= 400:
            units.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[“\"‘'A-Z])", paragraph)
        for sentence in sentences:
            words = sentence.split()
            units.extend(" ".join(words[i : i + 350]) for i in range(0, len(words), 350))

    chunks, current, count = [], [], 0
    for unit in units:
        n = len(unit.split())
        if current and (count + n > 400 or (count >= 240 and count + n > 340)):
            chunks.append("\n\n".join(current))
            current, count = [], 0
        current.append(unit)
        count += n
    if current:
        tail = "\n\n".join(current)
        if chunks and count < 100 and len(chunks[-1].split()) + count <= 450:
            chunks[-1] += "\n\n" + tail
        else:
            chunks.append(tail)

    # chunking must preserve the words and their order, without overlap
    original = " ".join(" ".join(paragraphs).split())
    combined = " ".join(" ".join(chunks).split())
    if original != combined:
        raise ValueError("Chunking dropped, duplicated, or reordered text")
    return chunks


def read_chapters(documents):
    starts = {heading: i for i, (_, heading, _) in enumerate(BOOKS)}
    chapters, skipped = [], []
    counts = [0] * 7
    book = None
    for document in documents:
        words = sum(len(p.split()) for p in document["paragraphs"])
        if not document["headings"] or words < 1000:
            skipped.append({"file": document["file"], "words": words})
            continue
        heading = document["headings"][0]
        if heading in starts:
            book = starts[heading]
        if book is None:
            raise ValueError("Narrative content appeared before the first book")
        counts[book] += 1
        chapters.append(
            {
                "book": book,
                "book_title": BOOKS[book][0],
                "chapter": counts[book],
                "title": heading,
                "file": document["file"],
                "words": words,
                "paragraphs": [p for p in document["paragraphs"] if p],
            }
        )
    if counts != [book[2] for book in BOOKS]:
        raise ValueError(f"Unexpected chapter coverage: {counts}")
    return chapters, skipped


def write_boundaries(passages, folder):
    lines = ["# Extraction boundaries", "", "First and last narrative text from each book.", ""]
    for book in range(7):
        items = [p for p in passages if p["book"] == book]
        first, last = items[0], items[-1]
        lines += [
            f"## {BOOKS[book][0]}",
            "",
            f"First chapter: {first['chapter_title']}",
            " ".join(first["text"].split()[:65]),
            "",
            f"Last chapter: {last['chapter_title']}",
            " ".join(last["text"].split()[-65:]),
            "",
        ]
    (folder / "extraction-boundaries.md").write_text("\n".join(lines) + "\n")


def prepare(source, folder):
    folder.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() != ".epub":
        return prepare_documents(source, folder)
    metadata, documents = epub_documents(source)
    chapters, skipped = read_chapters(documents)

    # leave out whole chapters, rather than neighbouring chunks of the same scene
    rng = np.random.default_rng(2026)
    heldout = set()
    for book in range(7):
        indices = [i for i, chapter in enumerate(chapters) if chapter["book"] == book]
        selected = rng.choice(indices, size=max(1, round(len(indices) * 0.2)), replace=False)
        heldout.update(int(i) for i in selected)

    passages = []
    for chapter_id, chapter in enumerate(chapters):
        for chunk_id, text in enumerate(chunk_paragraphs(chapter["paragraphs"])):
            passages.append(
                {
                    "id": len(passages),
                    "book": chapter["book"],
                    "book_title": chapter["book_title"],
                    "chapter": chapter["chapter"],
                    "chapter_title": chapter["title"],
                    "chapter_id": chapter_id,
                    "chunk": chunk_id + 1,
                    "file": chapter["file"],
                    "text": text,
                    "words": len(text.split()),
                    "split": "held-out" if chapter_id in heldout else "discovery",
                }
            )
    total_words = sum(chapter["words"] for chapter in chapters)
    if sum(p["words"] for p in passages) != total_words:
        raise ValueError("Narrative word coverage mismatch")

    books = []
    for i, (title, _, count) in enumerate(BOOKS):
        books.append(
            {
                "number": i + 1,
                "title": title,
                "chapters": count,
                "words": sum(c["words"] for c in chapters if c["book"] == i),
                "passages": sum(p["book"] == i for p in passages),
            }
        )
    result = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "metadata": metadata,
        "books": books,
        "narrative_words": total_words,
        "chapters": len(chapters),
        "heldout_chapters": len(heldout),
        "passages": passages,
        "skipped_spine_documents": skipped,
        "extraction": "Narrative text blocks, including lists, letters and embedded headings. "
        "Chapter titles and front matter are excluded. Non-overlapping paragraph-aware "
        "passages; whole chapters held out by book with seed 2026.",
    }
    (folder / "corpus.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_boundaries(passages, folder)
    print(
        f"{len(books)} books, {len(chapters)} chapters, {len(passages)} passages, {total_words:,} words"
    )
    return result


def prepare_documents(source, folder):
    if source.suffix.lower() == ".txt":
        records = [{"text": source.read_text()}]
    elif source.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    elif source.suffix.lower() == ".json":
        records = json.loads(source.read_text())
    else:
        raise ValueError(
            "Use the Harry Potter EPUB, a text file, or a JSON/JSONL document collection"
        )
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty list of documents")

    passages = []
    for document_id, record in enumerate(records):
        record = {"text": record} if isinstance(record, str) else record
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError("Each document needs a text string")
        paragraphs = re.split(r"\n\s*\n", record["text"])
        for chunk, text in enumerate(chunk_paragraphs(paragraphs)):
            passages.append(
                {
                    **record,
                    "id": len(passages),
                    "source_id": record.get("id", document_id),
                    "text": text,
                    "chunk": chunk + 1,
                    "chapter_id": document_id,
                    "book": record.get("book", 0),
                    "words": len(text.split()),
                    "split": record.get("split", "discovery"),
                }
            )
    if not passages:
        raise ValueError("No nonempty text found")
    result = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "passages": passages,
        "documents": len(records),
        "narrative_words": sum(p["words"] for p in passages),
    }
    (folder / "corpus.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(records)} documents, {len(passages)} passages")
    return result
