import json
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from findex4.index import DocMeta, Index, Posting
from findex4.utils import timed


def save_pickle(index: Index, path: Path | str) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path | str) -> Index:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(index: Index, path: Path | str) -> None:
    postings = getattr(index, "_postings", {})
    doc_lengths = getattr(index, "_doc_lengths", {})
    doc_meta = getattr(index, "_doc_meta", {})

    data = {
        "doc_lengths": doc_lengths,
        "doc_meta": {
            str(k): {"doc_id": v.doc_id, "path": v.path, "title": v.title}
            for k, v in doc_meta.items()
        },
        "postings": {
            term: [
                {"doc_id": p.doc_id, "tf": p.tf, "positions": list(p.positions)}
                for p in posting_list
            ]
            for term, posting_list in postings.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_json(path: Path | str) -> Index:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_lengths = {int(k): v for k, v in data["doc_lengths"].items()}
    doc_meta = {
        int(k): DocMeta(doc_id=v["doc_id"], path=v["path"], title=v["title"])
        for k, v in data["doc_meta"].items()
    }
    postings = {
        term: [
            Posting(doc_id=p["doc_id"], tf=p["tf"], positions=tuple(p["positions"]))
            for p in posting_list
        ]
        for term, posting_list in data["postings"].items()
    }

    return Index(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
    )


def save(index: Index, path: Path | str, fmt: str = "pickle") -> None:
    if fmt == "pickle":
        save_pickle(index, path)
    elif fmt == "json":
        save_json(index, path)
    else:
        raise ValueError(f"Unknown format: {fmt}")

@timed
def load(path: Path | str, fmt: str = "pickle") -> Index:
    if fmt == "pickle":
        return load_pickle(path)
    elif fmt == "json":
        return load_json(path)
    else:
        raise ValueError(f"Unknown format: {fmt}")


@contextmanager
def open_index(path: Path | str, fmt: str = "pickle") -> Iterator[Index]:
    path = Path(path)
    if path.exists():
        index = load(path, fmt=fmt)
    else:
        index = Index(postings={}, doc_lengths={}, doc_meta={})

    try:
        yield index
    finally:
        save(index, path, fmt=fmt)