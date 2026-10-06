from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import NotRequired, TypedDict

from findex4.utils import timed


class CorpusDocument(TypedDict):
    doc_id: int
    tokens: list[str]
    path: NotRequired[str]
    title: NotRequired[str]


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


class Index(Mapping[str, list[Posting]]):
    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
    ) -> None:
        self._postings = postings
        self._doc_lengths = doc_lengths
        self._doc_meta = doc_meta

    # --- Dunder / Protocol methods ---

    def __len__(self) -> int:
        return len(self._postings)

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __repr__(self) -> str:
        return (
            f"Index(terms={len(self)}, "
            f"num_docs={self.num_docs}, "
            f"avg_doc_length={self.avg_doc_length:.2f})"
        )

    @property
    def num_docs(self) -> int:
        return len(self._doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def doc_length(self, doc_id: int) -> int:
        return self._doc_lengths.get(doc_id, 0)

    def df(self, term: str) -> int:
        if term in self._postings:
            return len(self._postings[term])
        return 0


@timed
def build_index(
    corpus: Iterable[CorpusDocument],
    with_positions: bool = False,
) -> Index:
    postings: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc in corpus:
        doc_id: int = doc["doc_id"]
        tokens: list[str] = doc["tokens"]

        doc_meta[doc_id] = DocMeta(
            doc_id=doc_id,
            path=doc.get("path", ""),
            title=doc.get("title", ""),
        )
        doc_lengths[doc_id] = len(tokens)

        if with_positions:
            term_positions: dict[str, list[int]] = defaultdict(list)
            for pos, token in enumerate(tokens):
                term_positions[token].append(pos)

            for term, positions in term_positions.items():
                postings[term].append(
                    Posting(
                        doc_id=doc_id,
                        tf=len(positions),
                        positions=tuple(positions),
                    )
                )
        else:
            counts = Counter(tokens)
            for term, tf in counts.items():
                postings[term].append(Posting(doc_id=doc_id, tf=tf))

    return Index(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
    )