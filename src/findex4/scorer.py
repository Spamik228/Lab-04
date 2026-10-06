import math
from typing import Protocol, runtime_checkable

from findex4.index import Index, Posting


@runtime_checkable
class Scorer(Protocol):

    def score(self, term: str, posting: Posting, index: Index) -> float:
        ...


class TfIdf:


    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or index.num_docs == 0:
            return 0.0


        idf = math.log(index.num_docs / df)
        return posting.tf * idf

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


class BM25:


    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or index.num_docs == 0:
            return 0.0


        num = index.num_docs - df + 0.5
        denom = df + 0.5
        idf = math.log(1.0 + (num / denom))

        doc_len = index.doc_length(posting.doc_id)
        avg_len = index.avg_doc_length

        if avg_len == 0.0:
            len_norm = 1.0
        else:
            len_norm = 1.0 - self.b + self.b * (doc_len / avg_len)

        tf_component = (posting.tf * (self.k1 + 1.0)) / (posting.tf + self.k1 * len_norm)

        return idf * tf_component

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)