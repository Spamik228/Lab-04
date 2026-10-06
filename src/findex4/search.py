import functools
import heapq
from collections import defaultdict
from dataclasses import dataclass, field

from findex4.index import Index
from findex4.query import QueryNode, extract_terms, parse
from findex4.scorer import BM25, Scorer
from findex4.snippet import make_snippet
from findex4.utils import timed


@dataclass(frozen=True, order=True)
class SearchResult:
    score: float
    doc_id: int = field(compare=False)
    title: str = field(compare=False)
    snippet: str = field(default="", compare=False)


@functools.lru_cache(maxsize=256)
def _parse_cached(query_str: str) -> QueryNode:
    return parse(query_str)


@timed
def search(
    query: str | QueryNode,
    index: Index,
    scorer: Scorer | None = None,
    k: int = 10,
    include_snippets: bool = True,
    doc_texts: dict[int, str] | None = None,
) -> list[SearchResult]:
    if scorer is None:
        scorer = BM25()

    if isinstance(query, str):
        ast = _parse_cached(query)
    else:
        ast = query

    matched_doc_ids = ast.evaluate(index)

    if not matched_doc_ids:
        return []

    query_terms = extract_terms(ast)

    if not query_terms:
        results = [
            SearchResult(
                score=0.0,
                doc_id=doc_id,
                title=index._doc_meta[doc_id].title if doc_id in index._doc_meta else "",
                snippet="",
            )
            for doc_id in matched_doc_ids
        ]
        return heapq.nlargest(k, results, key=lambda res: res.score)

    doc_scores: dict[int, float] = defaultdict(float)

    for term in query_terms:
        if term not in index:
            continue

        for posting in index[term]:
            if posting.doc_id in matched_doc_ids:
                if callable(scorer) and not hasattr(scorer, "score"):
                    score_val = scorer(term, posting, index)
                else:
                    score_val = scorer.score(term, posting, index)
                doc_scores[posting.doc_id] += score_val

    results = []
    for doc_id in matched_doc_ids:
        title = index._doc_meta[doc_id].title if doc_id in index._doc_meta else ""
        text = doc_texts.get(doc_id, "") if doc_texts else ""

        snippet_text = ""
        if include_snippets and text:
            snippet_text = make_snippet(text, query_terms)

        results.append(
            SearchResult(
                score=doc_scores[doc_id],
                doc_id=doc_id,
                title=title,
                snippet=snippet_text,
            )
        )

    return heapq.nlargest(k, results, key=lambda res: res.score)