import pytest

from findex4.index import CorpusDocument, build_index
from findex4.query import And, Not, Or, Phrase, QueryNode, Term, parse
from findex4.search import search

# --- 1. Параметризований тест для перевірки AST (Tree Equality) ---

@pytest.mark.parametrize(
    ("query_str", "expected_ast"),
    [
        ("a OR b c", Or(Term("a"), And(Term("b"), Term("c")))),
        ("python NOT java", And(Term("python"), Not(Term("java")))),
        (
            '(a OR b) AND "hello world"',
            And(
                Or(Term("a"), Term("b")),
                Phrase(("hello", "world")),
            ),
        ),
    ],
)
def test_parser_tree_equality(query_str: str, expected_ast: QueryNode) -> None:
    parsed = parse(query_str)
    assert parsed == expected_ast


# --- 2. Перевантаження операторів (Operator Overloading) ---

def test_operator_overloading() -> None:
    a = Term("a")
    b = Term("b")
    c = Term("c")

    tree = (a | b) & ~c
    expected = And(Or(a, b), Not(c))
    assert tree == expected


# --- 3. Фразовий пошук (Phrase Query Evaluation) ---

def test_phrase_query_evaluation() -> None:
    corpus: list[CorpusDocument] = [
        {
            "doc_id": 1,
            "title": "Doc 1",
            "path": "/1.txt",
            "tokens": ["quick", "brown", "fox"],
        },
        {
            "doc_id": 2,
            "title": "Doc 2",
            "path": "/2.txt",
            "tokens": ["brown", "quick", "fox"],
        },
    ]
    index = build_index(corpus, with_positions=True)
    phrase_ast = parse('"quick brown"')
    matched_docs = phrase_ast.evaluate(index)

    assert matched_docs == {1}


# --- 4. Повний булевий пошук у пошуковому рушії ---

def test_full_search_with_boolean_query() -> None:
    corpus: list[CorpusDocument] = [
        {
            "doc_id": 1,
            "title": "Doc 1",
            "path": "/1.txt",
            "tokens": ["python", "search", "engine"],
        },
        {
            "doc_id": 2,
            "title": "Doc 2",
            "path": "/2.txt",
            "tokens": ["python", "django", "web"],
        },
        {
            "doc_id": 3,
            "title": "Doc 3",
            "path": "/3.txt",
            "tokens": ["java", "spring", "web"],
        },
    ]
    index = build_index(corpus, with_positions=True)

    results = search("python NOT web", index)
    assert len(results) == 1
    assert results[0].doc_id == 1