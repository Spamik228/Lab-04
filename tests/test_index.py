from pathlib import Path
import pytest
from hypothesis import given, strategies as st

from findex4.index import CorpusDocument, Index, build_index
from findex4.store import open_index


# --- 0. Фікстура sample_corpus ---

@pytest.fixture
def sample_corpus() -> list[CorpusDocument]:
    """Фікстура, яка надає тестовий корпус документів."""
    return [
        {
            "doc_id": 1,
            "tokens": ["cat", "dog", "cat"],
            "path": "/tmp/doc1.txt",
            "title": "Cat Document",
        },
        {
            "doc_id": 2,
            "tokens": ["dog", "bird"],
            "path": "/tmp/doc2.txt",
            "title": "Dog and Bird Document",
        },
        {
            "doc_id": 3,
            "tokens": [],
            "path": "/tmp/doc3.txt",
            "title": "Empty Document",
        },
    ]


# --- 1. Основні тести методів Index ---

def test_index_dunder_methods(sample_corpus: list[CorpusDocument]) -> None:
    index = build_index(sample_corpus)

    assert len(index) == 3
    assert "cat" in index
    assert "fish" not in index

    postings = index["cat"]
    assert isinstance(postings, list)
    assert postings[0].doc_id == 1
    assert postings[0].tf == 2

    with pytest.raises(KeyError):
        _ = index["non_existent_term"]

    terms = set(iter(index))
    assert terms == {"cat", "dog", "bird"}

    repr_str = repr(index)
    assert "Index(" in repr_str
    assert "terms=3" in repr_str


def test_index_properties_and_methods(sample_corpus: list[CorpusDocument]) -> None:
    index = build_index(sample_corpus)

    assert index.num_docs == 3
    assert index.avg_doc_length == 5 / 3

    assert index.doc_length(1) == 3
    assert index.doc_length(2) == 2
    assert index.doc_length(99) == 0

    assert index.df("dog") == 2
    assert index.df("fish") == 0


# --- 2. Save/Load Round Trip test ---

def test_open_index_context_manager(tmp_path: Path, sample_corpus: list[CorpusDocument]) -> None:
    index_file = tmp_path / "test_index.pkl"

    with open_index(index_file) as idx:
        temp_idx = build_index(sample_corpus)
        idx._postings.update(temp_idx._postings)
        idx._doc_lengths.update(temp_idx._doc_lengths)
        idx._doc_meta.update(temp_idx._doc_meta)

    assert index_file.exists()
    with open_index(index_file) as loaded_idx:
        assert loaded_idx.num_docs == 3
        assert "cat" in loaded_idx
        assert loaded_idx.df("dog") == 2


# --- 3. Hypothesis Property-based Test (Властивість 2 з 3) ---

@given(
    # Використовуємо st.sets, щоб уникнути дублікатів з однаковим doc_id у генераторі
    st.lists(
        st.fixed_dictionaries({
            "doc_id": st.integers(min_value=1, max_value=1000),
            "tokens": st.lists(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=5),
                min_size=1,
                max_size=10,
            ),
            "path": st.just("/tmp/doc.txt"),
            "title": st.just("Title"),
        }),
        min_size=1,
        max_size=10,
        unique_by=lambda doc: doc["doc_id"],  # Переконуємося, що у кожного документа унікальний doc_id
    )
)
def test_index_properties_hypothesis(corpus: list[CorpusDocument]) -> None:
    """Властивість 2: Кількість документів в індексі та збереження невід'ємності avg_doc_length."""
    index = build_index(corpus)

    # Кількість унікальних doc_id відповідає num_docs
    unique_doc_ids = {doc["doc_id"] for doc in corpus}
    assert index.num_docs == len(unique_doc_ids)
    assert index.avg_doc_length >= 0.0

    for term in index:
        assert 0 < index.df(term) <= index.num_docs