import pytest

from findex4.index import CorpusDocument, Posting, build_index
from findex4.scorer import BM25

# --- 1. Фікстура спеціального корпусу для перевірки властивостей BM25 ---

@pytest.fixture
def ranking_corpus() -> list[CorpusDocument]:
    return [
        # Doc 1: Короткий документ з рідкісним терміном "qubit"
        {
            "doc_id": 1,
            "title": "Short Quantum Paper",
            "path": "doc1.txt",
            "tokens": ["qubit", "algorithm"],
        },
        # Doc 2: Довгий документ з частим терміном "paper" і багатьма повторами "model"
        {
            "doc_id": 2,
            "title": "Long Machine Learning Paper",
            "path": "doc2.txt",
            "tokens": [
                "paper", "model", "model", "model", "model", "model",
                "model", "model", "model", "model", "model", "data",
                "algorithm", "system", "testing", "evaluation", "results"
            ],
        },
        # Doc 3: Ще один документ, щоб зробити "paper" частим словосполученням
        {
            "doc_id": 3,
            "title": "Another Paper",
            "path": "doc3.txt",
            "tokens": ["paper", "data", "system"],
        },
    ]


# --- 2. Sanity Check 1: IDF Effect (Рідкісний термін вище частого) ---

def test_bm25_sanity_check_idf_effect(ranking_corpus: list[CorpusDocument]) -> None:
    index = build_index(ranking_corpus, with_positions=True)
    bm25 = BM25(k1=1.5, b=0.75)

    rare_term = "qubit"
    frequent_term = "paper"

    rare_posting = index[rare_term][0]
    freq_posting = index[frequent_term][0]

    score_rare = bm25.score(rare_term, rare_posting, index)
    score_freq = bm25.score(frequent_term, freq_posting, index)

    # Рідкісний термін має нижчу DF, отже вищу IDF та більший скор
    assert index.df(rare_term) < index.df(frequent_term)
    assert score_rare > score_freq


# --- 3. Sanity Check 2: TF Saturation (Насичення частоти TF) ---

def test_bm25_sanity_check_tf_saturation(ranking_corpus: list[CorpusDocument]) -> None:
    index = build_index(ranking_corpus, with_positions=True)
    bm25 = BM25(k1=1.5, b=0.75)

    target_term = "model"
    posting_high_tf = index[target_term][0]  # Doc 2 має tf = 10

    p_tf1 = Posting(doc_id=posting_high_tf.doc_id, tf=1, positions=(0,))
    p_tf2 = Posting(doc_id=posting_high_tf.doc_id, tf=2, positions=(0, 1))

    score_tf1 = bm25.score(target_term, p_tf1, index)
    score_tf2 = bm25.score(target_term, p_tf2, index)
    score_tf_max = bm25.score(target_term, posting_high_tf, index)

    gain_1_to_2 = score_tf2 - score_tf1
    avg_gain_later = (score_tf_max - score_tf2) / (posting_high_tf.tf - 2)

    # Приріст від TF=1 до TF=2 має бути СУТТЄВО більшим, ніж середній приріст на наступних входженнях
    assert score_tf2 > score_tf1
    assert score_tf_max > score_tf2
    assert gain_1_to_2 > avg_gain_later


# --- 4. Sanity Check 3: Length Penalty (Штраф за довжину) ---

def test_bm25_sanity_check_length_penalty(ranking_corpus: list[CorpusDocument]) -> None:
    index = build_index(ranking_corpus, with_positions=True)
    bm25 = BM25(k1=1.5, b=0.75)

    term = "algorithm"
    postings = index[term]  # Присутній в Doc 1 (короткий) та Doc 2 (довгий) з TF=1

    p_short = next(p for p in postings if p.doc_id == 1)
    p_long = next(p for p in postings if p.doc_id == 2)

    score_short = bm25.score(term, p_short, index)
    score_long = bm25.score(term, p_long, index)

    # При однакому TF=1 коротший документ має отримати вищий скор завдяки b=0.75
    assert index.doc_length(p_short.doc_id) < index.doc_length(p_long.doc_id)
    assert score_short > score_long