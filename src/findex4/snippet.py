import re
from findex4.tokenizer import tokenize


STOP_OPERATORS = {"and", "or", "not"}


def make_snippet(
    text: str,
    query_terms: list[str],
    window: int = 80,
    highlight_fmt: str = "**{}**",
) -> str:
    if not text or not query_terms:
        return text[: window * 2] + "..." if len(text) > window * 2 else text


    clean_terms = set()
    for q in query_terms:
        for token in tokenize(q):
            token_lower = token.lower()
            if token_lower not in STOP_OPERATORS:
                clean_terms.add(token_lower)


    if not clean_terms:
        return text[: window * 2] + "..." if len(text) > window * 2 else text

    text_fold = text.casefold()


    first_match_idx = -1
    for term in clean_terms:
        idx = text_fold.find(term)
        if idx != -1:
            if first_match_idx == -1 or idx < first_match_idx:
                first_match_idx = idx

    if first_match_idx == -1:
        start = 0
        end = min(len(text), window * 2)
        snippet = text[start:end]
        prefix = ""
        suffix = "..." if len(text) > end else ""
    else:
        start = max(0, first_match_idx - window)
        end = min(len(text), first_match_idx + window)
        snippet = text[start:end]
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""


    for term in sorted(clean_terms, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        snippet = pattern.sub(lambda m: highlight_fmt.format(m.group(0)), snippet)

    return f"{prefix}{snippet}{suffix}"