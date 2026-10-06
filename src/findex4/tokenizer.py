import re
import unicodedata
from collections.abc import Iterator


WORD_PATTERN = re.compile(
    r"\b[\w]+(?:['’\-][\w]+)*\b",
    re.UNICODE,
)


def tokenize(text: str) -> Iterator[str]:
    if not text:
        return

    normalized_text = unicodedata.normalize("NFC", text)
    casefolded_text = normalized_text.casefold()


    for match in WORD_PATTERN.finditer(casefolded_text):
        token = match.group(0)

        if not token.isdigit() and not token.startswith("_"):
            yield token