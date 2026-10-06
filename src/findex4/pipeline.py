import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path | str) -> Iterator[Document]:
    root_path = Path(root)

    if not root_path.exists():
        logger.warning("Шлях %s не існує.", root_path)
        return

    for file_path in root_path.rglob("*.txt"):
        if not file_path.is_file():
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            doc_id = str(file_path.relative_to(root_path))

            yield Document(
                doc_id=doc_id,
                path=file_path,
                text=text,
            )

        except Exception as e:
            logger.error("Помилка при читанні файлу %s: %s", file_path, e)
            continue