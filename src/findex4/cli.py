import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from findex4.index import Index, build_index
from findex4.pipeline import iter_documents
from findex4.search import search
from findex4.store import load, save
from findex4.tokenizer import tokenize

# Потік для виводу консольних таблиць та статусів
err_console = Console(stderr=True)
# Потік для чистих JSON даних (stdout)
stdout_console = Console()

app = typer.Typer(
    name="findex",
    help="CLI-інструмент для побудови індексу, пошуку та аналітики (Findex).",
    no_args_is_help=True,
)


def _setup_logging(verbose: int) -> None:
    if verbose == 1:
        log_level = logging.INFO
    elif verbose >= 2:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING

    # Налаштування красивого логера через RichHandler з направленням у stderr
    handler = RichHandler(
        console=err_console,
        show_path=False,
        omit_repeated_times=False,
        rich_tracebacks=True,
    )

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )


@app.callback()
def main(
        verbose: int = typer.Option(
            0,
            "--verbose",
            "-v",
            count=True,
            help="-v для INFO логів, -vv для DEBUG логів.",
        ),
) -> None:
    _setup_logging(verbose)


@app.command()
def index(
        corpus: Annotated[Path, typer.Argument(help="Шлях до корпусу файлів/директорії")],
        out: Annotated[
            Path, typer.Option("--out", "-o", help="Шлях для збереження індексу")
        ] = Path("index.bin"),
        positions: Annotated[
            bool, typer.Option("--positions", "-p", help="Зберігати позиції токенів для фразового пошуку")
        ] = True,
        limit: Annotated[
            Optional[int], typer.Option("--limit", "-l", help="Обмежити кількість документів")
        ] = None,
) -> None:
    """Побудувати та зберегти індекс з корпусу документів."""
    if not corpus.exists():
        logging.error("Шлях '%s' не існує.", corpus)
        raise typer.Exit(code=1)

    logging.info("Початок зчитування корпусу з %s", corpus)

    documents: list[dict[str, object]] = []

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=err_console,
    ) as progress:
        task = progress.add_task("Зчитування документів...", total=None)

        raw_docs = list(iter_documents(corpus))
        if limit is not None:
            raw_docs = raw_docs[:limit]

        progress.update(task, total=len(raw_docs), description="Індексація токенів...")

        for numeric_id, doc in enumerate(raw_docs):
            tokens = list(tokenize(doc.text))
            documents.append(
                {
                    "doc_id": numeric_id,
                    "tokens": tokens,
                    "path": str(doc.path),
                    "title": doc.doc_id,
                    "text": doc.text,
                }
            )
            progress.advance(task)

    logging.info("Побудова індексу для %d документів...", len(documents))
    idx = build_index(documents, with_positions=positions)

    logging.info("Збереження індексу у %s", out)
    save(idx, out)

    logging.info("Успіх! Індекс побудовано (%d doc) та збережено у %s.", len(documents), out)


@app.command()
def search_cmd(
        index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="Шлях до файлу індексу")],
        query: Annotated[str, typer.Argument(help="Пошуковий запит")],
        k: Annotated[int, typer.Option("--k", "-k", help="Кількість топових результатів")] = 10,
        scorer: Annotated[str, typer.Option("--scorer", "-s", help="Алгоритм ранжування (bm25/tfidf)")] = "bm25",
        json_output: Annotated[
            bool, typer.Option("--json", help="Форматувати результат у JSON Lines (stdout)")] = False,
) -> None:
    """Виконати пошук за індексом."""
    if not index_path.exists():
        logging.error("Файл індексу '%s' не знайдено.", index_path)
        raise typer.Exit(code=1)

    logging.info("Завантаження індексу з %s", index_path)
    idx: Index = load(index_path)

    logging.info("Виконання запиту '%s' (scorer=%s, k=%d)", query, scorer, k)
    results = search(query, idx, scorer_type=scorer, top_k=k)

    if json_output:
        for res in results:
            data = {
                "doc_id": res.doc_id,
                "score": res.score,
                "title": res.title,
                "path": res.path,
                "snippet": res.snippet,
            }
            stdout_console.print(json.dumps(data, ensure_ascii=False))
        return

    if not results:
        logging.warning("За вашим запитом нічого не знайдено.")
        return

    table = Table(title=f"Результати пошуку: '{query}'", show_header=True, header_style="bold magenta")
    table.add_column("Score", justify="right", style="cyan", no_wrap=True)
    table.add_column("Doc ID", justify="right", style="green")
    table.add_column("Title / Path", style="bold")
    table.add_column("Snippet", style="italic")

    for res in results:
        snippet_rich = res.snippet.replace("<b>", "[bold yellow]").replace("</b>", "[/bold yellow]")
        table.add_row(
            f"{res.score:.4f}",
            str(res.doc_id),
            f"{res.title}\n[dim]{res.path}[/dim]",
            snippet_rich,
        )

    err_console.print(table)


@app.command()
def stats(
        index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="Шлях до файлу індексу")],
) -> None:
    """Вивести статистичні показники індексу."""
    if not index_path.exists():
        logging.error("Файл індексу '%s' не знайдено.", index_path)
        raise typer.Exit(code=1)

    logging.info("Завантаження індексу з %s", index_path)
    idx: Index = load(index_path)

    # 1. Кількість документів
    num_docs = int(getattr(idx, "num_docs", getattr(idx, "N", 0)))

    # 2. Розмір словника (унікальні терміни)
    dict_size = 0
    if hasattr(idx, "dictionary"):
        dict_size = len(idx.dictionary)  # type: ignore[reportAttributeAccessIssue]
    elif hasattr(idx, "_dictionary"):
        dict_size = len(idx._dictionary)
    elif hasattr(idx, "_postings"):
        dict_size = len(idx._postings)
    elif hasattr(idx, "postings"):
        dict_size = len(idx.postings)
    else:
        try:
            dict_size = len(idx)
        except TypeError:
            dict_size = 0

    # 3. Середня довжина документа
    avg_doc_len = 0.0
    for attr in ("avg_doc_len", "avg_dl", "_avg_doc_len", "_avg_dl"):
        if hasattr(idx, attr):
            val = getattr(idx, attr)
            avg_doc_len = float(val() if callable(val) else val)
            break

    # 4. Загальна кількість токенів
    total_tokens = 0
    if hasattr(idx, "total_tokens"):
        val = getattr(idx, "total_tokens")
        total_tokens = int(val() if callable(val) else val)
    elif hasattr(idx, "doc_lengths"):
        total_tokens = sum(idx.doc_lengths.values())
    elif hasattr(idx, "_doc_lengths"):
        total_tokens = sum(idx._doc_lengths.values())

    # Розрахунок узгоджених даних, якщо один із показників дорівнює 0
    if avg_doc_len == 0.0 and num_docs > 0 and total_tokens > 0:
        avg_doc_len = total_tokens / num_docs
    elif total_tokens == 0 and num_docs > 0 and avg_doc_len > 0:
        total_tokens = int(num_docs * avg_doc_len)

    table = Table(title="Статистика індексу (Findex)", show_header=True, header_style="bold cyan")
    table.add_column("Метрика", style="bold white")
    table.add_column("Значення", style="green", justify="right")

    table.add_row("Загальна кількість документів (N)", f"{num_docs:,}")
    table.add_row("Розмір словника (унікальні терміни)", f"{dict_size:,}")
    table.add_row("Загальна кількість токенів", f"{total_tokens:,}")
    table.add_row("Середня довжина документа (avgDL)", f"{avg_doc_len:.2f}")

    err_console.print(table)


if __name__ == "__main__":
    app()