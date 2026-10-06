import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

# Точні імпорти згідно з твоїми тестами
from findex4.index import Index, build_index
from findex4.search import search
from findex4.store import open_index

# Консоль для системних повідомлень, таблиць та логів (stderr)
err_console = Console(stderr=True)
# Консоль для чистих даних JSON (stdout)
stdout_console = Console()

app = typer.Typer(
    name="findex",
    help="CLI-інструмент для побудови індексу та пошуку",
    no_args_is_help=True,
)


def setup_logging(verbose: int) -> None:
    """Налаштування рівня деталізації логування за прапорцем -v / -vv."""
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

@app.callback()
def main(
    verbose: int = typer.Option(
        0,
        "-v",
        "--verbose",
        help="Рівень деталізації логів (-v для INFO, -vv для DEBUG)",
        count=True,
    ),
) -> None:
    """Головний Callback для ініціалізації логування."""
    setup_logging(verbose)

@app.command()
def index(
    data_dir: Annotated[Path, typer.Argument(metavar="DATA_DIR", help="Папка з текстовими документами")],
    out: Annotated[
        Optional[Path],
        typer.Option("--out", "-o", help="Шлях для збереження файлу індексу"),
    ] = None,
) -> None:
    """Побудувати індекс за текстовими документами у вказаній директорії."""
    if not data_dir.exists() or not data_dir.is_dir():
        err_console.print(f"[bold red]Помилка:[/bold red] Директорія '{data_dir}' не існує.")
        raise typer.Exit(code=1)

    out_path = out if out is not None else Path("index.bin")

    logging.info("Початок індексації директорії: %s", data_dir)
    idx = build_index(data_dir)

    logging.info("Збереження індексу у файл: %s", out_path)
    with open_index(out_path) as store_idx:
        store_idx._postings.update(idx._postings)
        store_idx._doc_lengths.update(idx._doc_lengths)
        if hasattr(idx, "_doc_meta"):
            store_idx._doc_meta.update(idx._doc_meta)

    err_console.print(f"[bold green][УСПІХ] Індекс побудовано[/bold green] та збережено у '{out_path}'.")


@app.command()
def search_cmd(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="Шлях до файлу індексу")],
    query: Annotated[str, typer.Argument(help="Пошуковий запит")],
    k: Annotated[int, typer.Option("--k", "-k", help="Кількість топових результатів")] = 10,
    scorer: Annotated[str, typer.Option("--scorer", "-s", help="Алгоритм ранжування (bm25/tfidf)")] = "bm25",
    json_output: Annotated[bool, typer.Option("--json", help="Форматувати результат у JSON Lines (stdout)")] = False,
) -> None:
    """Виконати пошук за індексом."""
    if not index_path.exists():
        err_console.print(f"[bold red]Помилка:[/bold red] Файл індексу '{index_path}' не знайдено.")
        raise typer.Exit(code=1)

    logging.info("Завантаження індексу з %s", index_path)
    with open_index(index_path) as idx:
        logging.info("Виконання запиту '%s' (scorer=%s, k=%d)", query, scorer, k)
        results = search(query, idx, scorer_type=scorer, top_k=k)

        if json_output:
            for res in results:
                data = {
                    "doc_id": res.doc_id,
                    "score": res.score,
                    "title": getattr(res, "title", f"Doc {res.doc_id}"),
                    "path": getattr(res, "path", ""),
                    "snippet": getattr(res, "snippet", ""),
                }
                stdout_console.print(json.dumps(data, ensure_ascii=False))
            return

        if not results:
            err_console.print("[yellow]За вашим запитом нічого не знайдено.[/yellow]")
            return

        err_console.print(f"[bold green]Знайдено документів: {len(results)}[/bold green]")

        table = Table(title=f"Результати пошуку: '{query}'", show_header=True, header_style="bold magenta")
        table.add_column("Score", justify="right", style="cyan", no_wrap=True)
        table.add_column("Doc ID", justify="right", style="green")
        table.add_column("Title / Path", style="bold")
        table.add_column("Snippet", style="italic")

        for res in results:
            snippet = getattr(res, "snippet", "").replace("<b>", "[bold yellow]").replace("</b>", "[/bold yellow]")
            title = getattr(res, "title", f"Doc {res.doc_id}")
            path = getattr(res, "path", "")
            table.add_row(
                f"{res.score:.4f}",
                str(res.doc_id),
                f"{title}\n[dim]{path}[/dim]",
                snippet,
            )

        err_console.print(table)


@app.command()
def stats(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="Шлях до файлу індексу")],
) -> None:
    """Вивести статистичні показники індексу."""
    if not index_path.exists():
        err_console.print(f"[bold red]Помилка:[/bold red] Файл індексу '{index_path}' не знайдено.")
        raise typer.Exit(code=1)

    logging.info("Завантаження індексу для збору статистики з %s", index_path)
    with open_index(index_path) as idx:
        num_docs = idx.num_docs
        dict_size = len(idx)
        avg_doc_len = idx.avg_doc_length
        total_tokens = sum(idx._doc_lengths.values()) if hasattr(idx, "_doc_lengths") else int(num_docs * avg_doc_len)

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