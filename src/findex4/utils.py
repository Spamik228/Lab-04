import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("findex3")

F = TypeVar("F", bound=Callable[..., Any])


def timed(func: F) -> F:


    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"[{func.__name__}] виконалася за {elapsed_ms:.2f} ms")
        return result

    return wrapper  # type: ignore[return-value]