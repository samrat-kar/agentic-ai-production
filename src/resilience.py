# Copyright (c) 2026 Samrat Kar
# Licensed under CC BY-NC-SA 4.0 — see LICENSE for details.

"""Retry logic, timeouts, and iteration caps for resilient agent workflows.

Provides:
- ``with_retry`` — decorator for exponential-backoff retries on transient errors.
- ``run_with_timeout`` — runs any callable with a hard wall-clock timeout.
- ``IterationLimiter`` — counter that raises once a loop cap is exceeded.

All retry attempts, timeouts, and limit violations are logged for traceability.
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
import time
from typing import Any, Callable, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES: int = 3
DEFAULT_INITIAL_WAIT: float = 1.0   # seconds before first retry
DEFAULT_MAX_WAIT: float = 30.0      # ceiling for exponential growth
DEFAULT_BACKOFF_FACTOR: float = 2.0


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------


def with_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_wait: float = DEFAULT_INITIAL_WAIT,
    max_wait: float = DEFAULT_MAX_WAIT,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator: retry a function with exponential backoff on transient failures.

    Example::

        @with_retry(max_retries=3, exceptions=(openai.RateLimitError,))
        def call_api(...):
            ...

    Args:
        max_retries: Maximum number of retry attempts (not counting the first try).
        initial_wait: Seconds to wait before the first retry.
        max_wait: Upper bound for the inter-retry delay.
        backoff_factor: Multiplier applied to the wait time after each retry.
        exceptions: Tuple of exception types that trigger a retry.

    Returns:
        Decorated function that automatically retries on the specified exceptions.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            wait = initial_wait
            for attempt in range(1, max_retries + 2):  # 1st try + max_retries retries
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # type: ignore[misc]
                    if attempt > max_retries:
                        logger.error(
                            "%s failed after %d attempt(s): %s",
                            func.__name__,
                            attempt,
                            exc,
                        )
                        raise
                    sleep_for = min(wait, max_wait)
                    logger.warning(
                        "%s attempt %d/%d failed (%s: %s). Retrying in %.1fs.",
                        func.__name__,
                        attempt,
                        max_retries + 1,
                        type(exc).__name__,
                        exc,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    wait *= backoff_factor

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------


def run_with_timeout(
    func: Callable[..., Any],
    timeout_seconds: float,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run *func* with a hard wall-clock timeout.

    Uses a ``ThreadPoolExecutor`` so it works on both Unix and Windows without
    requiring ``signal`` (which only works on the main thread).

    Args:
        func: Callable to execute.
        timeout_seconds: Maximum allowed wall-clock time in seconds.
        *args: Positional arguments forwarded to *func*.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        Whatever *func* returns.

    Raises:
        TimeoutError: If *func* does not finish within *timeout_seconds*.
    """
    func_name = getattr(func, "__name__", repr(func))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            logger.error(
                "Timeout: %s did not complete within %.0f seconds.",
                func_name,
                timeout_seconds,
            )
            raise TimeoutError(
                f"Operation '{func_name}' timed out after {timeout_seconds:.0f} seconds. "
                "The research pipeline may be under heavy load — please try again."
            )


# ---------------------------------------------------------------------------
# Iteration limiter
# ---------------------------------------------------------------------------


class IterationLimiter:
    """Enforce a hard cap on the number of iterations in a loop or agent cycle.

    Prevents silent infinite loops in multi-step agent workflows.

    Example::

        limiter = IterationLimiter(max_iterations=20, label="research_loop")
        while condition:
            limiter.tick()   # raises RuntimeError if cap exceeded
            ...

    Args:
        max_iterations: Maximum allowed calls to :meth:`tick`.
        label: Human-readable name for logging and error messages.
    """

    def __init__(self, max_iterations: int = 50, label: str = "loop") -> None:
        self.max_iterations = max_iterations
        self.label = label
        self._count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Increment the counter; raise ``RuntimeError`` if the cap is exceeded."""
        self._count += 1
        if self._count > self.max_iterations:
            logger.error(
                "IterationLimiter '%s': exceeded cap of %d iterations.",
                self.label,
                self.max_iterations,
            )
            raise RuntimeError(
                f"Iteration limit ({self.max_iterations}) exceeded for '{self.label}'. "
                "Aborting to prevent an infinite loop."
            )
        logger.debug(
            "IterationLimiter '%s': iteration %d/%d.",
            self.label,
            self._count,
            self.max_iterations,
        )

    def reset(self) -> None:
        """Reset the counter back to zero."""
        self._count = 0

    @property
    def count(self) -> int:
        """Current iteration count."""
        return self._count

    @property
    def remaining(self) -> int:
        """Remaining iterations before the cap is hit."""
        return max(0, self.max_iterations - self._count)
