"""Unit tests for src/resilience.py — retry, timeout, and iteration limits."""

import time

import pytest

from src.resilience import IterationLimiter, run_with_timeout, with_retry


# ---------------------------------------------------------------------------
# with_retry
# ---------------------------------------------------------------------------


class TestWithRetry:
    def test_succeeds_on_first_try(self):
        calls = []

        @with_retry(max_retries=3)
        def always_ok():
            calls.append(1)
            return "ok"

        assert always_ok() == "ok"
        assert len(calls) == 1

    def test_retries_and_succeeds(self):
        attempts = []

        @with_retry(max_retries=3, initial_wait=0, backoff_factor=1)
        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("transient")
            return "done"

        result = flaky()
        assert result == "done"
        assert len(attempts) == 3

    def test_raises_after_max_retries(self):
        attempts = []

        @with_retry(max_retries=2, initial_wait=0, backoff_factor=1)
        def always_fails():
            attempts.append(1)
            raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError, match="permanent failure"):
            always_fails()

        # 1 initial attempt + 2 retries = 3 total
        assert len(attempts) == 3

    def test_only_retries_specified_exceptions(self):
        """Should NOT retry if the exception type is not in the list."""
        attempts = []

        @with_retry(max_retries=3, initial_wait=0, exceptions=(ValueError,))
        def wrong_exception():
            attempts.append(1)
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            wrong_exception()

        assert len(attempts) == 1  # no retries

    def test_preserves_return_value(self):
        @with_retry(max_retries=1)
        def returns_dict():
            return {"key": "value"}

        assert returns_dict() == {"key": "value"}

    def test_preserves_function_name(self):
        @with_retry(max_retries=1)
        def my_function():
            return None

        assert my_function.__name__ == "my_function"

    def test_backoff_respects_max_wait(self):
        """Ensure sleep duration is capped at max_wait (verify via timing)."""
        # We patch time.sleep to avoid actual waiting
        sleep_calls = []
        import src.resilience as res_module
        original_sleep = res_module.time.sleep

        def fake_sleep(secs):
            sleep_calls.append(secs)

        res_module.time.sleep = fake_sleep
        try:
            @with_retry(
                max_retries=3,
                initial_wait=1.0,
                max_wait=2.0,
                backoff_factor=10.0,
            )
            def always_fails():
                raise ValueError("fail")

            with pytest.raises(ValueError):
                always_fails()

            for s in sleep_calls:
                assert s <= 2.0
        finally:
            res_module.time.sleep = original_sleep


# ---------------------------------------------------------------------------
# run_with_timeout
# ---------------------------------------------------------------------------


class TestRunWithTimeout:
    def test_fast_function_completes(self):
        def quick():
            return 42

        assert run_with_timeout(quick, timeout_seconds=5) == 42

    def test_passes_args_and_kwargs(self):
        def add(a, b, multiplier=1):
            return (a + b) * multiplier

        result = run_with_timeout(add, 5, 3, 4, multiplier=2)
        assert result == 14

    def test_raises_timeout_error(self):
        def slow():
            time.sleep(10)
            return "done"

        with pytest.raises(TimeoutError, match="timed out"):
            run_with_timeout(slow, timeout_seconds=0.1)

    def test_propagates_function_exception(self):
        def boom():
            raise ValueError("inner error")

        with pytest.raises(ValueError, match="inner error"):
            run_with_timeout(boom, timeout_seconds=5)


# ---------------------------------------------------------------------------
# IterationLimiter
# ---------------------------------------------------------------------------


class TestIterationLimiter:
    def test_allows_up_to_limit(self):
        limiter = IterationLimiter(max_iterations=5, label="test")
        for _ in range(5):
            limiter.tick()  # should not raise

    def test_raises_on_exceeded(self):
        limiter = IterationLimiter(max_iterations=3, label="test")
        for _ in range(3):
            limiter.tick()
        with pytest.raises(RuntimeError, match="Iteration limit"):
            limiter.tick()

    def test_count_increments(self):
        limiter = IterationLimiter(max_iterations=10, label="test")
        limiter.tick()
        limiter.tick()
        assert limiter.count == 2

    def test_remaining_decrements(self):
        limiter = IterationLimiter(max_iterations=10, label="test")
        limiter.tick()
        limiter.tick()
        assert limiter.remaining == 8

    def test_remaining_never_below_zero(self):
        limiter = IterationLimiter(max_iterations=2, label="test")
        limiter.tick()
        limiter.tick()
        # remaining should be 0 now
        assert limiter.remaining == 0

    def test_reset_clears_count(self):
        limiter = IterationLimiter(max_iterations=3, label="test")
        limiter.tick()
        limiter.tick()
        limiter.reset()
        assert limiter.count == 0
        # Should be able to tick again after reset
        limiter.tick()
        assert limiter.count == 1

    def test_default_max_iterations(self):
        limiter = IterationLimiter()
        assert limiter.max_iterations == 50

    def test_error_message_contains_label(self):
        limiter = IterationLimiter(max_iterations=1, label="my_workflow")
        limiter.tick()
        with pytest.raises(RuntimeError, match="my_workflow"):
            limiter.tick()
