"""
Progress reporting and cancellation support.

This module provides:
- ProgressInfo: Progress information structure
- ProgressReporter: Manages progress callbacks
- CancellationToken: Thread-safe cancellation mechanism
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple
import time
import threading


@dataclass
class ProgressInfo:
    """Progress information for frontend display.

    Attributes:
        stage: Current optimization stage ('phase' or 'fab')
        current_iter: Current iteration number
        total_iters: Total number of iterations
        current_loss: Current loss value
        elapsed_seconds: Time elapsed since stage start
        estimated_remaining_seconds: Estimated time remaining
        best_loss: Best loss value achieved so far
        loss_history: Recent loss history for plotting
    """
    stage: str
    current_iter: int
    total_iters: int
    current_loss: float
    elapsed_seconds: float
    estimated_remaining_seconds: float
    best_loss: Optional[float] = None
    loss_history: Optional[List[Tuple[int, float]]] = None

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total_iters <= 0:
            return 0.0
        return 100.0 * self.current_iter / self.total_iters

    @property
    def iterations_per_second(self) -> float:
        """Get current iteration rate."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.current_iter / self.elapsed_seconds

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'stage': self.stage,
            'current_iter': self.current_iter,
            'total_iters': self.total_iters,
            'current_loss': self.current_loss,
            'progress_percent': self.progress_percent,
            'elapsed_seconds': self.elapsed_seconds,
            'estimated_remaining_seconds': self.estimated_remaining_seconds,
            'iterations_per_second': self.iterations_per_second,
        }
        if self.best_loss is not None:
            result['best_loss'] = self.best_loss
        if self.loss_history:
            result['loss_history'] = self.loss_history
        return result


class CancellationToken:
    """Thread-safe cancellation token.

    Used to signal that an optimization should be cancelled.

    Example:
        token = CancellationToken()

        # In main thread
        token.cancel()

        # In optimization loop
        if token.is_cancelled:
            break
    """

    def __init__(self):
        """Initialize cancellation token."""
        self._cancelled = threading.Event()
        self._cancel_reason: Optional[str] = None

    def cancel(self, reason: Optional[str] = None) -> None:
        """Request cancellation.

        Args:
            reason: Optional reason for cancellation
        """
        self._cancel_reason = reason
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled.is_set()

    @property
    def reason(self) -> Optional[str]:
        """Get cancellation reason."""
        return self._cancel_reason

    def reset(self) -> None:
        """Reset the cancellation token for reuse."""
        self._cancelled.clear()
        self._cancel_reason = None

    def wait(self, timeout: float = None) -> bool:
        """Wait for cancellation.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if cancelled, False if timeout
        """
        return self._cancelled.wait(timeout)


class ProgressReporter:
    """Manages progress reporting during optimization.

    Provides:
    - Progress callbacks at configurable intervals
    - Estimated time remaining calculation
    - Loss history tracking
    - Cancellation checking

    Example:
        def on_progress(info: ProgressInfo):
            print(f"Progress: {info.progress_percent:.1f}%")

        reporter = ProgressReporter(
            callback=on_progress,
            report_interval=100,
            cancellation_token=token
        )

        reporter.start_stage('phase', total_iters=10000)
        for i in range(10000):
            # ... optimization step ...
            if not reporter.report(i, loss):
                break  # Cancelled
    """

    def __init__(
        self,
        callback: Optional[Callable[[ProgressInfo], None]] = None,
        report_interval: int = 100,
        cancellation_token: Optional[CancellationToken] = None,
        history_length: int = 100
    ):
        """Initialize progress reporter.

        Args:
            callback: Function to call with progress updates
            report_interval: Report every N iterations
            cancellation_token: Token for cancellation checking
            history_length: Number of loss values to keep in history
        """
        self.callback = callback
        self.report_interval = report_interval
        self.cancellation_token = cancellation_token
        self.history_length = history_length

        # State
        self.stage: str = ""
        self.total_iters: int = 0
        self.start_time: float = 0.0
        self.loss_history: List[Tuple[int, float]] = []
        self.best_loss: Optional[float] = None

    def start_stage(self, stage: str, total_iters: int) -> None:
        """Called when a new optimization stage begins.

        Args:
            stage: Stage name ('phase' or 'fab')
            total_iters: Total number of iterations for this stage
        """
        self.stage = stage
        self.total_iters = total_iters
        self.start_time = time.time()
        self.loss_history = []
        self.best_loss = None

    def report(self, iter_num: int, loss: float) -> bool:
        """Report progress for an iteration.

        Args:
            iter_num: Current iteration number (0-indexed)
            loss: Current loss value

        Returns:
            True to continue, False if cancelled
        """
        # Check for cancellation
        if self.cancellation_token and self.cancellation_token.is_cancelled:
            return False

        # Update best loss
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss

        # Add to history (subsampled)
        if len(self.loss_history) == 0 or iter_num % max(1, self.total_iters // self.history_length) == 0:
            self.loss_history.append((iter_num, loss))
            # Keep only recent history
            if len(self.loss_history) > self.history_length:
                self.loss_history = self.loss_history[-self.history_length:]

        # Report at intervals or at the end
        should_report = (
            iter_num % self.report_interval == 0 or
            iter_num == self.total_iters - 1 or
            iter_num == 0
        )

        if should_report and self.callback:
            elapsed = time.time() - self.start_time

            # Estimate remaining time
            if iter_num > 0:
                rate = elapsed / iter_num
                remaining = rate * (self.total_iters - iter_num)
            else:
                remaining = 0

            info = ProgressInfo(
                stage=self.stage,
                current_iter=iter_num,
                total_iters=self.total_iters,
                current_loss=loss,
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=remaining,
                best_loss=self.best_loss,
                loss_history=list(self.loss_history) if self.loss_history else None,
            )

            self.callback(info)

        return True

    def finish_stage(self, final_loss: float) -> ProgressInfo:
        """Mark stage as complete and return final progress info.

        Args:
            final_loss: Final loss value

        Returns:
            Final ProgressInfo for the stage
        """
        elapsed = time.time() - self.start_time

        return ProgressInfo(
            stage=self.stage,
            current_iter=self.total_iters,
            total_iters=self.total_iters,
            current_loss=final_loss,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=0,
            best_loss=self.best_loss or final_loss,
            loss_history=list(self.loss_history) if self.loss_history else None,
        )

    def get_elapsed(self) -> float:
        """Get elapsed time for current stage in seconds."""
        return time.time() - self.start_time


def create_simple_callback(
    print_interval: int = 1000
) -> Callable[[ProgressInfo], None]:
    """Create a simple print callback for console output.

    Args:
        print_interval: Print every N reports

    Returns:
        Callback function
    """
    counter = [0]  # Use list to allow modification in closure

    def callback(info: ProgressInfo) -> None:
        counter[0] += 1
        if counter[0] % print_interval == 0 or info.current_iter == info.total_iters - 1:
            print(
                f"[{info.stage}] {info.progress_percent:5.1f}% | "
                f"Iter {info.current_iter}/{info.total_iters} | "
                f"Loss: {info.current_loss:.6f} | "
                f"Best: {info.best_loss:.6f} | "
                f"ETA: {info.estimated_remaining_seconds:.0f}s"
            )

    return callback
