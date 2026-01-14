"""
Task Manager for handling optimization tasks.

Provides:
- Task creation and tracking
- Background execution with ThreadPoolExecutor
- WebSocket progress broadcasting
- Cancellation support
"""

import asyncio
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading

from doe_optimizer import (
    run_optimization,
    CancellationToken,
    ProgressInfo,
)
from doe_optimizer.evaluation.reevaluate import (
    reevaluate_at_resolution,
    extract_target_indices,
)

from ..config import config


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class OptimizationTask:
    """Represents an optimization task."""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    cancellation_token: CancellationToken = field(default_factory=CancellationToken)
    websocket_clients: List[Any] = field(default_factory=list)
    request_data: Optional[Dict[str, Any]] = None  # Store original request for re-evaluation
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_progress(self, progress_info: ProgressInfo) -> None:
        """Thread-safe progress update."""
        with self._lock:
            self.progress = progress_info.to_dict()


class TaskManager:
    """Manages optimization tasks with concurrent execution support."""

    def __init__(self, max_concurrent: int = None):
        """Initialize task manager.

        Args:
            max_concurrent: Maximum concurrent optimization tasks
        """
        self.max_concurrent = max_concurrent or config.max_concurrent_tasks
        self.tasks: Dict[str, OptimizationTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
        self._lock = threading.Lock()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for async callbacks."""
        self._event_loop = loop

    def create_task(self) -> str:
        """Create a new task and return its ID."""
        task_id = str(uuid.uuid4())[:8]  # Short ID for convenience
        with self._lock:
            self.tasks[task_id] = OptimizationTask(task_id=task_id)
        return task_id

    def get_task(self, task_id: str) -> Optional[OptimizationTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def start_optimization(
        self,
        task_id: str,
        request_data: Dict[str, Any]
    ) -> None:
        """Start optimization in a background thread.

        Args:
            task_id: Task ID
            request_data: Optimization request data
        """
        task = self.tasks.get(task_id)
        if not task:
            return

        task.status = TaskStatus.RUNNING

        # Submit to thread pool
        self.executor.submit(
            self._run_optimization_sync,
            task_id,
            request_data
        )

    def _run_optimization_sync(
        self,
        task_id: str,
        request_data: Dict[str, Any]
    ) -> None:
        """Run optimization synchronously in a worker thread."""
        task = self.tasks.get(task_id)
        if not task:
            return

        # Store request_data for potential re-evaluation
        task.request_data = request_data

        def progress_callback(info: ProgressInfo) -> None:
            """Callback for progress updates."""
            task.update_progress(info)

            # Broadcast to WebSocket clients
            if self._event_loop and task.websocket_clients:
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_progress(task_id, info.to_dict()),
                    self._event_loop
                )

        try:
            # Run the optimization
            response = run_optimization(
                request_data,
                progress_callback=progress_callback,
                cancellation_token=task.cancellation_token,
                max_resolution=config.max_resolution
            )

            # Convert response to dict
            result_dict = response.to_dict()

            if task.cancellation_token.is_cancelled:
                task.status = TaskStatus.CANCELLED
                task.error = "Cancelled by user"
            elif result_dict.get('success'):
                task.status = TaskStatus.COMPLETED
                task.result = result_dict
            else:
                task.status = TaskStatus.FAILED
                errors = result_dict.get('errors', [])
                task.error = errors[0].get('message', 'Unknown error') if errors else 'Unknown error'

        except Exception as e:
            import traceback
            traceback.print_exc()
            task.status = TaskStatus.FAILED
            task.error = str(e)

        finally:
            task.completed_at = time.time()

            # Notify completion via WebSocket
            if self._event_loop and task.websocket_clients:
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_complete(task_id, task.status.value),
                    self._event_loop
                )

    async def _broadcast_progress(
        self,
        task_id: str,
        progress_dict: Dict[str, Any]
    ) -> None:
        """Broadcast progress to all WebSocket clients."""
        task = self.tasks.get(task_id)
        if not task:
            return

        message = {
            "type": "progress",
            **progress_dict
        }

        # Copy list to avoid modification during iteration
        clients = list(task.websocket_clients)
        for client in clients:
            try:
                await client.send_json(message)
            except Exception:
                # Client disconnected, will be cleaned up elsewhere
                pass

    async def _broadcast_complete(
        self,
        task_id: str,
        status: str
    ) -> None:
        """Broadcast completion to all WebSocket clients."""
        task = self.tasks.get(task_id)
        if not task:
            return

        message = {
            "type": "complete",
            "status": status
        }

        clients = list(task.websocket_clients)
        for client in clients:
            try:
                await client.send_json(message)
            except Exception:
                pass

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.

        Args:
            task_id: Task ID to cancel

        Returns:
            True if cancellation was requested, False if task not found
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.cancellation_token.cancel("User requested cancellation")
        return True

    def add_websocket_client(self, task_id: str, websocket: Any) -> bool:
        """Add a WebSocket client to receive progress updates.

        Args:
            task_id: Task ID
            websocket: WebSocket connection

        Returns:
            True if added, False if task not found
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.websocket_clients.append(websocket)
        return True

    def remove_websocket_client(self, task_id: str, websocket: Any) -> None:
        """Remove a WebSocket client.

        Args:
            task_id: Task ID
            websocket: WebSocket connection to remove
        """
        task = self.tasks.get(task_id)
        if task and websocket in task.websocket_clients:
            task.websocket_clients.remove(websocket)

    def cleanup_old_tasks(self, max_age_seconds: int = None) -> int:
        """Clean up completed tasks older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age for completed tasks

        Returns:
            Number of tasks removed
        """
        max_age = max_age_seconds or config.task_cleanup_seconds
        current_time = time.time()
        removed = 0

        with self._lock:
            to_remove = []
            for task_id, task in self.tasks.items():
                if task.completed_at and (current_time - task.completed_at) > max_age:
                    to_remove.append(task_id)

            for task_id in to_remove:
                del self.tasks[task_id]
                removed += 1

        return removed

    def reevaluate_at_resolution(
        self,
        task_id: str,
        upsample_factor: int
    ) -> Optional[Dict[str, Any]]:
        """Re-evaluate optimization result at different resolution.

        Uses unified reevaluation module that handles all propagation types:
        - FFT: tiles phase k×k times (more periods → sharper diffraction)
        - ASM/SFR: interpolates phase to higher resolution

        Args:
            task_id: Task ID of completed optimization
            upsample_factor: Resolution multiplier (1-8)

        Returns:
            Dict with 'simulated_intensity', 'target_intensity', 'phase',
            'metrics', 'upsample_factor', and 'effective_pixel_size', or None on error
        """
        import numpy as np

        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.COMPLETED or not task.result:
            return None

        result = task.result.get('result', {})
        phase_data = result.get('phase')
        target_data = result.get('target_intensity')
        request_data = task.request_data or {}

        if not phase_data:
            return None

        # Extract parameters from request_data
        propagation_type = request_data.get('propagation_type', 'fft')
        pixel_size = request_data.get('pixel_size', 1e-6)  # Default 1um
        wavelength = request_data.get('wavelength', 532e-9)
        working_distance = request_data.get('working_distance')

        # target_span is the physical size in meters (stored as float, convert to tuple)
        # IMPORTANT: Must include margin factor to match the original optimization!
        # The optimization uses target_size_m = target_span * (1 + target_margin)
        target_span = request_data.get('target_span')
        advanced = request_data.get('advanced', {})
        target_margin = advanced.get('target_margin', 0.1)  # Default 10% margin
        margin_factor = 1.0 + target_margin

        if target_span:
            target_size_with_margin = target_span * margin_factor
            target_size = (target_size_with_margin, target_size_with_margin)
        else:
            target_size = None

        # Fallback for ASM/SFR: compute target_size from simulated_intensity if not stored
        if propagation_type in ('asm', 'sfr') and target_size is None:
            simulated_data = result.get('simulated_intensity')
            if simulated_data:
                # If target_size wasn't stored, compute from DOE size and margin
                # This is a fallback - normally target_span should be provided
                sim_np = np.array(simulated_data)
                if working_distance:
                    # Estimate target_size from typical scaling:
                    # For ASM/SFR, use natural Fresnel scale as reasonable fallback
                    natural_scale = wavelength * working_distance / pixel_size
                    target_size = (natural_scale, natural_scale)

        # Convert to numpy arrays
        phase_np = np.array(phase_data, dtype=np.float32)
        target_np = np.array(target_data, dtype=np.float32) if target_data else None

        # Extract target indices
        target_indices = extract_target_indices(target_np) if target_np is not None else []

        # If upsample_factor is 1, just return the original data with effective_pixel_size
        if upsample_factor <= 1:
            return {
                'simulated_intensity': result.get('simulated_intensity'),
                'target_intensity': result.get('target_intensity'),
                'phase': result.get('phase'),
                'metrics': result.get('metrics', {}),
                'upsample_factor': 1,
                'effective_pixel_size': pixel_size
            }

        try:
            # Use unified reevaluation function
            reeval_result = reevaluate_at_resolution(
                phase=phase_np,
                target=target_np if target_np is not None else np.zeros_like(phase_np),
                upsample_factor=upsample_factor,
                propagation_type=propagation_type,
                pixel_size=pixel_size,
                wavelength=wavelength,
                working_distance=working_distance,
                target_size=target_size,
                target_indices=target_indices
            )

            return {
                'simulated_intensity': reeval_result.simulated_intensity.tolist(),
                'target_intensity': reeval_result.target_intensity.tolist(),
                'phase': reeval_result.phase.tolist(),
                'metrics': reeval_result.metrics,
                'upsample_factor': reeval_result.upsample_factor,
                'effective_pixel_size': reeval_result.effective_pixel_size
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Reevaluate] Unified reevaluation failed: {e}, falling back to interpolation")
            # Fallback with effective_pixel_size
            fallback = self._fallback_upsample(result, upsample_factor)
            if fallback:
                fallback['effective_pixel_size'] = pixel_size / upsample_factor
            return fallback

    def _fallback_upsample(
        self,
        result: Dict[str, Any],
        upsample_factor: int
    ) -> Optional[Dict[str, Any]]:
        """Fallback upsampling using nearest-neighbor (kron-like) when re-propagation fails."""
        import numpy as np
        from scipy import ndimage

        simulated = result.get('simulated_intensity')
        target = result.get('target_intensity')
        phase = result.get('phase')

        if not simulated:
            return None

        try:
            sim_np = np.array(simulated, dtype=np.float32)
            is_1d = sim_np.ndim == 1 or (sim_np.ndim == 2 and min(sim_np.shape) == 1)

            # Nearest-neighbor upsample (kron-like: each pixel → k×k block)
            if upsample_factor > 1:
                if is_1d:
                    sim_np = sim_np.reshape(-1)
                    upsampled_sim = ndimage.zoom(sim_np, upsample_factor, order=0)
                else:
                    upsampled_sim = ndimage.zoom(sim_np, upsample_factor, order=0)
            else:
                upsampled_sim = sim_np

            # Upsample target if available (nearest-neighbor)
            upsampled_target = None
            if target is not None:
                target_np = np.array(target)
                if upsample_factor > 1:
                    if is_1d:
                        target_np = target_np.reshape(-1)
                    upsampled_target = ndimage.zoom(target_np, upsample_factor, order=0)
                else:
                    upsampled_target = target_np

            # Upsample phase if available (nearest-neighbor / kron-like)
            upsampled_phase = None
            if phase is not None:
                phase_np = np.array(phase)
                if upsample_factor > 1:
                    if is_1d:
                        phase_np = phase_np.reshape(-1)
                    upsampled_phase = ndimage.zoom(phase_np, upsample_factor, order=0)
                else:
                    upsampled_phase = phase_np

            return {
                'simulated_intensity': upsampled_sim.tolist(),
                'target_intensity': upsampled_target.tolist() if upsampled_target is not None else None,
                'phase': upsampled_phase.tolist() if upsampled_phase is not None else None,
                'metrics': result.get('metrics', {}),
                'upsample_factor': upsample_factor
            }
        except Exception:
            return None


# Global task manager instance
task_manager = TaskManager()
