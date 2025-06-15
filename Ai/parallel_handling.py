import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Any, Callable

# ──────────── THREAD POOL & PROGRESS MAP SETUP ────────────

# 1) Create a pool of worker threads. Adjust max_workers as desired.
executor = ThreadPoolExecutor(max_workers=4)

# 2) A lock to protect access to task IDs and our progress/future maps.
_task_counter_lock = threading.Lock()
_task_counter = 0

# 3) Map from task_id (int) → Future object.
futures_map: Dict[int, Future] = {}

# 4) Map from task_id → progress (an integer 0..100, or -1 on error)
progress_map: Dict[int, tuple[str, int]] = {}
_progress_lock = threading.Lock()


def submit_task(fn: Callable[..., Any], *args, **kwargs) -> int:
    """
    Submit fn(*args, **kwargs) to the thread pool.
    Returns a unique integer task_id.
    """
    global _task_counter
    with _task_counter_lock:
        task_id = _task_counter
        _task_counter += 1

    # Initialize progress to 0%
    with _progress_lock:
        progress_map[task_id] = ('Starting the process...', 0)

    # Submit a wrapper that knows about task_id
    future = executor.submit(_run_with_progress, task_id, fn, *args, **kwargs)
    futures_map[task_id] = future
    return task_id

def cancel_task(task_id):
    """
    Cancel a running task. If the thread hasn’t started yet, Future.cancel() will prevent it from running.
    If it’s already running, we’ll mark progress_map[task_id] as 'CANCELLED' so that
    the worker function can notice and exit cooperatively.
    """
    future: Future = futures_map.get(task_id)
    if future is None:
        return None

    with _progress_lock:
        progress_map[task_id] = ('CANCELED', -1)


    return "CANCELED"

def get_task_status(task_id: int) -> str:
    """
    Get a coarse status: "PENDING", "RUNNING", "DONE", or "FAILED",
    based on the Future’s state.
    """
    with _progress_lock:
        status_str, _ = progress_map.get(task_id)
    if status_str == 'CANCELED':
        return 'CANCELED'

    future = futures_map.get(task_id)
    if future is None:
        return "UNKNOWN_TASK_ID"

    if future.running():
        return "RUNNING"
    if future.done():
        return "ERROR" if future.exception() else "DONE"
    return "PENDING"


def get_task_progress(task_id: int):
    """
    Return the latest integer in [0..100] for this task_id.
    If the task has failed, return -1. If the ID is unknown, raise KeyError.
    """
    with _progress_lock:
        if task_id not in progress_map:
            return None
        return progress_map[task_id]


def get_task_result(task_id: int) -> Any:
    """
    Block until the task completes. If it succeeded, return its result.
    If it raised an exception, re‐raise it here.
    """
    future = futures_map.get(task_id)
    if future is None:
        return None
    return future.result()


def _run_with_progress(task_id: int, fn: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Internal helper that wraps the user‐provided fn.
    It ensures that, if fn completes successfully, progress_map[task_id] is set to 100.
    If fn raises, we set progress_map[task_id] = -1 (meaning “error”) and re‐raise.
    """
    try:
        result = fn(task_id, *args, **kwargs)
        # On normal completion, mark 100%
        with _progress_lock:
            progress_map[task_id] = ('DONE', 100)
        return result

    except Exception as exc:
        # On error, mark progress as -1
        with _progress_lock:
            progress_map[task_id] = ('ERROR', -1)
        raise

def update_progress(task_id: int, percent: int, status: str = "Unknown"):
    """
    Safely set progress_map[task_id] = percent (0 ≤ percent ≤ 100),
    or -1 for failure. Caller must ensure percent is in [0..100], or -1.
    """
    if not (0 <= percent <= 100 or percent == -1):
        raise ValueError("Progress must be 0..100 or -1")

    with _progress_lock:
        # Only update if this task_id still exists
        if task_id in progress_map:
            progress_map[task_id] = (status, percent)

def get_progress(task_id: int):
    with _progress_lock:
        if task_id in progress_map:
            return progress_map[task_id]

def verify_for_cancellation(task_id):
    with _progress_lock:
        if task_id in progress_map:
            status, _ = progress_map[task_id]
            if status == 'CANCELED':
                return True
            return False
