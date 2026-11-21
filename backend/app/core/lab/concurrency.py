"""Single-slot inference guard that protects the small Render instance."""

from __future__ import annotations

import gc
import threading


class InferenceSlot:
    def __init__(self, slots: int = 1):
        self._semaphore = threading.BoundedSemaphore(value=slots)
        # Tracked separately because BoundedSemaphore.locked()
        # is not available in every supported Python runtime.
        self._state_lock = threading.Lock()
        self._busy = False

    def try_acquire(self) -> bool:
        if not self._semaphore.acquire(blocking=False):
            return False
        with self._state_lock:
            self._busy = True
        return True

    def release(self) -> None:
        with self._state_lock:
            self._busy = False
        self._semaphore.release()
        # Encourage release of temporary Python objects.
        gc.collect()

    @property
    def busy(self) -> bool:
        with self._state_lock:
            return self._busy


INFERENCE_SLOT = InferenceSlot()
