"""Keep cyclic CUDA graph cleanup outside solver initialization captures."""
from contextlib import contextmanager
import gc
from threading import Lock


_initialization_lock = Lock()


@contextmanager
def capture_initialization():
    """Protect warmup and all capture blocks; never wrap graph replay.

    A discarded driver can retain graphs through Python reference cycles.
    Collect those before capture, then defer automatic cyclic collection so
    their C++ graph destructors cannot invalidate a new CUDA capture. The
    caller must enter this scope outside capture, keep live graphs referenced,
    and let torch.cuda.graph perform its usual stream synchronization. This
    lock serializes our capture initializers because GC state is process-wide.
    """
    with _initialization_lock:
        enabled = gc.isenabled()
        gc.disable()
        try:
            gc.collect()
            yield
        finally:
            if enabled:
                gc.enable()
            else:
                gc.disable()
