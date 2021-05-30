"""
Timer as context manager.
"""

import logging
import time
from contextlib import contextmanager


@contextmanager
def timer(name, disable=False):
    """Simple timer as context manager."""

    start = time.time()
    yield
    if not disable:
        logging.info(f'[{name}] done in {(time.time() - start)*1000:.1f} ms')
