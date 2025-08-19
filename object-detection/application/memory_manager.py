"""
Memory management utilities for the object detection application.
"""

import time
import logging
import torch
import gc
import psutil
from typing import Optional

from config.config import DEFAULT_MEMORY_CHECK_INTERVAL, DEFAULT_MEMORY_CLEANUP_INTERVAL


class MemoryManager:
    """Handles memory monitoring and cleanup"""
    
    def __init__(self, check_interval: int = DEFAULT_MEMORY_CHECK_INTERVAL, 
                 cleanup_interval: int = DEFAULT_MEMORY_CLEANUP_INTERVAL):
        self.last_memory_check = time.time()
        self.last_memory_cleanup = time.time()
        self.check_interval = check_interval
        self.cleanup_interval = cleanup_interval
    
    def check_memory_usage(self) -> None:
        """Monitor memory usage and log warnings"""
        current_time = time.time()
        if current_time - self.last_memory_check > self.check_interval:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                if memory_mb > 1000:  # Warning if over 1GB
                    logging.warning(f"High memory usage: {memory_mb:.1f} MB")
                self.last_memory_check = current_time
            except Exception as e:
                logging.debug(f"Could not check memory usage: {e}")
    
    def cleanup_memory(self) -> None:
        """Clean up memory and temporary files"""
        current_time = time.time()
        if current_time - self.last_memory_cleanup > self.cleanup_interval:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()
                self.last_memory_cleanup = current_time
                logging.info("Memory cleanup completed")
            except Exception as e:
                logging.warning(f"Memory cleanup failed: {e}")
    
    def force_cleanup(self) -> None:
        """Force immediate memory cleanup"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
            self.last_memory_cleanup = time.time()
            logging.info("Forced memory cleanup completed")
        except Exception as e:
            logging.warning(f"Forced memory cleanup failed: {e}") 