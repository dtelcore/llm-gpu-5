# logging_config.py
"""
Global Logging Configuration for Kepler GT 730 GPU Training System.

Provides centralized logging to both console (terminal) and file (output/logs/).
All modules can import and use this logger for consistent output formatting.

Usage:
    from logging_config import logger
    
    logger.info("Starting training...")
    logger.warning("VRAM usage high: 850MB")
    logger.error("GPU kernel compilation failed")
    logger.debug("Batch shape: (2, 8, 16)")
"""

import logging
import os
import sys
from datetime import datetime


class SafeConsoleHandler(logging.StreamHandler):
    """Console handler that degrades unsupported Unicode instead of crashing."""

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                encoding = getattr(stream, 'encoding', None) or 'utf-8'
                safe_msg = msg.encode(encoding, errors='replace').decode(encoding, errors='replace')
                stream.write(safe_msg + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def _prepare_console_stream():
    """Prefer stdout and configure it for UTF-8 when the runtime supports it."""
    stream = sys.stdout
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    return stream


def setup_logging(log_level=logging.INFO, log_dir='output/logs', log_filename=None):
    """Configure global logging to console and file.
    
    Args:
        log_level (int): Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir (str): Directory to save log files (created if not exists)
        log_filename (str): Custom log filename (without .log extension). If None, uses timestamp.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger instance
    logger = logging.getLogger('KepleGPT')
    logger.setLevel(log_level)
    logger.propagate = False
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        logger.handlers.clear()
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename: use custom name or timestamped default
    if log_filename:
        log_file = os.path.join(log_dir, f'{log_filename}.log')
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    
    # Console Handler (prefer stdout because stderr on Windows PowerShell often remains CP1252)
    console_handler = SafeConsoleHandler(stream=_prepare_console_stream())
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File Handler (detailed logging)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # File always captures all levels
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Log initialization message
    logger.info(f"Logging initialized | Level: {logging.getLevelName(log_level)} | File: {log_file}")
    
    return logger


# Initialize logger at module import time
logger = setup_logging(log_level=logging.INFO)


def get_logger():
    """Get the global logger instance.
    
    Returns:
        logging.Logger: Global KepleGPT logger
    """
    return logger


def set_log_level(level):
    """Change global log level at runtime.
    
    Args:
        level (int): New logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
    logger.info(f"Log level changed to {logging.getLevelName(level)}")
