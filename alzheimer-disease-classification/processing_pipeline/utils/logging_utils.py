# =====================================================================
# ADNI MRI PROCESSING PIPELINE - LOGGING UTILITIES
# =====================================================================
# This file provides utility functions for setting up consistent logging
# across the entire pipeline.

import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    """
    A logging handler that redirects logging output to `tqdm.write`,
    ensuring that logging messages do not interfere with the tqdm progress bars.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging(step_name: str, base_log_dir: Path) -> logging.Logger:
    """
    Sets up a logger for a specific pipeline step.

    This creates a unique, timestamped directory for the logs of a pipeline run
    and configures a logger to output to both a file within that directory
    and the console (in a tqdm-friendly way).

    Args:
        step_name (str): The name of the pipeline step (e.g., 'data_splitter').
        base_log_dir (Path): The root directory where logs should be stored.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Use a shared timestamp for the entire pipeline run if not already set
    if not hasattr(setup_logging, "log_dir"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        setup_logging.log_dir = base_log_dir / timestamp
    
    setup_logging.log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = setup_logging.log_dir / f"{step_name}.log"
    
    # Get logger and remove existing handlers to prevent duplicate messages
    logger = logging.getLogger(step_name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    
    # Create console handler compatible with tqdm
    console_handler = TqdmLoggingHandler()

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(logging.Formatter('%(message)s')) # Simpler format for console

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent logging from propagating to the root logger
    logger.propagate = False

    return logger