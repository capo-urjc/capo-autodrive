import logging

# Define VERBOSE level (more detailed than DEBUG)
VERBOSE_LEVEL = 5
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

def verbose(self, message, *args, **kwargs):
    """Method to log at VERBOSE level"""
    if self.isEnabledFor(VERBOSE_LEVEL):
        self._log(VERBOSE_LEVEL, message, args, kwargs)

# Add method to logging.Logger class
if not hasattr(logging.Logger, "verbose"):
    logging.Logger.verbose = verbose

# Function to configure logging with VERBOSE level
def get_logger(name: str, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Set formatter
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)

    # Avoid duplicate handlers
    if not logger.hasHandlers():
        logger.addHandler(console_handler)

    return logger
