import logging

__all__ = [
    "logger",
    "enable_logging",
    "disable_logging",
    "enable_file_logging",
    "disable_file_logging",
    "LOG_DEBUG",
    "LOG_INFO",
    "LOG_WARNING",
    "LOG_ERROR",
    "LOG_CRITICAL",
    "LOG_FATAL",
]

logger = logging.getLogger("fai")
logger.setLevel(logging.DEBUG)
logger.propagate = False

LOG_DEBUG = logging.DEBUG
LOG_INFO = logging.INFO
LOG_WARNING = logging.WARNING
LOG_ERROR = logging.ERROR
LOG_CRITICAL = logging.CRITICAL
LOG_FATAL = logging.FATAL


def enable_logging(level: int = LOG_INFO):
    """
    Enable logging.
    @param level: the logging level.
    """
    disable_logging()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def disable_logging():
    """
    Disable logging.
    """
    logger.setLevel(logging.CRITICAL)
    logger.handlers = []


def enable_file_logging(filename: str, level: int = LOG_INFO):
    """
    Enable logging to a file.
    @param filename: the filename.
    @param level: the logging level.
    """
    enable_logging(level)
    fh = logging.FileHandler(filename)
    fh.setLevel(level)
    logger.addHandler(fh)


def disable_file_logging():
    """
    Disable logging to a file.
    """
    logger.setLevel(logging.CRITICAL)
    if len(logger.handlers) > 1:
        logger.removeHandler(logger.handlers[1])
