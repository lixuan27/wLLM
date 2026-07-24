import logging
import sys

RESET = "\033[0m"
BRIGHT_GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"


class ColorFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)

        if record.levelno == logging.INFO:
            return f"{BRIGHT_GREEN}{message}{RESET}"
        elif record.levelno == logging.WARNING:
            return f"{YELLOW}{message}{RESET}"
        elif record.levelno == logging.ERROR:
            return f"{RED}{message}{RESET}"
        else:
            return message


def get_logger(name="app_logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        formatter = ColorFormatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger