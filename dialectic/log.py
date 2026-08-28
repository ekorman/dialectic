import logging
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

PURPLE = "\033[38;5;141m"
DEEP_PURPLE = "\033[38;5;99m"
PINK = "\033[38;5;213m"
HOT_PINK = "\033[38;5;205m"
LIGHT_PINK = "\033[38;5;218m"

LEVEL_COLORS = {
    logging.DEBUG: DEEP_PURPLE,
    logging.INFO: PINK,
    logging.WARNING: HOT_PINK,
    logging.ERROR: f"{BOLD}{HOT_PINK}",
    logging.CRITICAL: f"{BOLD}\033[38;5;201m",
}


class PinkPurpleFormatter(logging.Formatter):
    def __init__(self, use_color: bool) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        level = record.levelname
        message = record.getMessage()
        if not self.use_color:
            return f"{timestamp} | dialectic | {level} | {message}"
        level_color = LEVEL_COLORS.get(record.levelno, PINK)
        sep = f"{DIM}{PURPLE}|{RESET}"
        return (
            f"{DIM}{PURPLE}{timestamp}{RESET} {sep} "
            f"{BOLD}{HOT_PINK}dialectic{RESET} {sep} "
            f"{level_color}{level:<8}{RESET} {sep} "
            f"{LIGHT_PINK}{message}{RESET}"
        )


log = logging.getLogger("dialectic")
log.setLevel(logging.INFO)

_handler = logging.StreamHandler()
_handler.setFormatter(PinkPurpleFormatter(use_color=sys.stderr.isatty()))
log.addHandler(_handler)
