import logging
from typing import Optional, Literal


LOG_LEVEL_FORMAT = {
    "DEBUG": "DEBG",
    "INFO": "INFO",
    "WARNING": "WARN",
    "ERROR": "EROR",
    "CRITICAL": "CRIT"
}

class LoggerManager:
    
    """A flexible logger manager class that supports both console and file output."""

    _instances = {}  

    def __new__(cls, package_name: str, *args, **kwargs):
        if package_name in cls._instances:
            return cls._instances[package_name]
        instance = super().__new__(cls)
        cls._instances[package_name] = instance
        return instance

    def __init__(
        self,
        package_name: str,
        log_file: Optional[str] = None,
        level: str = "INFO",
        output: Literal["c", "f", "a"] = "c",
        mode: str = "w",
        formatter: Optional[logging.Formatter] = None,
    ):
        """
        Initialize the logger manager.
        
        Args:
            package_name: Name for the module
            log_file: Path to log file (required if output includes 'file')
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            output: Where to send logs, can be 'c'(console), 'f'(file), or 'a' (all))
            mode: File mode ('w' for overwrite, 'a' for append)
            formatter: Custom formatter
        """
        
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.package_name = package_name
        self.log_file = log_file
        self.level = level
        self.output = output
        self.mode = mode

        self.formatter = formatter or logging.Formatter(
                '[%(asctime)s]%(message)s',
                datefmt='%m-%d %H:%M:%S'
            )
        self._logger = logging.getLogger(self.package_name)
        self._logger.propagate = False  
        self._setup_logger()

        self._initialized = True

    def update_config(
        self,
        log_file: Optional[str] = None,
        level: str = "INFO",
        output: Literal["c", "f", "a"] = "c",
        mode: str = "w",
    ) -> None:
        """Reconfigure the logger"""
        self.log_file = log_file
        self.level = level
        self.output = output
        self.mode = mode
        self._setup_logger()

    def set_logger_name(self, logger_name: str) -> None:
        """Change logger name"""
        self._logger.name = logger_name

    def set_formatter(self, formatter: logging.Formatter) -> None:
        """Change formatter"""
        self.formatter = formatter
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Configure the logger with current settings."""

        self._logger.handlers.clear()

        if self.output == "c":
            self._add_console_handler()
        elif self.output == "f":
            if not self.log_file:
                raise ValueError("log_file must be specified for file output")
            self._add_file_handler()
        elif self.output == "a":
            self._add_console_handler()
            if not self.log_file:
                raise ValueError("log_file must be specified for file output")
            self._add_file_handler()
        else:
            raise ValueError(f"Invalid output option: {self.output}")

        self._logger.setLevel(self.level)
        self._logger.propagate = False

    def _add_console_handler(self):
        ch = logging.StreamHandler()
        ch.setFormatter(self.formatter)
        self._logger.addHandler(ch)

    def _add_file_handler(self):
        fh = logging.FileHandler(self.log_file, mode=self.mode)
        fh.setFormatter(self.formatter)
        self._logger.addHandler(fh)

    @property
    def logger(self):
        return self._logger
    
    # message
    def _format_message(self, level: str, msg: str) -> str:
        """formated message"""
        return f"[{LOG_LEVEL_FORMAT[level]}] {self.package_name}: {msg}"
    
    def debug(self, msg: str) -> None:
        """Log debug message with aligned level prefix."""
        formatted_msg = self._format_message("DEBUG", msg)
        self._logger.debug(formatted_msg)
    
    def info(self, msg: str) -> None:
        """Log info message with aligned level prefix."""
        formatted_msg = self._format_message("INFO", msg)
        self._logger.info(formatted_msg)
    
    def warning(self, msg: str) -> None:
        """Log warning message with aligned level prefix."""
        formatted_msg = self._format_message("WARNING", msg)
        self._logger.warning(formatted_msg)
    
    def error(self, msg: str) -> None:
        """Log error message with aligned level prefix."""
        formatted_msg = self._format_message("ERROR", msg)
        self._logger.error(formatted_msg)
        
    def critical(self, msg: str) -> None:
        """Log critical message with aligned level prefix."""
        formatted_msg = self._format_message("CRITICAL", msg)
        self._logger.critical(formatted_msg)
        
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self._logger