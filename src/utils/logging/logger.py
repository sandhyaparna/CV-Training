import logging
import logging.config
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class LoggingConfigurator:
    """
    Configures logging with a dynamic log file path.

    This class provides a way to configure logging for an application
    with a log file path that is dynamically generated based on the
    current date and time or current directory
    """

    def __init__(
        self,
        application: Union[str, int] = "application",
        config_path: Optional[str] = None,
        log_dir_path: Optional[str] = None,
        use_timestamp_logging: bool = False,
    ) -> None:
        """
        Initializes the logging configuration.

        Args:
            application: Name of the application. Defaults to "application".
            config_path: Path to the logging configuration file.
                Defaults to "default_logging_config.ini" in the current directory.
            log_dir_path: Base path for log files.
            use_timestamp_logging: Controls the format of the log file name.
                - If True: Log file names include timestamps in the format YYYY-MM-DD__HH_MM-application.log
                - If False (default): Log file names use a simple format: application.log
        """

        if not isinstance(application, (str, int)):
            raise TypeError("application must be a string or an integer")

        self.application = str(application)
        self.config_path = (
            config_path or Path(__file__).parent / "default_logging_config.ini"
        )
        self.log_dir_path = log_dir_path
        self.use_timestamp_logging = use_timestamp_logging

        # Call configure_logging on initialization
        self.configure_logging()

    def configure_logging(self) -> None:
        """
        Sets up the logging configuration based on provided or generated paths.
        """
        # Use provided path if available, otherwise create log directory within current folder
        log_dir = Path(self.log_dir_path) if self.log_dir_path else Path.cwd() / "logs"

        # Add a timestamp-based subdirectory if timestamp logging is enabled
        if self.use_timestamp_logging:
            current_datetime = datetime.utcnow()
            log_dir = log_dir / f"{current_datetime:%Y-%m}"

        # Construct the log file name based on timestamp preference
        log_file_name = (
            f"{current_datetime:%Y-%m-%d__%H_%M}-{self.application}.log"
            if self.use_timestamp_logging
            else f"{self.application}.log"
        )
        log_file_path = log_dir / log_file_name

        # Create directory structure if needed
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            if e.errno == 13:
                raise PermissionError(
                    f"Insufficient permissions to create log directory: {log_dir}"
                ) from e
            else:
                raise OSError(f"Failed to create log directory: {log_dir}") from e

        try:
            # Attempt to open the configuration file in read mode
            with open(self.config_path, "r"):
                pass  # file is implicitly opened and closed for existence check.
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Logging configuration file not found: {self.config_path}"
            ) from e
        except OSError as e:
            raise OSError(
                f"Failed to open logging configuration file: {self.config_path}"
            ) from e

        try:
            # Attempt to configure logging using the specified configuration file
            logging.config.fileConfig(
                self.config_path,
                disable_existing_loggers=False,
                defaults={"logfilename": log_file_path},
            )
        except KeyError as e:
            # If a key error occurs (likely missing section in the config file),
            # extract the missing section name and raise a more specific error message:
            missing_section = str(e).split("'")[1]
            raise ValueError(
                f"Logging configuration file is missing the {missing_section} section"
            ) from e

        # Get the logger for the current module
        self.logger = logging.getLogger(__name__)
        # Log a message indicating the log file being used
        self.logger.info(f"Logging to file {log_file_path}")
