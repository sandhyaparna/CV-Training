import datetime
import json
import logging
import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

# Pull data from environment to send to elastic search logs
ENV = os.environ.get("CV_ENV", "unknown")
CAMERA = os.environ.get("CAMERA_CONFIG", "unknown")
LOCATION = os.environ.get("LOCATION", "unknown")
DOCKER_TAG = os.environ.get("CV_DOCKER_TAG", "unknown")

# Remove ".yaml" string if it exists on the environment variable value
ENV = ENV.replace(".yaml", "").replace(".yml", "")
CAMERA = CAMERA.replace(".yaml", "").replace(".yml", "")


# Create the ElasticsearchHandler
class ElasticsearchHandler(logging.Handler):
    """
    Log handler to route logs from the base Python logging package to
    ElasticSearch or stdout.
    """

    def __init__(self) -> None:
        """
        Initializes ElasticSearch client with host, api key, and index
        taken form environment variables.
        """
        super().__init__()

        # Check environment variable to determine if logs should go to ES or stdout
        self.send_to_es = os.getenv("LOG_TO_ELASTICSEARCH", "False") == "True"

        if self.send_to_es:
            # Initialize the Elasticsearch client
            # The `hosts` argument is required, default to try to send to
            # locally running ES if the ELASTICSEARCH_HOSTS is not found.
            self.es = Elasticsearch(
                hosts=os.getenv("ELASTICSEARCH_HOSTS", "http://localhost:9200"),
                api_key=os.getenv("ELASTICSEARCH_API_KEY", None),
                ssl_show_warn=False,
                verify_certs=False,
            )

            self.index = os.getenv("ELASTICSEARCH_INDEX_NAME", "cv")

            # Add JSON formatter for ElasticSearch
            self.json_formatter = JsonFormatter()
        else:
            # Set up text string formatter for stdout
            stdout_format = "%(asctime)s - %(levelname)s - %(filename)s - %(message)s"
            self.text_formatter = logging.Formatter(stdout_format)

            # Create a standard stream handler for stdout with the text formatter
            self.stdout_handler = logging.StreamHandler()
            self.stdout_handler.setFormatter(self.text_formatter)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Whenever a log is written this method sends it to ElasticSearch
        if is an API key, otherwise it sends it to stdout.

        Args:
            record (logging.LogRecord): Log record

        Returns:
            None
        """
        # Send to elastic search if API key was found
        if self.send_to_es:
            # Format log entry as JSON
            log_entry = self.json_formatter.format(record)

            # Try to send to ElasticSearch. If it fails catch and log the
            # error to not interupt the rest of the code execution
            try:
                self.es.index(index=self.index, body=log_entry)
            except Exception as e:
                print(
                    f"Failed to send log entry to ElasticSearch: {e}"
                )  # Log the error
                self.handleError(record)

        else:
            # If the ElasticSearch API key is not found
            # Use the plain text formatter and log to stdout
            log_entry = self.text_formatter.format(record)
            self.stdout_handler.emit(record)


# Create a JSON Formatter for the log to be searchable in Kibana
class JsonFormatter(logging.Formatter):
    """
    Handles reformatting Python logs to JSON format for ElasticSearch
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Converts a LogRecord to a dictionary, then dumps the dictionary to a
        JSON string to be sent to ElasticSearch.

        Args:
            record (logging.LogRecord): Log record to format and send to ES

        Returns:
            (str): JSON formatted log
        """
        # Create a dictionary with log fields
        record_dict = {
            "time": datetime.datetime.utcnow().isoformat(),
            "name": record.name,
            "levelname": record.levelname,
            "pathname": record.pathname,
            "filename": record.filename,
            "module": record.module,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "threadName": record.threadName,
            "processName": record.processName,
            "message": record.getMessage(),
            "env": ENV,
            "camera": CAMERA,
            "location": LOCATION,
            "dockertag": DOCKER_TAG,
        }

        # If there's an exception, add it to the log record
        if record.exc_info:
            record_dict["exception"] = self.formatException(record.exc_info)

        # Return the log record as a JSON string
        return json.dumps(record_dict)


# Configure Python’s logging system with Elastic
def get_logger(logger_name: str) -> logging.Logger:
    """
    Returns a Logger object configured to send JSON logs to ElasticSearch
    if the API KEY is available as an env var, else the logger sends
    plain text logs to stdout.

    Args:
        logger_name (str): Name of the logger. Usually current __name__.

    Returns:
        logger (logging.Logger): A logger object
    """
    # Create the logger instance
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set the log level to DEBUG

    # Create ElasticsearchHandler
    es_handler = ElasticsearchHandler()

    # Add the Elasticsearch handler to the logger
    logger.addHandler(es_handler)

    return logger
CONFIG", "unknown")
LOCATION = os.environ.get("LOCATION", "unknown")
DOCKER_TAG = os.environ.get("DOCKER_TAG", "unknown")

# Remove ".yaml" string if it exists on the environment variable value
ENV = ENV.replace(".yaml", "").replace(".yml", "")
CAMERA = CAMERA.replace(".yaml", "").replace(".yml", "")


# Create the ElasticsearchHandler
class ElasticsearchHandler(logging.Handler):
    """
    Log handler to route logs from the base Python logging package to
    ElasticSearch or stdout.
    """

    def __init__(self) -> None:
        """
        Initializes ElasticSearch client with host, api key, and index
        taken form environment variables.
        """
        super().__init__()

        # Check environment variable to determine if logs should go to ES or stdout
        self.send_to_es = os.getenv("LOG_TO_ELASTICSEARCH", "False") == "True"

        if self.send_to_es:
            # Initialize the Elasticsearch client
            # The `hosts` argument is required, default to try to send to
            # locally running ES if the ELASTICSEARCH_HOSTS is not found.
            self.es = Elasticsearch(
                hosts=os.getenv("ELASTICSEARCH_HOSTS", "http://localhost:9200"),
                api_key=os.getenv("ELASTICSEARCH_API_KEY", None),
                ssl_show_warn=False,
                verify_certs=False,
            )

            self.index = os.getenv("ELASTICSEARCH_INDEX_NAME", "cv")

            # Add JSON formatter for ElasticSearch
            self.json_formatter = JsonFormatter()
        else:
            # Set up text string formatter for stdout
            stdout_format = "%(asctime)s - %(levelname)s - %(filename)s - %(message)s"
            self.text_formatter = logging.Formatter(stdout_format)

            # Create a standard stream handler for stdout with the text formatter
            self.stdout_handler = logging.StreamHandler()
            self.stdout_handler.setFormatter(self.text_formatter)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Whenever a log is written this method sends it to ElasticSearch
        if is an API key, otherwise it sends it to stdout.

        Args:
            record (logging.LogRecord): Log record

        Returns:
            None
        """
        # Send to elastic search if API key was found
        if self.send_to_es:
            # Format log entry as JSON
            log_entry = self.json_formatter.format(record)

            # Try to send to ElasticSearch. If it fails catch and log the
            # error to not interupt the rest of the code execution
            try:
                self.es.index(index=self.index, body=log_entry)
            except Exception as e:
                print(
                    f"Failed to send log entry to ElasticSearch: {e}"
                )  # Log the error
                self.handleError(record)

        else:
            # If the ElasticSearch API key is not found
            # Use the plain text formatter and log to stdout
            log_entry = self.text_formatter.format(record)
            self.stdout_handler.emit(record)


# Create a JSON Formatter for the log to be searchable in Kibana
class JsonFormatter(logging.Formatter):
    """
    Handles reformatting Python logs to JSON format for ElasticSearch
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Converts a LogRecord to a dictionary, then dumps the dictionary to a
        JSON string to be sent to ElasticSearch.

        Args:
            record (logging.LogRecord): Log record to format and send to ES

        Returns:
            (str): JSON formatted log
        """
        # Create a dictionary with log fields
        record_dict = {
            "time": datetime.datetime.utcnow().isoformat(),
            "name": record.name,
            "levelname": record.levelname,
            "pathname": record.pathname,
            "filename": record.filename,
            "module": record.module,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "threadName": record.threadName,
            "processName": record.processName,
            "message": record.getMessage(),
            "env": ENV,
            "camera": CAMERA,
            "location": LOCATION,
            "dockertag": DOCKER_TAG,
        }

        # If there's an exception, add it to the log record
        if record.exc_info:
            record_dict["exception"] = self.formatException(record.exc_info)

        # Return the log record as a JSON string
        return json.dumps(record_dict)


# Configure Python’s logging system with Elastic
def get_logger(logger_name: str) -> logging.Logger:
    """
    Returns a Logger object configured to send JSON logs to ElasticSearch
    if the API KEY is available as an env var, else the logger sends
    plain text logs to stdout.

    Args:
        logger_name (str): Name of the logger. Usually current __name__.

    Returns:
        logger (logging.Logger): A logger object
    """
    # Create the logger instance
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set the log level to DEBUG

    # Create ElasticsearchHandler
    es_handler = ElasticsearchHandler()

    # Add the Elasticsearch handler to the logger
    logger.addHandler(es_handler)

    return logger
