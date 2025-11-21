import logging

import requests

from cds_vision_tools.utils.logging.logger import LoggingConfigurator

LoggingConfigurator()
logger = logging.getLogger(__name__)


class BackendAPI:
    """
    Interacts with a generic backend API using GET requests,
    supporting public and company-specific APIs with optional token-based authentication.
    """

    def __init__(
        self,
        api_url: str,
        access_token: str = None,
        health_check_endpoint: str = None,
        headers: dict = None,
    ):
        """
        Initializes the BackendAPI instance.

        Args:
            base_url (str): The base URL of the backend API (e.g., "https://api.example.com").
            access_token (str, optional): An access token for authentication (if not a public API).
            health_check_endpoint (str, optional): The endpoint for health check (e.g., "/health"). Defaults to None.
            headers (dict, optional): A dictionary of headers to include in requests.
        """
        self.api_url = api_url
        self.access_token = access_token
        self.health_check_endpoint = health_check_endpoint
        self.headers = headers

        self.api_health_check()  # Perform health check if enabled

    def _build_url_and_headers(self, endpoint: str) -> tuple[str, dict]:
        """
        Constructs the complete API endpoint URL and optionally adds authorization header
        if necessary based on API type and access token availability.

        Args:
            endpoint (str): The API endpoint to access (relative to base URL).

        Returns:
            tuple: A tuple containing the endpoint URL (str) and the headers dictionary (dict).
        """

        # Combine API URL and endpoint for a complete URL
        endpoint_url = f"{self.api_url}/{endpoint}"

        # Ensure headers dictionary exists
        headers = self.headers or {}

        # Add authorization header only for private APIs with a valid access token
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        return endpoint_url, headers

    def api_health_check(self) -> bool:
        """
        Checks the API health using the provided health check endpoint (if any).

        Args:
            None

        Returns:
            bool: True if the health check is successful, False otherwise.
        """

        if not self.health_check_endpoint:
            logger.info("Skipping health check as no endpoint is configured.")
            return True  # Assume healthy if no health check endpoint is specified

        # Construct the full health check URL and headers
        endpoint_url, headers = self._build_url_and_headers(self.health_check_endpoint)

        try:
            # Send a GET request to the health check endpoint
            response = requests.get(endpoint_url, headers=headers)
            response.raise_for_status()  # Raise an exception if the request fails (for non-200 status codes)

            # Log successful health check and response details
            logger.info("API Check Success")
            logger.info(f"API returned : {response.json()}")
            return True

        except requests.exceptions.RequestException as e:
            # Log health check failure and error details
            logger.error(f"API health check failed: {e}")
            return False

    def get(self, endpoint: str, params=None):
        """
        Sends a GET request to the specified API endpoint and returns the parsed JSON response.

        Args:
            endpoint (str): The API endpoint to access (relative to base URL).
            params (dict, optional): A dictionary of query parameters for the request.

        Returns:
            dict: The JSON-parsed response data from the API.

        Raises:
            ValueError: If the response size is less than or equal to 2 bytes.
            requests.exceptions.RequestException: If the request fails due to other reasons (e.g., network issues, server errors).
        """

        endpoint_url, headers = self._build_url_and_headers(endpoint)

        try:
            response = requests.get(endpoint_url, params=params, headers=headers)
            response.raise_for_status()  # Raise error for non-200 status codes

            # Return the JSON-parsed response data
            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making GET request to {endpoint_url}: {e}") from e
