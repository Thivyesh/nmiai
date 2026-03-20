"""Tripletex API tools for the LangChain agent."""

import json

import requests
from langchain_core.tools import tool


class TripletexClient:
    """HTTP client for Tripletex API calls via the competition proxy."""

    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        resp = requests.get(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            auth=self.auth,
            params=params,
        )
        resp.raise_for_status()
        return resp.json()

    def post(self, endpoint: str, data: dict) -> dict:
        resp = requests.post(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            auth=self.auth,
            json=data,
        )
        resp.raise_for_status()
        return resp.json()

    def put(self, endpoint: str, data: dict) -> dict:
        resp = requests.put(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            auth=self.auth,
            json=data,
        )
        resp.raise_for_status()
        return resp.json()

    def delete(self, endpoint: str) -> int:
        resp = requests.delete(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            auth=self.auth,
        )
        resp.raise_for_status()
        return resp.status_code


# Module-level client reference, set at runtime by the agent
_client: TripletexClient | None = None


def set_client(client: TripletexClient) -> None:
    global _client
    _client = client


def _get_client() -> TripletexClient:
    if _client is None:
        raise RuntimeError("TripletexClient not initialized. Call set_client() first.")
    return _client


@tool
def tripletex_get(endpoint: str, params: str = "{}") -> str:
    """GET a Tripletex API endpoint.

    Args:
        endpoint: API path, e.g. "/employee", "/customer", "/invoice".
        params: JSON string of query parameters, e.g. '{"fields": "id,name", "count": "10"}'.

    Returns:
        JSON response as string.
    """
    client = _get_client()
    parsed_params = json.loads(params) if params else None
    result = client.get(endpoint, parsed_params)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tripletex_post(endpoint: str, body: str) -> str:
    """POST to a Tripletex API endpoint to create a resource.

    Args:
        endpoint: API path, e.g. "/employee", "/customer", "/invoice".
        body: JSON string of the request body.

    Returns:
        JSON response as string.
    """
    client = _get_client()
    data = json.loads(body)
    result = client.post(endpoint, data)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tripletex_put(endpoint: str, body: str) -> str:
    """PUT to a Tripletex API endpoint to update a resource.

    Args:
        endpoint: API path, e.g. "/employee/123", "/customer/456".
        body: JSON string of the request body.

    Returns:
        JSON response as string.
    """
    client = _get_client()
    data = json.loads(body)
    result = client.put(endpoint, data)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tripletex_delete(endpoint: str) -> str:
    """DELETE a Tripletex API resource.

    Args:
        endpoint: API path with ID, e.g. "/employee/123", "/travelExpense/456".

    Returns:
        HTTP status code as string.
    """
    client = _get_client()
    status = client.delete(endpoint)
    return f"Deleted successfully (HTTP {status})"


ALL_TOOLS = [tripletex_get, tripletex_post, tripletex_put, tripletex_delete]
