"""Tripletex API tools for the LangChain agent."""

import json

import requests
from langchain_core.tools import tool


class TripletexClient:
    """HTTP client for Tripletex API calls via the competition proxy."""

    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        resp = requests.request(
            method,
            f"{self.base_url}/{endpoint.lstrip('/')}",
            auth=self.auth,
            timeout=30,
            **kwargs,
        )
        if not resp.ok:
            try:
                error_body = resp.json()
            except Exception:
                error_body = resp.text
            return {"_error": True, "status_code": resp.status_code, "detail": error_body}
        if method == "DELETE" or not resp.content:
            return {"_success": True, "status_code": resp.status_code}
        return resp.json()

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        return self._request("GET", endpoint, params=params)

    def post(self, endpoint: str, data: dict) -> dict:
        return self._request("POST", endpoint, json=data)

    def put(self, endpoint: str, data: dict | None = None, params: dict | None = None) -> dict:
        kwargs = {}
        if data:
            kwargs["json"] = data
        if params:
            kwargs["params"] = params
        return self._request("PUT", endpoint, **kwargs)

    def delete(self, endpoint: str) -> dict:
        return self._request("DELETE", endpoint)


# Module-level client — safe because requests are serialized via asyncio.Lock in agent.py
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
def tripletex_put(endpoint: str, body: str = "{}", params: str = "{}") -> str:
    """PUT to a Tripletex API endpoint to update a resource.

    Args:
        endpoint: API path, e.g. "/employee/123", "/customer/456".
            Can include query params in the URL for action endpoints, e.g.
            "/employee/entitlement/:grantEntitlementsByTemplate?employeeId=123&template=ALL_PRIVILEGES"
        body: JSON string of the request body. Use "{}" if no body needed.
        params: JSON string of query parameters. Alternative to putting params in the URL.

    Returns:
        JSON response as string.
    """
    client = _get_client()
    data = json.loads(body) if body and body != "{}" else None
    parsed_params = json.loads(params) if params and params != "{}" else None
    result = client.put(endpoint, data, params=parsed_params)
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
    result = client.delete(endpoint)
    return json.dumps(result, ensure_ascii=False, indent=2)


from task2_tripletex.api_docs_tool import lookup_api_docs
from task2_tripletex.devdocs_tool import search_tripletex_docs
from task2_tripletex.task_patterns_tool import lookup_task_pattern
from task2_tripletex.web_search_tool import web_search
from task2_tripletex.workflow_tools import get_payload_template, get_task_workflow

# Researcher: workflow + payload templates + API reference
PLANNER_TOOLS = [get_task_workflow, get_payload_template, tripletex_get, lookup_api_docs, search_tripletex_docs, web_search]

# Executor: write tools + payload templates for error recovery
EXECUTOR_TOOLS = [tripletex_post, tripletex_put, tripletex_delete, tripletex_get, get_payload_template, lookup_api_docs]
