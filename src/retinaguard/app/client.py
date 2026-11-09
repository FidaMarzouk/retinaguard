"""Thin HTTP client for the RetinaGuard API, used by the Streamlit app."""

from typing import Any

import httpx

TIMEOUT_S = 60.0


class ApiError(RuntimeError):
    """The API answered with an error, or could not be reached."""


class ApiClient:
    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=TIMEOUT_S, transport=transport)

    def _get(self, path: str, **params: Any) -> Any:
        try:
            response = self._client.get(
                path, params={k: v for k, v in params.items() if v is not None}
            )
        except httpx.HTTPError as error:
            raise ApiError(f"API not reachable: {error}") from error
        if response.status_code != 200:
            raise ApiError(f"{path}: {response.status_code} {response.text[:200]}")
        return response.json()

    def health(self) -> dict[str, Any]:
        result: dict[str, Any] = self._get("/health")
        return result

    def models(self) -> dict[str, Any]:
        result: dict[str, Any] = self._get("/models")
        return result

    def predictions(self, decision: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._get("/predictions", decision=decision, limit=limit)
        return result

    def predict(
        self, image: bytes, filename: str, *, site: str | None = None, lesions: bool = False
    ) -> dict[str, Any]:
        data = {"lesions": str(lesions).lower()} | ({"site": site} if site else {})
        try:
            response = self._client.post("/predict", files={"image": (filename, image)}, data=data)
        except httpx.HTTPError as error:
            raise ApiError(f"API not reachable: {error}") from error
        if response.status_code != 200:
            raise ApiError(f"/predict: {response.status_code} {response.text[:200]}")
        result: dict[str, Any] = response.json()
        return result
