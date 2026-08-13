"""Thin REST client for the Opensolr platform APIs.

Two base URLs, by platform design:
- Management API (index list/info/create): https://opensolr.com/solr_manager/api
- AI API (embed, batch_embed, embed_and_search, ai_summary): https://api.opensolr.com/solr_manager/api

Direct Solr access (select/update) goes to the index's own host, resolved via
``get_core_info`` (``connection_url`` + HTTP basic auth).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import httpx

MGMT_BASE = "https://opensolr.com/solr_manager/api"
AI_BASE = "https://api.opensolr.com/solr_manager/api"

#: Only these Opensolr environments run vector-enabled (Solr 9.x + knn_vector
#: schema) servers. ``create_index`` for a vector index must target one of them.
VECTOR_LOCATIONS: Dict[str, str] = {
    "us": "CHICAGO-96",
    "de": "DE-SOLR-9",
    "fi": "FINLAND9",
}

#: Server-side limit for one batch_embed call.
BATCH_EMBED_MAX = 50


class OpensolrError(RuntimeError):
    """Raised when an Opensolr API call fails."""


class OpensolrClient:
    """Authenticated client for Opensolr management + AI endpoints.

    Args:
        email: Opensolr account email.
        api_key: Opensolr API key (Account > API in the control panel).
        timeout: Per-request timeout in seconds. Embedding calls run on GPU
            infrastructure and are usually fast, but cold starts happen.
    """

    def __init__(self, email: str, api_key: str, timeout: float = 120.0) -> None:
        self.email = email
        self.api_key = api_key
        self._http = httpx.Client(timeout=timeout, follow_redirects=True)
        self._core_info_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # low level                                                          #
    # ------------------------------------------------------------------ #

    def _auth_params(self) -> Dict[str, str]:
        return {"email": self.email, "api_key": self.api_key}

    def _request(self, base: str, method: str, params: Dict[str, Any]) -> Any:
        url = f"{base}/{method}"
        data = {**self._auth_params(), **params}
        resp = self._http.post(url, data=data)
        if resp.status_code >= 500:
            raise OpensolrError(f"{method}: HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            body = resp.json()
        except json.JSONDecodeError as exc:
            raise OpensolrError(f"{method}: non-JSON response: {resp.text[:200]}") from exc
        if isinstance(body, dict) and body.get("status") is False:
            raise OpensolrError(f"{method}: {body.get('msg', body)}")
        return body

    def mgmt(self, method: str, **params: Any) -> Any:
        return self._request(MGMT_BASE, method, params)

    def ai(self, method: str, **params: Any) -> Any:
        return self._request(AI_BASE, method, params)

    # ------------------------------------------------------------------ #
    # management                                                         #
    # ------------------------------------------------------------------ #

    def get_index_list(self) -> List[Dict[str, str]]:
        return self.mgmt("get_index_list")

    def get_core_info(self, index: str, refresh: bool = False) -> Dict[str, Any]:
        """Resolve an index's Solr endpoint + HTTP auth. Cached per client."""
        if not refresh and index in self._core_info_cache:
            return self._core_info_cache[index]
        body = self.mgmt("get_core_info", core_name=index)
        msg = body.get("msg") if isinstance(body, dict) else None
        if not isinstance(msg, dict) or "info" not in msg:
            raise OpensolrError(f"get_core_info({index}): unexpected response: {str(body)[:200]}")
        info = msg["info"]
        self._core_info_cache[index] = info
        return info

    def create_index(self, index: str, location: str = "us") -> Dict[str, Any]:
        """Create a vector-enabled index on one of the vector locations.

        ``location`` is one of :data:`VECTOR_LOCATIONS` keys ("us", "de", "fi")
        or a raw Opensolr environment identifier.
        """
        env = VECTOR_LOCATIONS.get(location.lower(), location)
        if env not in VECTOR_LOCATIONS.values():
            raise ValueError(
                f"Vector-enabled indexes are only available in these locations: "
                f"{sorted(VECTOR_LOCATIONS)} (got {location!r})"
            )
        return self.mgmt(
            "create_index", index_name=index, core_type="generic", server_country=env
        )

    # ------------------------------------------------------------------ #
    # AI                                                                 #
    # ------------------------------------------------------------------ #

    def embed(self, index: str, text: str, is_query: bool = False) -> List[float]:
        body = self.ai(
            "embed", index_name=index, payload=text, is_query="1" if is_query else "0"
        )
        if not isinstance(body, list) or not body:
            raise OpensolrError(f"embed: unexpected response: {str(body)[:200]}")
        return body

    def batch_embed(self, index: str, texts: List[str]) -> List[List[float]]:
        """Embed many texts. Chunks transparently at the server's batch limit."""
        out: List[List[float]] = []
        for i in range(0, len(texts), BATCH_EMBED_MAX):
            chunk = texts[i : i + BATCH_EMBED_MAX]
            resp = self._http.post(
                f"{AI_BASE}/batch_embed",
                json={
                    **self._auth_params(),
                    "index_name": index,
                    "payloads": chunk,
                },
            )
            try:
                body = resp.json()
            except json.JSONDecodeError as exc:
                raise OpensolrError(f"batch_embed: non-JSON response: {resp.text[:200]}") from exc
            if isinstance(body, dict) and body.get("status") is False:
                raise OpensolrError(f"batch_embed: {body.get('msg', body)}")
            embeddings = body.get("embeddings") if isinstance(body, dict) else None
            if not isinstance(embeddings, list) or len(embeddings) != len(chunk):
                raise OpensolrError(f"batch_embed: unexpected response: {str(body)[:200]}")
            out.extend(embeddings)
        return out

    def embed_and_search(self, index: str, query: str, rows: int = 10, **params: Any) -> Dict[str, Any]:
        """Server-side one-shot: embed the query, run hybrid search, return docs."""
        body = self.ai(
            "embed_and_search",
            index_name=index,
            q=query,
            rows=rows,
            **{"in": "all", "fresh": "no", **params},
        )
        return body

    # ------------------------------------------------------------------ #
    # direct Solr                                                        #
    # ------------------------------------------------------------------ #

    def solr_endpoint(self, index: str) -> Tuple[str, Optional[Tuple[str, str]]]:
        """Return (base_url, basic_auth) for the index's native Solr API."""
        info = self.get_core_info(index)
        url = info.get("connection_url")
        if not url:
            raise OpensolrError(f"No connection_url for index {index!r}")
        auth = None
        if info.get("auth_username"):
            auth = (info["auth_username"], info.get("auth_password") or "")
        return url, auth

    def solr_select(self, index: str, params: Dict[str, Any]) -> Dict[str, Any]:
        base, auth = self.solr_endpoint(index)
        resp = self._http.post(f"{base}/select", data={"wt": "json", **params}, auth=auth)
        resp.raise_for_status()
        return resp.json()

    def solr_update(self, index: str, payload: Any, commit: bool = True) -> Dict[str, Any]:
        base, auth = self.solr_endpoint(index)
        params = {"commit": "true"} if commit else {"commitWithin": "10000"}
        resp = self._http.post(
            f"{base}/update",
            params=params,
            json=payload,
            auth=auth,
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._http.close()
