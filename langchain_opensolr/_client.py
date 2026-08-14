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

#: Convenience aliases for Opensolr's vector-enabled environments. The
#: authoritative list is served live by the platform (``vector_regions``
#: endpoint) — new regions become valid automatically, and additional
#: dedicated regions can be deployed on request (paid): support@opensolr.com.
VECTOR_LOCATIONS: Dict[str, str] = {
    "us": "CHICAGO-96",
    "de": "DE-SOLR-9",
    "fi": "FINLAND9",
}


def resolve_location(location: str) -> str:
    """Map a friendly alias ("us"/"de"/"fi") to its environment identifier.

    Unknown values pass through unchanged — validity is decided against the
    live ``vector_regions`` list (or, ultimately, by the server), so newly
    deployed vector regions work without a package upgrade.
    """
    return VECTOR_LOCATIONS.get(location.strip().lower(), location.strip())

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

    def vector_regions(self) -> List[Dict[str, str]]:
        """Live list of vector-enabled environments (Solr 9.x + knn_vector +
        hybrid parser): ``[{environment, country, solr_version}, ...]``.

        Cached per client. Additional dedicated regions can be deployed on
        request (paid) — contact support@opensolr.com.
        """
        if not hasattr(self, "_vector_regions_cache"):
            body = self.mgmt("vector_regions")
            self._vector_regions_cache = body if isinstance(body, list) else []
        return self._vector_regions_cache

    def create_index(self, index: str, location: str = "us") -> Dict[str, Any]:
        """Create a vector-enabled index in a vector location.

        ``location`` is an alias ("us", "de", "fi") or a raw Opensolr
        environment identifier. Validated against the live ``vector_regions``
        list when reachable; otherwise the server has the final word.
        """
        env = resolve_location(location)
        try:
            live = {r["environment"] for r in self.vector_regions()}
        except OpensolrError:
            live = set(VECTOR_LOCATIONS.values())  # offline fallback
        if live and env not in live:
            raise ValueError(
                f"{location!r} is not a vector-enabled Opensolr location. "
                f"Currently available: {sorted(live)}. Additional regions can "
                f"be deployed on request — contact support@opensolr.com."
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

    def ingest(self, index: str, documents: List[Dict[str, Any]], wait: bool = False, timeout: float = 180.0) -> Dict[str, Any]:
        """Queue documents through the Opensolr Data Ingestion API (async).

        Embeddings, sentiment, language detection, and all crawler-identical
        derived fields are computed server-side. Max 50 documents per call.
        Returns ``{status, msg, job_id, total_docs, doc_ids}``. With
        ``wait=True``, polls ``ingest_status`` until the job completes
        (the queue is processed every minute).
        """
        resp = self._http.post(
            f"{AI_BASE}/ingest",
            json={**self._auth_params(), "core_name": index, "documents": documents},
        )
        try:
            body = resp.json()
        except json.JSONDecodeError as exc:
            raise OpensolrError(f"ingest: non-JSON response: {resp.text[:200]}") from exc
        if isinstance(body, dict) and body.get("status") is False:
            raise OpensolrError(f"ingest: {body.get('msg', body)}: {body.get('errors', '')}")
        if wait and body.get("job_id"):
            import time
            deadline = time.time() + timeout
            while time.time() < deadline:
                st = self.ingest_status(body["job_id"])
                job = st.get("job", {}) if isinstance(st, dict) else {}
                state = int(job.get("state", 0)) if str(job.get("state", "")).lstrip("-").isdigit() else 0
                if state == 1:
                    body["final_status"] = st
                    return body
                if state in (3, 4):
                    raise OpensolrError(f"ingest: job {body['job_id']} {job.get('state_label', state)}: {job.get('error')}")
                time.sleep(5)
            raise OpensolrError(f"ingest: job {body['job_id']} not completed within {timeout}s")
        return body

    def ingest_status(self, job_id: str) -> Dict[str, Any]:
        """Status of an ingestion job (also visible in the Control Panel)."""
        resp = self._http.post(
            f"{AI_BASE}/ingest_status",
            data={**self._auth_params(), "job_id": job_id},
        )
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise OpensolrError(f"ingest_status: non-JSON response: {resp.text[:200]}") from exc

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
