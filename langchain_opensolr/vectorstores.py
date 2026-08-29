"""Opensolr vector store for LangChain.

A :class:`~langchain_core.vectorstores.VectorStore` backed by a managed,
vector-enabled Opensolr index (Apache Solr 9.x, ``knn_vector`` 1024-dim,
cosine). Embedding happens **server-side** on Opensolr's GPU infrastructure —
no local embedding model or third-party API key is needed.

Highlights:

- Zero-config constructor: ``OpensolrVectorStore(index=..., email=..., api_key=...)``.
  Host, port and HTTP auth of the underlying Solr core are resolved
  automatically through the Opensolr management API.
- ``hybrid=True`` search uses Opensolr's native ``{!hybrid}`` query parser,
  fusing BM25 and kNN per document with a tunable ``alpha`` balance and four
  modes: ``union``, ``keywords_required``, ``meaning_required``, ``intersection``.
- Metadata round-trips losslessly (stored as JSON alongside filterable
  ``meta_*`` string fields).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ._client import (
    HYBRID_MODES,
    VECTOR_LOCATIONS,
    OpensolrClient,
    OpensolrError,
    apply_fresh_bias,
    resolve_location,
)
from .embeddings import OpensolrEmbeddings

# Re-exported from the client so there is ONE list. The client validates before it spends an
# embedding call; this name is kept because the wrapper's own signatures reference it.
_HYBRID_MODES = HYBRID_MODES

#: Solr fields managed by this integration or by the Opensolr schema that
#: should not leak into Document.metadata.
_INTERNAL_FIELDS = {
    "_version_",
    "_root_",
    "score",
    "embeddings",
    "meta_lc_json",
}

_META_KEY_RE = re.compile(r"[^a-z0-9_]+")

#: Metadata keys that map to first-class ingestion fields, not meta_* copies.
_RESERVED_META = {"uri", "url", "title", "description", "timestamp", "rtf", "author", "category", "content_type", "og_image"}


def _sanitize_meta_key(key: str) -> str:
    return _META_KEY_RE.sub("_", key.lower()).strip("_")


def _uri_for(index: str, doc_id: str, metadata: Optional[dict]) -> str:
    """Document identity = its URI (Data Ingestion contract: id = md5(uri)).

    Uses metadata['uri'] / metadata['url'] when it is a real http(s) URL,
    otherwise synthesizes a deterministic one from the caller's id.
    """
    from urllib.parse import quote

    meta = metadata or {}
    uri = meta.get("uri") or meta.get("url")
    if not (isinstance(uri, str) and uri.startswith(("http://", "https://"))):
        uri = f"https://ingest.opensolr.com/{index}/{quote(str(doc_id), safe='')}"
    return uri.rstrip("/")


def _ingest_doc(index: str, text: str, metadata: Optional[dict], doc_id: str) -> Dict[str, Any]:
    """Build one Data Ingestion API document (uri/title/description/text
    required; custom metadata as meta_* fields; lossless meta_lc_json)."""
    import hashlib

    meta = dict(metadata or {})
    uri = _uri_for(index, doc_id, meta)
    text = text or " "
    doc: Dict[str, Any] = {
        "uri": uri,
        "title": str(meta.get("title") or text[:100] or uri)[:250],
        "description": str(meta.get("description") or text[:200]),
        "text": text,
        "meta_ext_id": str(doc_id),
        "meta_lc_json": json.dumps(meta, ensure_ascii=False),
    }
    if meta.get("rtf"):
        doc["rtf"] = True
    if meta.get("timestamp"):
        doc["timestamp"] = meta["timestamp"]
    for opt in ("author", "category", "content_type", "og_image"):
        if meta.get(opt):
            doc[opt] = str(meta[opt])
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) and key not in ("rtf", "uri", "url"):
            field = f"meta_{_sanitize_meta_key(str(key))}"
            if field not in ("meta_lc_json", "meta_ext_id"):
                doc[field] = str(value)
    doc["_solr_id"] = hashlib.md5(uri.encode()).hexdigest()
    return doc


def _escape_fq_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


class OpensolrVectorStore(VectorStore):
    """Managed hybrid vector store on Opensolr.

    Example:
        .. code-block:: python

            from langchain_opensolr import OpensolrVectorStore

            vs = OpensolrVectorStore(
                index="mysite__dense",
                email="you@example.com",
                api_key="...",
            )
            vs.add_texts(["Solr is a search platform", "Cats sleep a lot"])
            docs = vs.similarity_search("search engines", k=2)

            # or run it as-is against the public demo account (see Note below)
            demo = OpensolrVectorStore(
                index="mcp_demo_d1__dense",
                email="mcp@opensolr.com",
                api_key="420b8b23e7b12dc8ab838932145a5065",
            )
            docs = demo.similarity_search("interest rate decision", k=3)

            # hybrid BM25 + kNN with filters
            docs = vs.similarity_search(
                "search engines", k=2, hybrid=True, filter={"category": "docs"}
            )

    Args:
        index: Name of the vector-enabled Opensolr index. Vector indexes live
            on Opensolr's Solr 9.x environments (locations: ``us``, ``de``, ``fi``).
        email: Opensolr account email.
        api_key: Opensolr API key.
        client: Optional pre-configured :class:`OpensolrClient` (overrides
            email/api_key).
        location: Where to create the index if ``create_if_missing`` is set:
            ``us`` (Chicago), ``de`` (Germany) or ``fi`` (Finland) — the
            Opensolr environments with vector-enabled servers.
        create_if_missing: Create the index automatically on first use.
        text_field: Solr field holding page content (default ``text``).
        vector_field: Solr ``knn_vector`` field (default ``embeddings``).

    Note:
        The demo account above is shared publicly — its preloaded
        ``mcp_demo_d1__dense`` index holds 300 news articles, other people can
        change or delete what you create, anything created there is deleted after
        3 days, automatically, and limits are per index and small on purpose:
        200 MB bandwidth, 50 MB disk. For a private index that persists:
        https://opensolr.com/register (free 15-day trial, no card).
    """

    def __init__(
        self,
        index: str,
        email: str = "",
        api_key: str = "",
        client: Optional[OpensolrClient] = None,
        location: str = "us",
        create_if_missing: bool = False,
        text_field: str = "text",
        vector_field: str = "embeddings",
    ) -> None:
        if client is None:
            if not (email and api_key):
                raise ValueError("Provide either a client or email + api_key")
            client = OpensolrClient(email, api_key)
        # No hard validation here: the authoritative location list is fetched
        # live in create_index (vector_regions endpoint), so newly deployed
        # vector regions work without upgrading this package.
        self._client = client
        self._index = index
        self._location = location
        self._create_if_missing = create_if_missing
        self._text_field = text_field
        self._vector_field = vector_field
        self._checked = False

    # ------------------------------------------------------------------ #
    # plumbing                                                           #
    # ------------------------------------------------------------------ #

    @property
    def embeddings(self) -> Embeddings:
        """Server-side embeddings bound to this index."""
        return OpensolrEmbeddings(client=self._client, index=self._index)

    def _ensure_index(self, check_vector: bool = True) -> None:
        if self._checked:
            return
        try:
            info = self._client.get_core_info(self._index)
        except OpensolrError:
            if not self._create_if_missing:
                raise
            self._client.create_index(self._index, self._location)
            info = None
            for _ in range(5):
                time.sleep(2)
                try:
                    info = self._client.get_core_info(self._index, refresh=True)
                    break
                except OpensolrError:
                    continue
            if info is None:
                raise
        version = str(info.get("solr_version", ""))
        if check_vector and version and not version.startswith("9"):
            raise OpensolrError(
                f"Index {self._index!r} runs Solr {version}, but vector search "
                f"requires Solr 9.x. Use lexical=True for keyword-only search on "
                f"this index, or create a vector index in: {sorted(VECTOR_LOCATIONS)}."
            )
        self._checked = True

    def _doc_from_solr(self, solr_doc: Dict[str, Any]) -> Document:
        def _flat(v: Any) -> Any:
            if isinstance(v, list):
                return v[0] if len(v) == 1 else v
            return v

        content = _flat(solr_doc.get(self._text_field, "")) or ""
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)

        metadata: Dict[str, Any] = {}
        raw_json = _flat(solr_doc.get("meta_lc_json"))
        if raw_json:
            try:
                metadata = json.loads(raw_json)
            except (TypeError, json.JSONDecodeError):
                metadata = {}
        if not metadata:
            for key, value in solr_doc.items():
                if key.startswith("meta_") and key not in _INTERNAL_FIELDS:
                    metadata[key[5:]] = _flat(value)
        return Document(
            id=str(_flat(solr_doc.get("id", ""))),
            page_content=str(content),
            metadata=metadata,
        )

    def _filter_to_fq(self, filter: Any) -> List[str]:
        if filter is None:
            return []
        if isinstance(filter, str):
            return [filter]
        if isinstance(filter, list):
            return [str(f) for f in filter]
        if isinstance(filter, dict):
            fq = []
            for key, value in filter.items():
                field = f"meta_{_sanitize_meta_key(key)}"
                if isinstance(value, list):
                    joined = " OR ".join(f'"{_escape_fq_value(str(v))}"' for v in value)
                    fq.append(f"{field}:({joined})")
                else:
                    fq.append(f'{field}:"{_escape_fq_value(str(value))}"')
            return fq
        raise ValueError(f"Unsupported filter type: {type(filter)}")

    # ------------------------------------------------------------------ #
    # write path                                                         #
    # ------------------------------------------------------------------ #

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        wait: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """Queue texts through the Opensolr Data Ingestion API.

        Ingestion is ASYNC: embeddings and all derived fields are computed
        server-side and documents become searchable within ~1 minute (the
        queue is processed every minute; progress is visible in the Control
        Panel and via ``client.ingest_status``). Pass ``wait=True`` to block
        until the job completes. Returns the Solr document ids (md5 of each
        document's URI, per the ingestion contract).
        """
        texts = list(texts)
        if not texts:
            return []
        self._ensure_index(check_vector=False)
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        if not (len(texts) == len(metadatas) == len(ids)):
            raise ValueError("texts, metadatas and ids must have the same length")

        docs = [
            _ingest_doc(self._index, text, meta, doc_id)
            for text, meta, doc_id in zip(texts, metadatas, ids)
        ]
        solr_ids = [d.pop("_solr_id") for d in docs]
        for i in range(0, len(docs), 50):
            self._client.ingest(self._index, docs[i : i + 50], wait=wait)
        return solr_ids

    def delete(
        self,
        ids: Optional[List[str]] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete by ids (Solr ids or your original ids) or by a raw Solr query."""
        self._ensure_index(check_vector=False)
        if query:
            self._client.solr_update(self._index, {"delete": {"query": query}})
            return True
        if ids is None:
            if kwargs.get("delete_all"):
                self._client.solr_update(self._index, {"delete": {"query": "*:*"}})
                return True
            raise ValueError("Provide ids, query=..., or delete_all=True")
        joined = " OR ".join(f'"{_escape_fq_value(str(i))}"' for i in ids)
        self._client.solr_update(
            self._index,
            {"delete": {"query": f"id:({joined}) OR meta_ext_id:({joined})"}},
        )
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        self._ensure_index(check_vector=False)
        joined = " OR ".join(f'"{_escape_fq_value(str(i))}"' for i in ids)
        body = self._client.solr_select(
            self._index,
            {"q": f"id:({joined}) OR meta_ext_id:({joined})", "rows": max(len(ids), 10), "fl": "*"},
        )
        docs = [self._doc_from_solr(d) for d in body["response"]["docs"]]
        found = {d.id: d for d in docs}
        for d, raw in zip(docs, body["response"]["docs"]):
            ext = raw.get("meta_ext_id")
            ext = ext[0] if isinstance(ext, list) else ext
            if ext:
                found.setdefault(str(ext), d)
        return [found[i] for i in ids if i in found]

    # ------------------------------------------------------------------ #
    # read path                                                          #
    # ------------------------------------------------------------------ #

    def _knn_query(self, vector: List[float], k: int) -> str:
        compact = json.dumps(vector, separators=(",", ":"))
        return f"{{!knn f={self._vector_field} topK={k}}}{compact}"

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Any] = None,
        hybrid: bool = False,
        lexical: bool = False,
        mode: str = "union",
        alpha: float = 0.5,
        fresh_bias: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return documents most similar to ``query`` with their scores.

        Args:
            query: Natural language query. Embedded server-side (unless lexical).
            k: Number of documents to return.
            filter: Metadata filter — a dict (``{"key": "value"}`` matches the
                ``meta_key`` field), a raw Solr ``fq`` string, or a list of them.
            hybrid: Fuse BM25 (lexical) and kNN (semantic) scores per document
                using Opensolr's ``{!hybrid}`` query parser instead of pure kNN.
            lexical: Pure keyword (edismax) search — no embedding call, zero AI
                quota, and works on ANY Opensolr index, including non-vector
                and older Solr versions.
            mode: Hybrid mode — ``union`` (default), ``keywords_required``,
                ``meaning_required`` or ``intersection``.
            alpha: Hybrid semantic↔lexical balance, 0 = all semantic,
                1 = all lexical.
            fresh_bias: Bias the ranking toward recent documents by
                multiplying each score by a recency curve on
                ``creation_date``. It re-orders and never filters — the hit
                count is unchanged, nothing old becomes unreachable, and a
                document with no ``creation_date`` is simply left unboosted.
                Works on all three shapes above (kNN, hybrid, lexical). Off
                by default.
        """
        self._ensure_index(check_vector=not lexical)

        params: Dict[str, Any] = {
            "rows": k,
            "fl": "*,score",
        }
        if lexical:
            clean = query.replace("{", " ").replace("}", " ").replace('"', " ")
            params["q"] = f'{{!edismax qf="title^100 description^20 {self._text_field}^1"}}{clean}'
            # Fresh Results Bias on the lexical path. Wrapped rather than set as an
            # edismax `bf`: edismax is invoked here through local params inside q, not
            # as the request's defType, so a top-level bf is not reliably its own.
            if fresh_bias:
                apply_fresh_bias(params)
            for fq in self._filter_to_fq(filter):
                params.setdefault("fq", [])
                params["fq"].append(fq)
            body = self._client.solr_select(self._index, params)
            return [
                (self._doc_from_solr(d), float(d.get("score", 0.0)))
                for d in body["response"]["docs"]
            ]

        # Validate before embedding: the check is local and free, the embedding is a billed GPU
        # round-trip. Rejecting the caller's typo after paying for it charged them for our own
        # argument validation (2026-08-29).
        if hybrid and mode not in _HYBRID_MODES:
            raise ValueError(f"mode must be one of {_HYBRID_MODES}, got {mode!r}")

        vector = self._client.embed(self._index, query, is_query=True)
        knn = self._knn_query(vector, max(k, 10))
        if hybrid:
            clean = query.replace("{", " ").replace("}", " ").replace('"', " ")
            params["q"] = (
                f"{{!hybrid lexical=$lexicalRaw vector=$vectorQuery "
                f"mode={mode} alpha={alpha} topN={max(k, 10)}}}"
            )
            params["lexicalRaw"] = (
                f'{{!edismax qf="title^100 {self._text_field}^1"}}{clean}'
            )
            params["vectorQuery"] = knn
        else:
            params["q"] = knn

        # Fresh Results Bias wraps whichever shape was just built — fused {!hybrid}
        # or bare {!knn} — so the recency multiplier reaches every candidate,
        # including the vector-only ones that an edismax bf would never see.
        if fresh_bias:
            apply_fresh_bias(params)

        for i, fq in enumerate(self._filter_to_fq(filter)):
            params.setdefault("fq", [])
            params["fq"].append(fq)

        body = self._client.solr_select(self._index, params)
        docs = body["response"]["docs"]
        return [
            (self._doc_from_solr(d), float(d.get("score", 0.0)))
            for d in docs
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return [
            doc
            for doc, _ in self.similarity_search_with_score(
                query, k=k, filter=filter, **kwargs
            )
        ]

    def ai_answer(
        self,
        query: str,
        filter: Optional[Any] = None,
        # 4 documents is the platform's measured context size (OpensolrClient.RAG_DOCS);
        # this used to pass 3, which quietly overrode the client default with a smaller one.
        rag_docs: int = 4,
        rag_words: int = 1500,
        instruction: Optional[str] = None,
        tuning: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Grounded RAG answer generated only from this index's content.

        Two-step pattern: hybrid (BM25 + kNN) retrieval picks the top
        ``rag_docs`` hits (first ``rag_words`` words of text each), whose
        title/description/text become the LLM context — the same pipeline as
        Opensolr's hosted search UI. Pass ``instruction`` to fully control
        the prompt (e.g. "Answer in German, cite the sources you used").
        Retrieval uses the platform's tuned pipeline: your index's saved
        Search Tuning (Control Panel) applies automatically; ``tuning``
        overrides any knob per call. The list below is the whole set, not a
        sample: an abbreviated one reads as everything that is supported, and
        ``freshness_boost`` was invisible to callers because of it.
        ``fw_title``, ``fw_description``, ``fw_uri``, ``fw_text``,
        ``fw_text_t``, ``lexical_weight``, ``vector_weight``, ``vector_topk``,
        ``search_mode`` (union / keywords_required / meaning_required /
        intersection), ``quality_boost``, ``min_score``, ``freshness_boost``,
        ``fresh_bias``, ``lexical_norm_k``, ``mm`` (flexible / balanced /
        strict or raw Solr mm syntax). ``freshness_boost`` and ``fresh_bias``
        are different knobs despite the names: the first is a hard window in
        DAYS that filters older documents out, the second only re-orders,
        multiplying each score by a recency curve on ``creation_date`` so
        recent documents win ties while nothing becomes unreachable.
        Returns plain text.
        """
        self._ensure_index()
        fqs = self._filter_to_fq(filter)
        fq = " AND ".join(f"({f})" for f in fqs) if fqs else None
        return self._client.ai_summary(
            self._index, query, filter_query=fq,
            rag_docs=rag_docs, rag_words=rag_words, instruction=instruction,
            tuning=tuning,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Document]:
        self._ensure_index()
        params: Dict[str, Any] = {
            "q": self._knn_query(embedding, max(k, 10)),
            "rows": k,
            "fl": "*,score",
        }
        fq = self._filter_to_fq(filter)
        if fq:
            params["fq"] = fq
        body = self._client.solr_select(self._index, params)
        return [self._doc_from_solr(d) for d in body["response"]["docs"]]

    # ------------------------------------------------------------------ #
    # constructors                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        *,
        index: str = "",
        email: str = "",
        api_key: str = "",
        location: str = "us",
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "OpensolrVectorStore":
        """Build a store from texts.

        ``embedding`` is accepted for interface compatibility but ignored —
        Opensolr embeds server-side with its own multilingual model.
        """
        if not index:
            raise ValueError("index is required")
        store = cls(
            index=index,
            email=email,
            api_key=api_key,
            location=location,
            create_if_missing=True,
            **kwargs,
        )
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
