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

from ._client import VECTOR_LOCATIONS, OpensolrClient, OpensolrError, resolve_location
from .embeddings import OpensolrEmbeddings

_HYBRID_MODES = ("union", "keywords_required", "meaning_required", "intersection")

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


def _sanitize_meta_key(key: str) -> str:
    return _META_KEY_RE.sub("_", key.lower()).strip("_")


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

    def _ensure_index(self) -> None:
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
        if version and not version.startswith("9"):
            raise OpensolrError(
                f"Index {self._index!r} runs Solr {version}, but vector search "
                f"requires Solr 9.x. Create the index in one of the vector-enabled "
                f"locations: {sorted(VECTOR_LOCATIONS)}."
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
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        if not texts:
            return []
        self._ensure_index()
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        if not (len(texts) == len(metadatas) == len(ids)):
            raise ValueError("texts, metadatas and ids must have the same length")

        vectors = self._client.batch_embed(self._index, texts)

        docs = []
        for text, meta, doc_id, vector in zip(texts, metadatas, ids, vectors):
            doc: Dict[str, Any] = {
                "id": doc_id,
                self._text_field: text,
                self._vector_field: vector,
                "meta_lc_json": json.dumps(meta, ensure_ascii=False),
            }
            title = meta.get("title") if isinstance(meta, dict) else None
            doc["title"] = str(title) if title else text[:100]
            for key, value in (meta or {}).items():
                if _scalar(value):
                    field = f"meta_{_sanitize_meta_key(key)}"
                    if field not in ("meta_lc_json",):
                        doc[field] = str(value)
            docs.append(doc)

        self._client.solr_update(self._index, docs)
        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        self._ensure_index()
        if ids is None:
            if kwargs.get("delete_all"):
                self._client.solr_update(self._index, {"delete": {"query": "*:*"}})
                return True
            raise ValueError("Provide ids, or delete_all=True to clear the index")
        self._client.solr_update(self._index, {"delete": list(ids)})
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        self._ensure_index()
        joined = " OR ".join(f'"{_escape_fq_value(str(i))}"' for i in ids)
        body = self._client.solr_select(
            self._index, {"q": f"id:({joined})", "rows": len(ids), "fl": "*"}
        )
        found = {d.id: d for d in map(self._doc_from_solr, body["response"]["docs"])}
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
        mode: str = "union",
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return documents most similar to ``query`` with their scores.

        Args:
            query: Natural language query. Embedded server-side.
            k: Number of documents to return.
            filter: Metadata filter — a dict (``{"key": "value"}`` matches the
                ``meta_key`` field), a raw Solr ``fq`` string, or a list of them.
            hybrid: Fuse BM25 (lexical) and kNN (semantic) scores per document
                using Opensolr's ``{!hybrid}`` query parser instead of pure kNN.
            mode: Hybrid mode — ``union`` (default), ``keywords_required``,
                ``meaning_required`` or ``intersection``.
            alpha: Hybrid semantic↔lexical balance, 0 = all semantic,
                1 = all lexical.
        """
        self._ensure_index()
        vector = self._client.embed(self._index, query, is_query=True)
        knn = self._knn_query(vector, max(k, 10))

        params: Dict[str, Any] = {
            "rows": k,
            "fl": "*,score",
        }
        if hybrid:
            if mode not in _HYBRID_MODES:
                raise ValueError(f"mode must be one of {_HYBRID_MODES}, got {mode!r}")
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
