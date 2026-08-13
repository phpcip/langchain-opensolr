"""Opensolr server-side embeddings for LangChain.

Texts are embedded by Opensolr's GPU-backed multilingual model
(E5-large-instruct, 1024 dimensions). No local model, no extra API keys —
the same credentials as your Opensolr account.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.embeddings import Embeddings

from ._client import OpensolrClient


class OpensolrEmbeddings(Embeddings):
    """Embeddings backed by the Opensolr AI API.

    Example:
        .. code-block:: python

            from langchain_opensolr import OpensolrEmbeddings

            embeddings = OpensolrEmbeddings(
                email="you@example.com",
                api_key="...",
                index="mysite__dense",
            )
            vec = embeddings.embed_query("budget-friendly dining")

    Args:
        email: Opensolr account email.
        api_key: Opensolr API key.
        index: A vector-enabled Opensolr index name. Embedding requests are
            accounted against this index's plan.
        client: Optional pre-configured :class:`OpensolrClient` to reuse
            (e.g. shared with an :class:`OpensolrVectorStore`).
    """

    def __init__(
        self,
        email: str = "",
        api_key: str = "",
        index: str = "",
        client: Optional[OpensolrClient] = None,
    ) -> None:
        if client is None:
            if not (email and api_key):
                raise ValueError("Provide either a client or email + api_key")
            client = OpensolrClient(email, api_key)
        if not index:
            raise ValueError("index is required (embedding is accounted per index)")
        self._client = client
        self._index = index

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._client.batch_embed(self._index, texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed(self._index, text, is_query=True)
