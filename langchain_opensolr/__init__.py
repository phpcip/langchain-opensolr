"""LangChain integration for Opensolr — managed Apache Solr with server-side
embeddings and native hybrid (BM25 + kNN) search."""

from langchain_opensolr._client import OpensolrClient, OpensolrError
from langchain_opensolr.embeddings import OpensolrEmbeddings
from langchain_opensolr.vectorstores import OpensolrVectorStore

__all__ = [
    "OpensolrClient",
    "OpensolrError",
    "OpensolrEmbeddings",
    "OpensolrVectorStore",
]
