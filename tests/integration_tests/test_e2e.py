"""End-to-end integration test against a live Opensolr vector index.

Requires env vars: OPENSOLR_EMAIL, OPENSOLR_API_KEY, OPENSOLR_INDEX.
Run: OPENSOLR_EMAIL=... OPENSOLR_API_KEY=... OPENSOLR_INDEX=... pytest tests/integration_tests -v
"""

import os
import time

import pytest

from langchain_opensolr import OpensolrEmbeddings, OpensolrVectorStore

EMAIL = os.environ.get("OPENSOLR_EMAIL")
API_KEY = os.environ.get("OPENSOLR_API_KEY")
INDEX = os.environ.get("OPENSOLR_INDEX")

pytestmark = pytest.mark.skipif(
    not (EMAIL and API_KEY and INDEX),
    reason="OPENSOLR_EMAIL / OPENSOLR_API_KEY / OPENSOLR_INDEX not set",
)

TEXTS = [
    "Apache Solr is an open-source enterprise search platform",
    "Cats spend most of the day sleeping in warm places",
    "Hybrid search combines lexical BM25 with semantic vector retrieval",
    "The recipe calls for two cups of flour and one egg",
]
METAS = [
    {"category": "search", "rank": 1},
    {"category": "animals", "rank": 2},
    {"category": "search", "rank": 3},
    {"category": "cooking", "rank": 4},
]
IDS = ["e2e_1", "e2e_2", "e2e_3", "e2e_4"]


@pytest.fixture(scope="module")
def store():
    vs = OpensolrVectorStore(index=INDEX, email=EMAIL, api_key=API_KEY)
    vs.delete(delete_all=True)
    yield vs
    vs.delete(delete_all=True)


def test_embeddings_shape():
    emb = OpensolrEmbeddings(email=EMAIL, api_key=API_KEY, index=INDEX)
    q = emb.embed_query("hello world")
    assert len(q) == 1024
    docs = emb.embed_documents(["one", "two"])
    assert len(docs) == 2 and all(len(v) == 1024 for v in docs)


def test_add_and_search(store):
    ids = store.add_texts(TEXTS, metadatas=METAS, ids=IDS)
    assert ids == IDS
    time.sleep(1)

    docs = store.similarity_search("open source search engines", k=2)
    assert len(docs) == 2
    assert "Solr" in docs[0].page_content or "Hybrid" in docs[0].page_content
    assert docs[0].metadata.get("category") == "search"


def test_search_with_score(store):
    results = store.similarity_search_with_score("sleepy pets", k=2)
    assert len(results) == 2
    doc, score = results[0]
    assert "Cats" in doc.page_content
    assert score > 0


def test_filter(store):
    docs = store.similarity_search("search technology", k=4, filter={"category": "cooking"})
    assert len(docs) == 1
    assert "flour" in docs[0].page_content


def test_hybrid(store):
    docs = store.similarity_search("BM25 lexical", k=2, hybrid=True, alpha=0.5)
    assert len(docs) >= 1
    assert "Hybrid" in docs[0].page_content or "Solr" in docs[0].page_content


def test_get_by_ids(store):
    docs = store.get_by_ids(["e2e_2", "e2e_4"])
    assert [d.id for d in docs] == ["e2e_2", "e2e_4"]
    assert docs[0].metadata == {"category": "animals", "rank": 2}


def test_delete(store):
    store.delete(["e2e_4"])
    time.sleep(1)
    docs = store.get_by_ids(["e2e_4"])
    assert docs == []
