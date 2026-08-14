"""End-to-end integration test against a live Opensolr vector index.

Writes go through the async Data Ingestion API (queue processed every minute),
so this suite takes ~1-2 minutes. Requires env:
OPENSOLR_EMAIL, OPENSOLR_API_KEY, OPENSOLR_INDEX.
"""

import os
import time
import uuid

import pytest

from langchain_opensolr import OpensolrEmbeddings, OpensolrVectorStore

EMAIL = os.environ.get("OPENSOLR_EMAIL")
API_KEY = os.environ.get("OPENSOLR_API_KEY")
INDEX = os.environ.get("OPENSOLR_INDEX")

pytestmark = pytest.mark.skipif(
    not (EMAIL and API_KEY and INDEX),
    reason="OPENSOLR_EMAIL / OPENSOLR_API_KEY / OPENSOLR_INDEX not set",
)

RUN = uuid.uuid4().hex[:8]
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
IDS = [f"e2e_{RUN}_{i}" for i in range(1, 5)]


@pytest.fixture(scope="module")
def store():
    vs = OpensolrVectorStore(index=INDEX, email=EMAIL, api_key=API_KEY)
    vs.delete(delete_all=True)
    # One ingestion round for the whole suite (async queue, ~1 min).
    solr_ids = vs.add_texts(TEXTS, metadatas=METAS, ids=IDS, wait=True)
    assert len(solr_ids) == 4 and all(len(i) == 32 for i in solr_ids)
    vs._solr_ids = solr_ids
    time.sleep(2)
    yield vs
    vs.delete(delete_all=True)


def test_embeddings_shape():
    emb = OpensolrEmbeddings(email=EMAIL, api_key=API_KEY, index=INDEX)
    q = emb.embed_query("hello world")
    assert len(q) == 1024


def test_ingested_and_semantic(store):
    docs = store.similarity_search("sleepy pets", k=1)
    assert len(docs) == 1
    assert "Cats" in docs[0].page_content
    assert docs[0].metadata.get("category") == "animals"


def test_search_with_score(store):
    results = store.similarity_search_with_score("open source search engines", k=2)
    assert len(results) == 2
    assert results[0][1] > 0


def test_hybrid(store):
    docs = store.similarity_search("BM25 lexical", k=2, hybrid=True, alpha=0.5)
    assert len(docs) >= 1
    assert "Hybrid" in docs[0].page_content or "Solr" in docs[0].page_content


def test_lexical(store):
    # Pure keyword search — no embedding call, zero AI quota.
    docs = store.similarity_search("flour recipe egg", k=2, lexical=True)
    assert len(docs) >= 1
    assert "flour" in docs[0].page_content


def test_filter(store):
    docs = store.similarity_search("anything at all", k=5, filter={"category": "cooking"})
    assert len(docs) == 1
    assert "flour" in docs[0].page_content


def test_get_by_original_ids(store):
    docs = store.get_by_ids([IDS[1], IDS[3]])
    assert len(docs) == 2
    assert docs[0].metadata.get("category") == "animals"


def test_get_by_solr_ids(store):
    docs = store.get_by_ids(store._solr_ids[:2])
    assert len(docs) == 2


def test_delete_by_original_id(store):
    store.delete([IDS[3]])
    time.sleep(1)
    assert store.get_by_ids([IDS[3]]) == []


def test_delete_by_query(store):
    store.delete(query=f'meta_category:"animals"')
    time.sleep(1)
    assert store.get_by_ids([IDS[1]]) == []
