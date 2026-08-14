"""Offline unit tests — no network."""

import pytest

from langchain_opensolr._client import VECTOR_LOCATIONS, resolve_location
from langchain_opensolr.vectorstores import (
    OpensolrVectorStore,
    _escape_fq_value,
    _sanitize_meta_key,
)


def _store():
    return OpensolrVectorStore(index="t__dense", email="a@b.c", api_key="k")


def test_location_aliases():
    assert set(VECTOR_LOCATIONS) == {"us", "de", "fi"}
    assert resolve_location("us") == "CHICAGO-96"
    assert resolve_location("DE") == "DE-SOLR-9"
    # Unknown values pass through — validated live against vector_regions
    assert resolve_location("TOKYO-9") == "TOKYO-9"


def test_meta_key_sanitization():
    assert _sanitize_meta_key("My Key!") == "my_key"
    assert _sanitize_meta_key("source.url") == "source_url"


def test_fq_escaping():
    assert _escape_fq_value('a"b\\c') == 'a\\"b\\\\c'


def test_filter_to_fq():
    s = _store()
    assert s._filter_to_fq(None) == []
    assert s._filter_to_fq({"category": "docs"}) == ['meta_category:"docs"']
    assert s._filter_to_fq({"tag": ["a", "b"]}) == ['meta_tag:("a" OR "b")']
    assert s._filter_to_fq('raw:query') == ["raw:query"]


def test_knn_query_shape():
    s = _store()
    q = s._knn_query([0.1, 0.2], 5)
    assert q == "{!knn f=embeddings topK=5}[0.1,0.2]"


def test_missing_credentials():
    with pytest.raises(ValueError):
        OpensolrVectorStore(index="t")
