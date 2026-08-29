#!/usr/bin/env python3
"""Live production test for langchain-opensolr.

Run it with the package's own virtualenv, from the package root::

    .venv/bin/python test_live.py

What it does
------------
Exercises EVERY public method of ``langchain_opensolr`` against the LIVE
Opensolr platform (opensolr.com + api.opensolr.com) with the public MCP demo
account, asserting VALUES rather than "no exception was raised".

Safety
------
* The seeded demo index (``mcp_demo_d1__dense``, 300 real news articles) is
  READ-ONLY here — nothing in this file writes to it or deletes from it.
* Everything that writes goes to two temporary vector indexes named
  ``mcp_t_*__dense``, created at the start and deleted in a ``finally`` block,
  so a crash, a failed assertion or a Ctrl-C still cleans up. Names carry a
  random suffix, so the script is safe to run twice at the same time.
* Every request to opensolr.com / api.opensolr.com is paced through a rolling
  window (default 25/minute) so a full run stays under the documented 30/min
  API rate limit. Direct Solr traffic goes to the index's own cluster host and
  is not rate limited, so it is not paced.

Exit code is 0 only when every check passed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections import deque
from typing import Any, Callable, Dict, List
from urllib.parse import quote

import httpx

from langchain_core.documents import Document

from langchain_opensolr import (
    OpensolrClient,
    OpensolrEmbeddings,
    OpensolrError,
    OpensolrVectorStore,
)
from langchain_opensolr._client import (
    BATCH_EMBED_MAX,
    DEFAULT_RAG_DOCS,
    DEFAULT_RAG_WORDS,
    DOC_FENCE,
    FRESH_BIAS_FUNCTION,
    VECTOR_LOCATIONS,
    apply_fresh_bias,
    build_context,
    build_instruction,
    resolve_location,
)

# --------------------------------------------------------------------------- #
# configuration                                                                #
# --------------------------------------------------------------------------- #

EMAIL = os.environ.get("OPENSOLR_EMAIL", "mcp@opensolr.com")
API_KEY = os.environ.get("OPENSOLR_API_KEY", "420b8b23e7b12dc8ab838932145a5065")

#: Seeded, read-only. 300 crawled news articles, vector-enabled, Solr 9.6.
DEMO = "mcp_demo_d1__dense"

#: Temp indexes. The __dense suffix is what marks an index vector-enabled on
#: the platform, so it is mandatory. The random suffix keeps concurrent runs
#: from colliding and makes every run independently re-runnable.
RUN = uuid.uuid4().hex[:6]
TMP_CLIENT = f"mcp_t_lc{RUN}__dense"      # written to by the client + vectorstore
TMP_FROMTEXTS = f"mcp_t_lf{RUN}__dense"   # created by OpensolrVectorStore.from_texts
LOCATION = "fi"                            # FINLAND9 — where the demo index lives

#: Documented API rate limit is 30 requests/minute; stay under it.
MAX_PER_MIN = int(os.environ.get("OPENSOLR_MAX_PER_MIN", "25"))

#: Ingestion is async: the queue is drained by a once-a-minute cron, so a
#: document needs one queue tick plus embedding time to become searchable.
INGEST_TIMEOUT = float(os.environ.get("OPENSOLR_INGEST_TIMEOUT", "120"))
POLL_EVERY = 8.0

QUERY = "football transfer news"
RAG_QUESTION = "Which football transfer deals are reported?"

# --------------------------------------------------------------------------- #
# request pacing                                                               #
# --------------------------------------------------------------------------- #
# Wraps httpx.Client.post for the whole process (the package builds its own
# httpx client internally, so patching the class is the only place that catches
# every call). Only management/AI API traffic is paced — direct Solr traffic
# goes to *.solrcluster.com and is not rate limited.

_calls: "deque[float]" = deque()
_api_requests = 0
_solr_requests = 0
_original_post = httpx.Client.post


def _pace() -> None:
    """Block until another API request fits inside the rolling minute."""
    while True:
        now = time.monotonic()
        while _calls and now - _calls[0] >= 60.0:
            _calls.popleft()
        if len(_calls) < MAX_PER_MIN:
            _calls.append(now)
            return
        time.sleep(max(0.1, 60.0 - (now - _calls[0])) + 0.05)


def _paced_post(self: httpx.Client, url: Any, *args: Any, **kwargs: Any) -> Any:
    global _api_requests, _solr_requests
    if "opensolr.com" in str(url):
        _pace()
        _api_requests += 1
    else:
        _solr_requests += 1
    return _original_post(self, url, *args, **kwargs)


httpx.Client.post = _paced_post  # type: ignore[method-assign]

# --------------------------------------------------------------------------- #
# tiny test harness                                                            #
# --------------------------------------------------------------------------- #

PASSED: List[str] = []
FAILED: List[str] = []
STATE: Dict[str, Any] = {}


def check(name: str, fn: Callable[[], str]) -> None:
    """Run one check. Prints exactly one line, records pass/fail."""
    try:
        detail = fn() or ""
    except Exception as exc:  # noqa: BLE001 — every failure is a reportable result
        msg = f"{type(exc).__name__}: {exc}"
        FAILED.append(f"{name} — {msg}")
        print(f"✘ {name} — {_one_line(msg)}", flush=True)
        return
    PASSED.append(name)
    print(f"✔ {name}" + (f" — {_one_line(detail)}" if detail else ""), flush=True)


def _one_line(text: str, limit: int = 170) -> str:
    """Collapse a detail string onto a single output line."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def require(condition: Any, message: str) -> None:
    """Assert with a message that says what the expected VALUE was."""
    if not condition:
        raise AssertionError(message)


def section(title: str) -> None:
    print(f"\n--- {title} ---", flush=True)


def norm(vector: List[float]) -> float:
    """L2 norm — the Opensolr embedding model returns unit vectors."""
    return math.sqrt(sum(x * x for x in vector))


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two embeddings."""
    return sum(x * y for x, y in zip(a, b)) / (norm(a) * norm(b))


def md5(text: str) -> str:
    """Opensolr's document identity: id = md5(uri)."""
    return hashlib.md5(text.encode()).hexdigest()


def wait_until(fn: Callable[[], Any], what: str, timeout: float = INGEST_TIMEOUT) -> Any:
    """Poll ``fn`` until it returns something truthy. Never sleeps a fixed time."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = fn()
        if last:
            return last
        time.sleep(POLL_EVERY)
    raise TimeoutError(f"gave up after {timeout:.0f}s waiting for {what} (last seen: {last!r})")


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #

#: Documents for the direct client.ingest() call. uri/title/description/text
#: are all mandatory on the Data Ingestion API; id is md5(uri).
INGEST_DOCS = [
    {
        "uri": f"https://example.com/langchain-live/{RUN}/solr-vector-search",
        "title": "Opensolr live test: vector search on Apache Solr",
        "description": "A test document about dense vector retrieval on Solr.",
        "text": (
            "Opensolr runs managed Apache Solr with knn_vector fields and a native hybrid "
            "query parser. This document exists only so the langchain-opensolr live test "
            "can prove that the Data Ingestion API queued it, embedded it server side, and "
            "made it searchable through the same index. Dense retrieval fuses BM25 and kNN."
        ),
        "timestamp": int(time.time()),
    },
    {
        "uri": f"https://example.com/langchain-live/{RUN}/marmalade-recipe",
        "title": "Opensolr live test: seville orange marmalade",
        "description": "An unrelated document about making marmalade, used as a negative control.",
        "text": (
            "Seville oranges are boiled whole until soft, then sliced and simmered with sugar "
            "until the setting point is reached. This document is deliberately unrelated to "
            "search engines so the live test can tell a real relevance ranking from a fake one. "
            "Marmalade keeps for a year in sterilised jars."
        ),
        "timestamp": int(time.time()),
    },
]

#: Texts pushed through the LangChain vectorstore, with metadata that must
#: round-trip losslessly through meta_lc_json.
LC_TEXTS = [
    "LangChain retrievers hand documents to a language model. This text was written by the "
    "langchain-opensolr live test and indexed through the Opensolr vector store.",
    "Hybrid search fuses lexical BM25 scoring with dense kNN similarity per document, which "
    "is what the Opensolr hybrid query parser does inside Solr itself.",
    "Norwegian fjords were carved by glaciers during the last ice age and have very steep "
    "walls, which has nothing whatsoever to do with search infrastructure.",
]
LC_IDS = [f"lc-{RUN}-{i}" for i in range(len(LC_TEXTS))]
LC_METAS = [
    {"title": f"LC live doc {i}", "source": "langchain-opensolr-live-test", "batch": RUN, "seq": i}
    for i in range(len(LC_TEXTS))
]


# --------------------------------------------------------------------------- #
# phase 0 — pure module-level builders (no network at all)                     #
# --------------------------------------------------------------------------- #

def phase_pure() -> None:
    section("module-level builders (pure, no network)")

    def t_fresh_bias() -> str:
        inner = '{!knn f=embeddings topK=10}[0.1,0.2]'
        params = {"q": inner, "rows": 3}
        out = apply_fresh_bias(params)
        require(out is params, "apply_fresh_bias must mutate and return the same dict")
        require(params["q"] == "{!boost b=$freshBias v=$freshBiasInner}",
                f"q must be the boost wrapper, got {params['q']!r}")
        require(params["freshBiasInner"] == inner, "inner query must be preserved verbatim")
        require(params["freshBias"] == FRESH_BIAS_FUNCTION, "boost function must be the shipped curve")
        require(params["rows"] == 3, "other params must be untouched")
        return "q wrapped in {!boost}, inner query moved to $freshBiasInner verbatim"

    check("apply_fresh_bias wraps q in {!boost} and keeps the inner query intact", t_fresh_bias)

    docs = [
        {"id": "a", "title": "Alpha", "description": "Alpha desc",
         "text": "one two three four five six seven.", "score": 1.0},
        {"id": "b", "title": "Beta", "description": "",
         "text": "body of beta.", "score": 0.9,
         "text_t": "structured beta payload long enough to clear the fifty byte husk floor."},
        {"id": "c", "title": "Weak", "description": "", "text": "body c.", "score": 0.1},
        {"id": "d", "title": "Delta", "description": "", "text": "body d.", "score": 0.8},
    ]
    hl = {"a": {"text": ["<em>alpha</em> matched <b class='x'>fragment</b> here"]}}

    def t_build_context() -> str:
        ctx = build_context(docs, hl, top_n=3, max_words=5)
        require(ctx.count(DOC_FENCE) == 2, f"expected 2 kept documents, got {ctx.count(DOC_FENCE)}")
        require("Weak" not in ctx, "a hit scoring below half the top score must be dropped")
        require("Delta" not in ctx, "top_n=3 must never reach the 4th hit")
        require("<em>" not in ctx and "<b" not in ctx, "all highlighter markup must be stripped")
        require("... alpha matched fragment here ..." in ctx,
                "a cut-open fragment must be marked with ellipses on both ends")
        require("MOST RELEVANT EXCERPTS:" in ctx, "highlight fragments must be labelled")
        require("one two three four five" in ctx and "six" not in ctx,
                "max_words=5 must cut the body after the fifth word")
        require(ctx.index("===== DOCUMENT 1 =====") < ctx.index("===== DOCUMENT 2 ====="),
                "fences must be numbered in retrieval order")
        # Same fixture without the word budget: text_t is prepended to text, never
        # substituted for it, so both bodies have to be in there.
        full = build_context(docs, hl, top_n=3, max_words=1000)
        require("structured beta payload" in full and "body of beta." in full,
                "text_t must be concatenated with text, not substituted for it")
        return "2 of 4 docs kept (weak dropped, top_n honoured), tags stripped, body cut at 5 words"

    check("build_context selects, cleans and lays out the RAG context", t_build_context)

    def t_build_instruction() -> str:
        ctx = build_context(docs[:2], {}, top_n=2, max_words=50)
        ins = build_instruction(ctx, "who is alpha?")
        require(ins.startswith(ctx), "the context must open the prompt")
        require("Those were the 2 documents." in ins, "document count must match the fences")
        require(ins.endswith("Question: who is alpha?\nAnswer:"),
                "the question must occupy the final slot")
        require('Never begin with "Based on" or "According to"' in ins,
                "the measured ban on those two openings must be in the prompt")
        require("There is no information about" in ins, "the pinned refusal opening must be present")
        empty = build_instruction("", "q")
        require("Those were the 1 documents." in empty, "an empty context must floor the count at 1")
        return "context first, 'Those were the 2 documents.', question last, ban clause present"

    check("build_instruction assembles the whole measured prompt", t_build_instruction)

    def t_resolve_location() -> str:
        require(resolve_location("us") == "CHICAGO-96", "us must map to CHICAGO-96")
        require(resolve_location(" DE ") == "DE-SOLR-9", "aliases are trimmed and case-insensitive")
        require(resolve_location("fi") == "FINLAND9", "fi must map to FINLAND9")
        require(resolve_location("FINLAND9") == "FINLAND9", "raw environment ids pass through")
        require(set(VECTOR_LOCATIONS) == {"us", "de", "fi"}, "alias table changed unexpectedly")
        return "us/de/fi -> CHICAGO-96/DE-SOLR-9/FINLAND9, unknown values pass through"

    check("resolve_location maps aliases to environments", t_resolve_location)


# --------------------------------------------------------------------------- #
# phase 1 — create the temp indexes and start the async writes early           #
# --------------------------------------------------------------------------- #

def phase_write_setup(client: OpensolrClient) -> None:
    section(f"write path — temp indexes {TMP_CLIENT} / {TMP_FROMTEXTS}")

    def t_regions() -> str:
        regions = client.vector_regions()
        require(isinstance(regions, list) and regions, f"expected a non-empty list, got {regions!r}")
        for r in regions:
            require(isinstance(r, dict) and {"environment", "country", "solr_version"} <= set(r),
                    f"malformed region entry: {r!r}")
        envs = {r["environment"] for r in regions}
        require("FINLAND9" in envs, f"FINLAND9 must be vector-enabled, got {sorted(envs)}")
        require(client.vector_regions() is regions, "vector_regions must be cached per client")
        return f"{len(regions)} live vector regions: {', '.join(sorted(envs))} (cached on 2nd call)"

    check("vector_regions returns the live vector-enabled environments", t_regions)

    def t_bad_location() -> str:
        try:
            client.create_index("mcp_t_never_created__dense", "mars")
        except ValueError as exc:
            require("not a vector-enabled" in str(exc), f"unexpected message: {exc}")
            return "location 'mars' rejected client-side against the live region list"
        raise AssertionError("an unknown location must raise ValueError before creating anything")

    check("create_index refuses a non vector-enabled location", t_bad_location)

    def t_create() -> str:
        body = client.create_index(TMP_CLIENT, LOCATION)
        require(isinstance(body, dict), f"expected a dict, got {type(body).__name__}")
        require(body.get("status") is True, f"create failed: {body}")
        require("CREATED" in str(body.get("msg", "")).upper(), f"unexpected msg: {body.get('msg')}")
        STATE["tmp_created"] = True
        return f"{TMP_CLIENT} created in {resolve_location(LOCATION)}: {body.get('msg')}"

    check(f"create_index creates {TMP_CLIENT}", t_create)

    def t_core_info() -> str:
        require(STATE.get("tmp_created"), "temp index was never created")
        info = wait_until(
            lambda: client.get_core_info(TMP_CLIENT, refresh=True) or None,
            f"{TMP_CLIENT} to resolve", timeout=60,
        )
        require(info.get("connection_url", "").startswith("https://"),
                f"bad connection_url: {info.get('connection_url')!r}")
        require(TMP_CLIENT in info["connection_url"], "connection_url must point at this index")
        require(str(info.get("solr_version", "")).startswith("9"),
                f"a vector index must run Solr 9.x, got {info.get('solr_version')!r}")
        require(info.get("auth_username"), "the new index must come with HTTP basic auth")
        STATE["tmp_ready"] = True
        return f"{info['connection_url']} solr {info['solr_version']} auth={info['auth_username']}"

    check("get_core_info(refresh=True) resolves the new index endpoint", t_core_info)

    def t_index_list() -> str:
        names = [i["index_name"] for i in client.get_index_list()]
        require(DEMO in names, f"{DEMO} missing from {names}")
        require(TMP_CLIENT in names, f"{TMP_CLIENT} missing from {names}")
        return f"{len(names)} indexes on the account, both {DEMO} and the new temp index listed"

    check("get_index_list lists the demo index and the new one", t_index_list)

    def t_ingest() -> str:
        require(STATE.get("tmp_ready"), "temp index not ready")
        body = client.ingest(TMP_CLIENT, INGEST_DOCS)
        require(body.get("status") is True, f"ingest refused: {body}")
        require(body.get("total_docs") == 2, f"expected total_docs=2, got {body.get('total_docs')}")
        job_id = body.get("job_id")
        require(isinstance(job_id, str) and len(job_id) == 32, f"bad job_id: {job_id!r}")
        expected = [md5(d["uri"]) for d in INGEST_DOCS]
        require(body.get("doc_ids") == expected,
                f"doc_ids must be md5(uri): expected {expected}, got {body.get('doc_ids')}")
        STATE["job_id"] = job_id
        STATE["ingest_ids"] = expected
        return f"job {job_id[:12]}… queued, 2 docs, ids = md5(uri) as documented"

    check("ingest queues documents and returns md5(uri) ids", t_ingest)

    def t_status_now() -> str:
        job_id = STATE.get("job_id")
        require(job_id, "no job id from ingest")
        body = client.ingest_status(job_id)
        require(body.get("status") is True, f"ingest_status failed: {body}")
        job = body.get("job", {})
        require(job.get("id") == job_id, f"status returned the wrong job: {job.get('id')}")
        require(int(job.get("total_docs", 0)) == 2, f"expected total_docs=2, got {job.get('total_docs')}")
        require(job.get("state_label") in ("pending", "processing", "completed"),
                f"unexpected state: {job.get('state_label')!r}")
        return f"job {job_id[:12]}… state={job['state_label']} total_docs={job['total_docs']}"

    check("ingest_status reports the queued job", t_status_now)

    def t_add_texts() -> str:
        require(STATE.get("tmp_ready"), "temp index not ready")
        store = OpensolrVectorStore(index=TMP_CLIENT, client=client)
        STATE["tmp_store"] = store
        ids = store.add_texts(LC_TEXTS, metadatas=LC_METAS, ids=LC_IDS)
        expected = [
            md5(f"https://ingest.opensolr.com/{TMP_CLIENT}/{quote(i, safe='')}") for i in LC_IDS
        ]
        require(ids == expected, f"add_texts must return md5 of the synthesized uri: {ids} != {expected}")
        STATE["lc_solr_ids"] = ids
        return f"3 texts queued, returned ids {ids[0][:8]}…/{ids[1][:8]}…/{ids[2][:8]}… (md5 of uri)"

    check("OpensolrVectorStore.add_texts queues texts and returns Solr ids", t_add_texts)

    def t_from_texts() -> str:
        store = OpensolrVectorStore.from_texts(
            ["A vector store built by from_texts, about managed Solr search infrastructure.",
             "A second from_texts document, about baking sourdough bread at home."],
            embedding=None,
            metadatas=[{"title": "ft one", "batch": RUN}, {"title": "ft two", "batch": RUN}],
            index=TMP_FROMTEXTS,
            location=LOCATION,
            client=client,
        )
        STATE["fromtexts_created"] = True
        require(isinstance(store, OpensolrVectorStore), f"got {type(store).__name__}")
        info = client.get_core_info(TMP_FROMTEXTS, refresh=True)
        require(TMP_FROMTEXTS in info.get("connection_url", ""),
                f"from_texts did not create the index: {info}")
        STATE["fromtexts_store"] = store
        return f"{TMP_FROMTEXTS} created on demand and 2 texts queued into it"

    check("from_texts creates the index and queues the texts", t_from_texts)


# --------------------------------------------------------------------------- #
# phase 2 — read-only work on the seeded demo index                            #
# --------------------------------------------------------------------------- #

def phase_read(client: OpensolrClient) -> None:
    section(f"client — read-only against {DEMO}")

    def t_core_info() -> str:
        info = client.get_core_info(DEMO)
        require(info.get("connection_url", "").startswith("https://"),
                f"bad connection_url: {info.get('connection_url')!r}")
        require(str(info.get("solr_version", "")).startswith("9"),
                f"expected Solr 9.x, got {info.get('solr_version')!r}")
        require(client.get_core_info(DEMO) is info, "get_core_info must cache per client")
        return f"{info['connection_url']} solr {info['solr_version']} (cached on 2nd call)"

    check("get_core_info resolves the demo index and caches it", t_core_info)

    def t_endpoint() -> str:
        url, auth = client.solr_endpoint(DEMO)
        require(url.startswith("https://") and url.endswith(DEMO), f"bad endpoint {url!r}")
        require(isinstance(auth, tuple) and len(auth) == 2 and auth[0],
                f"expected (user, password) basic auth, got {auth!r}")
        return f"{url} with basic auth as {auth[0]!r}"

    check("solr_endpoint returns the native Solr URL plus basic auth", t_endpoint)

    def t_select() -> str:
        body = client.solr_select(DEMO, {"q": "*:*", "rows": 3, "fl": "id,title"})
        found = body["response"]["numFound"]
        docs = body["response"]["docs"]
        require(found > 0, "the demo index must not be empty")
        require(len(docs) == 3, f"rows=3 must return 3 docs, got {len(docs)}")
        require(all(d.get("id") for d in docs), "every doc must carry an id")
        require(all(set(d) <= {"id", "title"} for d in docs), f"fl was ignored: {sorted(docs[0])}")
        STATE["demo_total"] = found
        return f"numFound={found}, 3 docs returned, fl honoured (id,title only)"

    check("solr_select runs a native Solr query", t_select)

    def t_embed() -> str:
        passage = client.embed(DEMO, QUERY, is_query=False)
        query_vec = client.embed(DEMO, QUERY, is_query=True)
        require(len(query_vec) == 1024, f"expected 1024 dimensions, got {len(query_vec)}")
        require(len(passage) == 1024, f"expected 1024 dimensions, got {len(passage)}")
        require(all(isinstance(x, float) for x in query_vec[:32]), "vector must be floats")
        require(abs(norm(query_vec) - 1.0) < 1e-3, f"expected a unit vector, norm={norm(query_vec):.4f}")
        require(any(x != 0.0 for x in query_vec), "an all-zero vector means the model never ran")
        require(query_vec != passage, "is_query=1 must produce a different vector than is_query=0")
        STATE["query_vector"] = query_vec
        return (f"1024 dims, |v|={norm(query_vec):.4f}, query vs passage differ "
                f"(cos={cosine(query_vec, passage):.4f})")

    check("embed returns a 1024-dim unit vector and honours is_query", t_embed)

    def t_batch_embed() -> str:
        texts = ["managed apache solr hosting", "orange marmalade recipe", "hybrid vector search"]
        vectors = client.batch_embed(DEMO, texts)
        require(len(vectors) == 3, f"expected 3 vectors, got {len(vectors)}")
        require(all(len(v) == 1024 for v in vectors), f"dims: {[len(v) for v in vectors]}")
        require(all(abs(norm(v) - 1.0) < 1e-3 for v in vectors), "all vectors must be unit length")
        require(vectors[0] != vectors[1], "different texts must produce different vectors")
        near = cosine(vectors[0], vectors[2])
        far = cosine(vectors[0], vectors[1])
        require(near > far, f"'solr hosting' must sit closer to 'vector search' ({near:.3f}) "
                            f"than to 'marmalade' ({far:.3f})")
        require(BATCH_EMBED_MAX == 50, f"server batch limit constant changed: {BATCH_EMBED_MAX}")
        return f"3x1024 unit vectors, semantically ordered (near={near:.3f} > far={far:.3f})"

    check("batch_embed embeds many texts in one call", t_batch_embed)

    def t_embed_and_search() -> str:
        body = client.embed_and_search(DEMO, QUERY, rows=3)
        require(body.get("status") is not False, f"call failed: {str(body)[:200]}")
        results = body.get("results", {})
        docs = results.get("docs", [])
        require(len(docs) == 3, f"rows=3 must return 3 docs, got {len(docs)}")
        require(int(results.get("num", 0)) > 0, f"num must be positive, got {results.get('num')}")
        scores = [float(d.get("score", 0)) for d in docs]
        require(all(s > 0 for s in scores), f"every hit needs a score, got {scores}")
        require(scores == sorted(scores, reverse=True), f"hits must be ranked: {scores}")
        require(isinstance(results.get("hl"), dict) and results["hl"],
                "the tuned pipeline must return highlight fragments for the RAG context")
        return (f"num={results['num']}, top scores {['%.3f' % s for s in scores]}, "
                f"{len(results['hl'])} highlighted docs")

    check("embed_and_search runs the platform's tuned hybrid pipeline", t_embed_and_search)

    # --- hybrid_search: three shapes, each with fresh_bias off and on -------- #

    def hybrid_pair(label: str, **kwargs: Any) -> str:
        """Run one hybrid_search shape twice — bias off, then on — and prove the
        recency curve re-orders without ever filtering (numFound identical)."""
        off = client.hybrid_search(DEMO, QUERY, fresh_bias=False, **kwargs)
        on = client.hybrid_search(DEMO, QUERY, fresh_bias=True, **kwargs)
        n_off = off["response"]["numFound"]
        n_on = on["response"]["numFound"]
        docs_off = off["response"]["docs"]
        require(n_off > 0, f"{label}: no hits at all")
        # A narrow shape (intersection) can legitimately match fewer documents
        # than the page size, so rows is a ceiling, not a promise.
        require(len(docs_off) == min(kwargs.get("rows", 5), n_off),
                f"{label}: expected min(rows, {n_off}) docs, got {len(docs_off)}")
        scores = [float(d.get("score", 0)) for d in docs_off]
        require(all(s > 0 for s in scores), f"{label}: hits without scores: {scores}")
        require(scores == sorted(scores, reverse=True), f"{label}: hits not ranked: {scores}")
        require(n_on == n_off,
                f"{label}: fresh_bias must re-order, never filter — numFound {n_off} -> {n_on}")
        return f"numFound={n_off} with and without fresh_bias, top score {scores[0]:.3f}"

    check("hybrid_search union mode + fresh_bias keeps numFound identical",
          lambda: hybrid_pair("union", rows=5))
    check("hybrid_search intersection mode + fresh_bias keeps numFound identical",
          lambda: hybrid_pair("intersection", rows=5, mode="intersection", alpha=0.8))

    def t_hybrid_filtered() -> str:
        kwargs = dict(rows=3, mode="meaning_required", fl="id,meta_domain,score",
                      fq='meta_domain:"goal.com"')
        detail = hybrid_pair("filtered", **kwargs)
        body = client.hybrid_search(DEMO, QUERY, fresh_bias=False, **kwargs)
        docs = body["response"]["docs"]
        require(all(d.get("meta_domain") == "goal.com" for d in docs),
                f"fq leaked: {[d.get('meta_domain') for d in docs]}")
        require(all(set(d) <= {"id", "meta_domain", "score"} for d in docs),
                f"fl ignored: {sorted(docs[0])}")
        require(body["response"]["numFound"] < STATE.get("demo_total", 10 ** 9),
                "the filter must actually narrow the result set")
        return detail + ", every hit from goal.com, fl honoured"

    check("hybrid_search meaning_required mode with fq + fl filters correctly", t_hybrid_filtered)

    def t_ai_summary() -> str:
        answer = client.ai_summary(DEMO, RAG_QUESTION)
        require(answer, "the RAG answer must not be empty")
        require(len(answer) > 80, f"suspiciously short answer: {answer!r}")
        low = answer.lower()
        require(not low.startswith("based on"), f"shipped prompt forbids this opening: {answer[:60]!r}")
        require(not low.startswith("according to"), f"shipped prompt forbids this opening: {answer[:60]!r}")
        require(not low.startswith("there is no information about"),
                f"retrieval found nothing for a well-covered question: {answer[:80]!r}")
        return f"{len(answer)} chars, opens {answer.splitlines()[0][:70]!r}"

    check("ai_summary answers from the index and obeys the shipped instruction", t_ai_summary)

    def t_ai_summary_instruction() -> str:
        answer = client.ai_summary(
            DEMO, "ignored",
            instruction="Reply with exactly the single word PONG and nothing else.",
        )
        require(answer, "custom-instruction answer must not be empty")
        require("PONG" in answer.upper(), f"the caller's own instruction never reached the model: {answer[:120]!r}")
        return f"custom instruction honoured verbatim: {answer[:60]!r}"

    check("ai_summary honours a caller-supplied instruction override", t_ai_summary_instruction)

    # --- LangChain wrappers against the demo index -------------------------- #

    section(f"langchain wrappers — read-only against {DEMO}")

    def t_embeddings_query() -> str:
        emb = OpensolrEmbeddings(client=client, index=DEMO)
        vec = emb.embed_query(QUERY)
        require(len(vec) == 1024, f"expected 1024 dims, got {len(vec)}")
        require(abs(norm(vec) - 1.0) < 1e-3, f"expected a unit vector, norm={norm(vec):.4f}")
        stored = STATE.get("query_vector")
        if stored:
            require(cosine(vec, stored) > 0.99,
                    f"embed_query must use the query-side model (cos={cosine(vec, stored):.4f})")
        return f"1024 dims, |v|={norm(vec):.4f}, matches client.embed(is_query=True)"

    check("OpensolrEmbeddings.embed_query returns a 1024-dim query vector", t_embeddings_query)

    def t_embeddings_documents() -> str:
        emb = OpensolrEmbeddings(client=client, index=DEMO)
        vectors = emb.embed_documents(["managed solr hosting", "orange marmalade"])
        require(len(vectors) == 2, f"expected 2 vectors, got {len(vectors)}")
        require(all(len(v) == 1024 for v in vectors), f"dims: {[len(v) for v in vectors]}")
        require(vectors[0] != vectors[1], "distinct texts must embed differently")
        require(emb.embed_documents([]) == [], "an empty list must short-circuit to []")
        return "2x1024 dims, empty input short-circuits without a request"

    check("OpensolrEmbeddings.embed_documents batches texts", t_embeddings_documents)

    def t_ctor_validation() -> str:
        for factory, label in (
            (lambda: OpensolrEmbeddings(index=DEMO), "OpensolrEmbeddings without credentials"),
            (lambda: OpensolrEmbeddings(client=client, index=""), "OpensolrEmbeddings without index"),
            (lambda: OpensolrVectorStore(index=DEMO), "OpensolrVectorStore without credentials"),
        ):
            try:
                factory()
            except ValueError:
                continue
            raise AssertionError(f"{label} must raise ValueError")
        return "missing credentials / missing index both raise ValueError"

    check("constructors reject incomplete configuration", t_ctor_validation)

    demo_store = OpensolrVectorStore(index=DEMO, client=client)
    STATE["demo_store"] = demo_store

    def t_embeddings_property() -> str:
        emb = demo_store.embeddings
        require(isinstance(emb, OpensolrEmbeddings), f"got {type(emb).__name__}")
        require(emb._index == DEMO, "the property must bind to this store's index")
        require(emb._client is client, "the property must reuse this store's client")
        return "OpensolrEmbeddings bound to this index and client"

    check("VectorStore.embeddings exposes server-side embeddings", t_embeddings_property)

    def t_similarity_search() -> str:
        docs = demo_store.similarity_search(QUERY, k=3)
        require(len(docs) == 3, f"k=3 must return 3 documents, got {len(docs)}")
        require(all(isinstance(d, Document) for d in docs), "results must be langchain Documents")
        require(all(d.page_content.strip() for d in docs), "page_content must not be empty")
        require(all(d.id for d in docs), "each Document must carry its Solr id")
        require(len({d.id for d in docs}) == 3, "results must be distinct documents")
        return f"3 Documents, ids {[d.id[:8] for d in docs]}, first {len(docs[0].page_content)} chars"

    check("similarity_search (pure kNN) returns Documents", t_similarity_search)

    def t_similarity_with_score() -> str:
        pairs = demo_store.similarity_search_with_score(QUERY, k=4)
        require(len(pairs) == 4, f"k=4 must return 4 pairs, got {len(pairs)}")
        scores = [s for _, s in pairs]
        require(all(isinstance(s, float) and s > 0 for s in scores), f"bad scores: {scores}")
        require(scores == sorted(scores, reverse=True), f"scores must descend: {scores}")
        require(all(isinstance(d, Document) for d, _ in pairs), "must return (Document, score) pairs")
        return f"4 (Document, score) pairs, descending {['%.3f' % s for s in scores]}"

    check("similarity_search_with_score returns ranked scores", t_similarity_with_score)

    def t_hybrid_shape() -> str:
        docs = demo_store.similarity_search(QUERY, k=3, hybrid=True, mode="union", alpha=0.5)
        require(len(docs) == 3, f"expected 3 documents, got {len(docs)}")
        require(all(d.page_content.strip() for d in docs), "page_content must not be empty")
        return f"3 Documents via the {{!hybrid}} parser, first id {docs[0].id[:8]}…"

    check("similarity_search(hybrid=True) fuses BM25 and kNN", t_hybrid_shape)

    def t_lexical_shape() -> str:
        plain = demo_store.similarity_search_with_score("transfer", k=3, lexical=True)
        biased = demo_store.similarity_search_with_score("transfer", k=3, lexical=True, fresh_bias=True)
        require(len(plain) == 3, f"expected 3 lexical hits, got {len(plain)}")
        require(all(s > 0 for _, s in plain), f"lexical hits need BM25 scores: {[s for _, s in plain]}")
        require(len(biased) == len(plain),
                f"fresh_bias must not drop hits: {len(plain)} -> {len(biased)}")
        return f"3 keyword hits (top score {plain[0][1]:.3f}), same count with fresh_bias on"

    check("similarity_search(lexical=True) does keyword-only search", t_lexical_shape)

    def t_by_vector() -> str:
        vector = STATE.get("query_vector")
        require(vector, "no query vector available from the embed check")
        docs = demo_store.similarity_search_by_vector(vector, k=2)
        require(len(docs) == 2, f"expected 2 documents, got {len(docs)}")
        require(all(isinstance(d, Document) and d.page_content.strip() for d in docs),
                "must return non-empty Documents")
        return f"2 Documents for a pre-computed 1024-dim vector, top id {docs[0].id[:8]}…"

    check("similarity_search_by_vector searches with a supplied embedding", t_by_vector)

    def t_bad_mode() -> str:
        try:
            demo_store.similarity_search(QUERY, k=1, hybrid=True, mode="nonsense")
        except ValueError as exc:
            require("mode must be one of" in str(exc), f"unexpected message: {exc}")
            return "an unknown hybrid mode raises ValueError"
        raise AssertionError("an unknown hybrid mode must raise ValueError")

    check("similarity_search validates the hybrid mode", t_bad_mode)


# --------------------------------------------------------------------------- #
# phase 3 — verify the async writes, then exercise the rest of the write path  #
# --------------------------------------------------------------------------- #

def phase_write_verify(client: OpensolrClient) -> None:
    section(f"write path — verifying {TMP_CLIENT} (ingestion is async)")

    def t_job_completes() -> str:
        job_id = STATE.get("job_id")
        require(job_id, "no job id from ingest")

        def done() -> Any:
            body = client.ingest_status(job_id)
            job = body.get("job", {})
            if job.get("state_label") in ("completed", "failed", "stopped"):
                return job
            return None

        job = wait_until(done, f"ingest job {job_id[:12]}… to finish")
        require(job["state_label"] == "completed", f"job ended as {job['state_label']}: {job.get('error')}")
        require(int(job["success_docs"]) == 2, f"expected 2 successful docs, got {job['success_docs']}")
        require(int(job["failed_docs"]) == 0, f"{job['failed_docs']} documents failed: {job.get('error')}")
        return f"state=completed success_docs=2 failed_docs=0 (job {job_id[:12]}…)"

    check("ingest_status polls the job through to completion", t_job_completes)

    def t_ingest_wait() -> str:
        """ingest(wait=True) is its own branch — it polls ingest_status itself
        and hangs the finished job off final_status."""
        doc = {
            "uri": f"https://example.com/langchain-live/{RUN}/blocking",
            "title": "Opensolr live test: blocking ingest",
            "description": "A document queued with wait=True so ingest blocks until the job finishes.",
            "text": ("This document was submitted with wait=True, which makes the client poll the "
                     "ingestion queue itself instead of returning a job id for the caller to chase."),
        }
        body = client.ingest(TMP_CLIENT, [doc], wait=True, timeout=INGEST_TIMEOUT)
        require(body.get("status") is True, f"ingest refused: {body}")
        final = body.get("final_status")
        require(isinstance(final, dict), "wait=True must attach the finished job as final_status")
        job = final.get("job", {})
        require(job.get("state_label") == "completed", f"job ended as {job.get('state_label')!r}")
        require(int(job.get("success_docs", 0)) == 1, f"expected 1 successful doc, got {job.get('success_docs')}")
        got = client.solr_select(TMP_CLIENT, {"q": f'id:"{md5(doc["uri"])}"', "rows": 1, "fl": "id"})
        require(got["response"]["numFound"] == 1,
                "wait=True returned before the document was actually indexed")
        return f"blocked until job {body['job_id'][:12]}… completed; the document is already searchable"

    check("ingest(wait=True) blocks until the queue has processed the job", t_ingest_wait)

    def t_docs_searchable() -> str:
        ids = (STATE.get("ingest_ids") or []) + (STATE.get("lc_solr_ids") or [])
        require(len(ids) == 5, "expected 2 ingested + 3 add_texts documents")
        joined = " OR ".join(f'"{i}"' for i in ids)

        def all_there() -> Any:
            body = client.solr_select(TMP_CLIENT, {"q": f"id:({joined})", "rows": 10, "fl": "id"})
            return body if body["response"]["numFound"] == 5 else None

        body = wait_until(all_there, "all 5 documents to become searchable")
        found = {d["id"] for d in body["response"]["docs"]}
        require(found == set(ids), f"missing documents: {set(ids) - found}")
        return "all 5 queued documents are searchable by their md5(uri) ids"

    check("queued documents become searchable in the temp index", t_docs_searchable)

    def t_server_side_vectors() -> str:
        body = client.hybrid_search(TMP_CLIENT, "dense vector retrieval on apache solr",
                                    rows=3, mode="meaning_required")
        docs = body["response"]["docs"]
        require(body["response"]["numFound"] > 0, "no vector hits — ingestion computed no embeddings")
        require(float(docs[0].get("score", 0)) > 0, f"top hit has no score: {docs[0]}")
        require(docs[0]["id"] == STATE["ingest_ids"][0],
                f"the Solr article should outrank the marmalade one, got {docs[0].get('title')!r}")
        return (f"meaning_required kNN matched {body['response']['numFound']} docs, "
                f"top hit is the Solr article (score {float(docs[0]['score']):.3f})")

    check("ingestion computed embeddings server-side (kNN finds the right doc)", t_server_side_vectors)

    def t_metadata_roundtrip() -> str:
        store: OpensolrVectorStore = STATE["tmp_store"]
        docs = store.get_by_ids(STATE["lc_solr_ids"])
        require(len(docs) == 3, f"expected 3 documents, got {len(docs)}")
        require([d.id for d in docs] == STATE["lc_solr_ids"], "get_by_ids must preserve the given order")
        for doc, meta in zip(docs, LC_METAS):
            require(doc.metadata == meta, f"metadata did not round-trip: {doc.metadata} != {meta}")
        require(docs[0].page_content.startswith("LangChain retrievers"),
                f"page_content is wrong: {docs[0].page_content[:60]!r}")
        return f"3 Documents in order, metadata identical incl. int seq and batch={RUN}"

    check("get_by_ids round-trips ids, text and metadata losslessly", t_metadata_roundtrip)

    def t_filter() -> str:
        store: OpensolrVectorStore = STATE["tmp_store"]
        hits = store.similarity_search("glaciers and fjords", k=5, filter={"batch": RUN, "seq": 2})
        require(len(hits) == 1, f"the metadata filter must leave exactly 1 doc, got {len(hits)}")
        require(hits[0].metadata["seq"] == 2, f"wrong document: {hits[0].metadata}")
        require("fjords" in hits[0].page_content, f"wrong content: {hits[0].page_content[:60]!r}")
        return "dict filter {batch, seq} narrowed 5 documents down to the right one"

    check("similarity_search honours a metadata filter", t_filter)

    def t_retriever() -> str:
        store: OpensolrVectorStore = STATE["tmp_store"]
        retriever = store.as_retriever(search_kwargs={"k": 2, "filter": {"batch": RUN}})
        docs = retriever.invoke("hybrid search fusing bm25 and knn")
        require(len(docs) == 2, f"the retriever must return 2 documents, got {len(docs)}")
        require(all(isinstance(d, Document) for d in docs), "retriever must return Documents")
        require(all(isinstance(d.metadata, dict) and d.metadata for d in docs),
                f"Documents came back without metadata: {[d.metadata for d in docs]}")
        require(all(d.metadata.get("batch") == RUN for d in docs),
                f"the retriever dropped the filter: {[d.metadata for d in docs]}")
        require(all(d.metadata.get("source") == "langchain-opensolr-live-test" for d in docs),
                "custom metadata keys must survive retrieval")
        require("bm25" in docs[0].page_content.lower() or "hybrid" in docs[0].page_content.lower(),
                f"top retrieved document is off-topic: {docs[0].page_content[:70]!r}")
        return f"2 Documents with full metadata, top one is the hybrid-search text"

    check("as_retriever returns Documents with their metadata", t_retriever)

    def t_ai_answer() -> str:
        store: OpensolrVectorStore = STATE["tmp_store"]
        answer = store.ai_answer("What does hybrid search fuse?", filter={"batch": RUN})
        require(answer.strip(), "ai_answer must not be empty")
        low = answer.lower()
        require(not low.startswith("based on"), f"shipped prompt forbids this opening: {answer[:60]!r}")
        require(not low.startswith("according to"), f"shipped prompt forbids this opening: {answer[:60]!r}")
        require("bm25" in low or "lexical" in low or "knn" in low,
                f"the answer is not grounded in the filtered documents: {answer[:120]!r}")
        return f"{len(answer)} chars grounded in the filtered docs: {answer.splitlines()[0][:70]!r}"

    check("ai_answer does RAG over a filtered subset of the index", t_ai_answer)

    def t_solr_update() -> str:
        doc_id = f"direct-{RUN}"
        body = client.solr_update(TMP_CLIENT, {"add": {"doc": {
            "id": doc_id,
            "uri": f"https://example.com/langchain-live/{RUN}/direct",
            "title": "Direct solr_update document",
            "text": "Written straight to Solr with solr_update, bypassing the ingestion queue.",
        }}}, commit=True)
        require(body.get("responseHeader", {}).get("status") == 0, f"update failed: {body}")
        got = client.solr_select(TMP_CLIENT, {"q": f'id:"{doc_id}"', "rows": 1, "fl": "id,title"})
        require(got["response"]["numFound"] == 1, "the directly written document is not searchable")
        require(got["response"]["docs"][0]["title"] == "Direct solr_update document",
                "the written document came back wrong")
        STATE["direct_id"] = doc_id
        return f"doc {doc_id!r} written with commit=true and immediately searchable"

    check("solr_update writes straight to Solr and commits", t_solr_update)

    def t_delete_by_id() -> str:
        store: OpensolrVectorStore = STATE["tmp_store"]
        victim = STATE["lc_solr_ids"][2]
        require(store.delete(ids=[victim]) is True, "delete(ids=...) must return True")
        body = client.solr_select(TMP_CLIENT, {"q": f'id:"{victim}"', "rows": 1})
        require(body["response"]["numFound"] == 0, "the targeted document survived the delete")
        others = client.solr_select(
            TMP_CLIENT,
            {"q": "id:(" + " OR ".join(f'"{i}"' for i in STATE["lc_solr_ids"][:2]) + ")", "rows": 5},
        )
        require(others["response"]["numFound"] == 2, "delete by id removed more than it should have")
        return "1 document deleted by id, the other 2 untouched"

    check("delete(ids=...) removes exactly the named documents", t_delete_by_id)

    def t_delete_by_query() -> str:
        store: OpensolrVectorStore = STATE["tmp_store"]
        require(store.delete(query=f'id:"{STATE["direct_id"]}"') is True,
                "delete(query=...) must return True")
        body = client.solr_select(TMP_CLIENT, {"q": f'id:"{STATE["direct_id"]}"', "rows": 1})
        require(body["response"]["numFound"] == 0, "the document matched by the query survived")
        return f"raw Solr query deleted {STATE['direct_id']!r}"

    check("delete(query=...) removes documents by raw Solr query", t_delete_by_query)

    def t_delete_requires_target() -> str:
        store: OpensolrVectorStore = STATE["tmp_store"]
        try:
            store.delete()
        except ValueError as exc:
            require("delete_all" in str(exc), f"unexpected message: {exc}")
            return "delete() with no target raises ValueError instead of wiping the index"
        raise AssertionError("delete() with no arguments must raise ValueError")

    check("delete() refuses to run without a target", t_delete_requires_target)

    def t_delete_all() -> str:
        store: OpensolrVectorStore = STATE["tmp_store"]
        require(store.delete(delete_all=True) is True, "delete(delete_all=True) must return True")
        body = client.solr_select(TMP_CLIENT, {"q": "*:*", "rows": 0})
        require(body["response"]["numFound"] == 0,
                f"{body['response']['numFound']} documents survived delete_all")
        return "delete_all=True emptied the temp index (numFound=0)"

    check("delete(delete_all=True) empties the index", t_delete_all)

    def t_fromtexts_searchable() -> str:
        require(STATE.get("fromtexts_created"), "from_texts never created its index")
        store: OpensolrVectorStore = STATE["fromtexts_store"]

        def both() -> Any:
            body = client.solr_select(TMP_FROMTEXTS, {"q": "*:*", "rows": 0})
            return body if body["response"]["numFound"] >= 2 else None

        wait_until(both, f"from_texts documents to appear in {TMP_FROMTEXTS}")
        docs = store.similarity_search("managed search infrastructure", k=2)
        require(len(docs) == 2, f"expected 2 documents, got {len(docs)}")
        require(docs[0].metadata.get("batch") == RUN, f"metadata lost: {docs[0].metadata}")
        require("solr" in docs[0].page_content.lower(),
                f"the wrong document ranked first: {docs[0].page_content[:60]!r}")
        return "both from_texts documents are searchable and correctly ranked"

    check("from_texts documents land in the new index and rank correctly", t_fromtexts_searchable)


# --------------------------------------------------------------------------- #
# cleanup — always runs                                                        #
# --------------------------------------------------------------------------- #

def cleanup(client: OpensolrClient) -> None:
    section("cleanup")

    for name, created_flag in ((TMP_CLIENT, "tmp_created"), (TMP_FROMTEXTS, "fromtexts_created")):
        def drop(name: str = name, created_flag: str = created_flag) -> str:
            if not STATE.get(created_flag):
                return f"{name} was never created, nothing to delete"
            # delete_index has no wrapper on the client, so this also exercises
            # the generic client.mgmt() passthrough.
            body = client.mgmt("delete_index", index_name=name)
            require(isinstance(body, dict) and body.get("status") is True, f"delete failed: {body}")
            require("DELETE" in str(body.get("msg", "")).upper(), f"unexpected msg: {body.get('msg')}")
            names = [i["index_name"] for i in client.get_index_list()]
            require(name not in names, f"{name} still listed on the account after deletion")
            return f"{name} deleted ({body['msg']}) and gone from the index list"

        check(f"temp index {name} deleted", drop)

    def t_close() -> str:
        client.close()
        try:
            client.get_index_list()
        except Exception as exc:  # httpx raises once the transport is closed
            return f"the http client is closed ({type(exc).__name__} on the next request)"
        raise AssertionError("requests still go through after close()")

    check("close() shuts the HTTP client down", t_close)


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    started = time.time()
    print(f"langchain-opensolr live test — account {EMAIL}, run id {RUN}")
    print(f"read-only index {DEMO}; temp indexes {TMP_CLIENT}, {TMP_FROMTEXTS} "
          f"(deleted at the end); pacing at {MAX_PER_MIN} API req/min")

    client = OpensolrClient(EMAIL, API_KEY)
    try:
        phase_pure()
        phase_write_setup(client)
        phase_read(client)
        phase_write_verify(client)
    except KeyboardInterrupt:
        print("\ninterrupted — cleaning up", flush=True)
    finally:
        cleanup(client)

    total = len(PASSED) + len(FAILED)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed "
          f"({total} checks, {_api_requests} API + {_solr_requests} direct-Solr requests, "
          f"{time.time() - started:.0f}s)")
    if FAILED:
        print("\nfailures:")
        for failure in FAILED:
            print(f"  ✘ {_one_line(failure, 200)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
