# langchain-opensolr

LangChain integration for [Opensolr](https://opensolr.com) — managed Apache Solr
with **server-side embeddings** and native **hybrid (BM25 + kNN) search**.

**See it live (real news index, hybrid + AI answer):** https://search.opensolr.com/news__dense?q=how+am+I+supposed+to+save+money%3F

**Product page:** [opensolr.com/langchain](https://opensolr.com/langchain) ·
**Platform:** [opensolr.com](https://opensolr.com) (managed Solr hosting since 2011 —
free 15-day trial, no card)

No local embedding model. No third-party embedding API key. One set of
credentials, and the vectors are computed on Opensolr's GPU infrastructure
(multilingual E5-large-instruct, 1024 dimensions, cosine).

```bash
pip install langchain-opensolr
```

## The whole tutorial

```python
from langchain_opensolr import OpensolrVectorStore

vs = OpensolrVectorStore(
    index="mysite__dense",          # vector-enabled Opensolr index
    email="you@example.com",
    api_key="YOUR_OPENSOLR_API_KEY",
    create_if_missing=True,          # provisions the index on first use
)

vs.add_texts(
    ["Hybrid search fuses BM25 with vector similarity",
     "Cats sleep sixteen hours a day"],
    metadatas=[{"category": "search"}, {"category": "animals"}],
)

docs = vs.similarity_search("how do lexical and semantic search combine?", k=1)
print(docs[0].page_content)
```

That's it — no embedding model was configured, because embedding happens on
the server at both index and query time.

## Try it without an account

There is a public demo account. Point the package at it and everything in this
README works immediately, with no signup:

```bash
export OPENSOLR_EMAIL=mcp@opensolr.com
export OPENSOLR_API_KEY=420b8b23e7b12dc8ab838932145a5065
```

`mcp_demo_d1__dense` is already loaded with 300 news articles, so search, filtering
and grounded answers work the moment you connect. You also get the full write path:
create your own index on the account, ingest into it, query it, delete it.

Know what you are working with:

- **Anything you create there is deleted after 3 days.** Automatically, without warning
  or export. That includes indexes you created and every document in them.
- **The account is shared with everyone reading this.** Your index is visible to them,
  they can change or delete it, and you can do the same to theirs. Never put anything
  real, private or client-owned in it.
- **The limits are per index, and deliberately small.** 200 MB of bandwidth and 50 MB
  of disk per index. Bandwidth is the one you will hit first: it covers a demo, a
  tutorial and a proof of concept, and it will not carry an application.

When you want an index that is private, yours and still there next week, get your own
key — [free 15-day trial, no card](https://opensolr.com/register) — and change the two
variables above. Nothing else in your code changes.

## Hybrid search

Pure vector search fails on exact identifiers; pure BM25 fails on meaning.
Opensolr's `{!hybrid}` query parser fuses both scores **per document**:

```python
docs = vs.similarity_search(
    "affordable restaurants",
    k=5,
    hybrid=True,
    mode="union",     # union | keywords_required | meaning_required | intersection
    alpha=0.5,        # 0 = all semantic … 1 = all lexical
)
```

## Metadata filters

```python
vs.similarity_search("search engines", k=5, filter={"category": "search"})
vs.similarity_search("anything", k=5, filter='meta_rank:[2 TO *]')   # raw Solr fq
```

Metadata round-trips losslessly (stored as JSON alongside filterable
`meta_*` fields).

## As a retriever, in any chain

```python
retriever = vs.as_retriever(search_kwargs={"k": 5, "hybrid": True})
```

## Standalone embeddings

Use Opensolr's embedding endpoint with any other LangChain component:

```python
from langchain_opensolr import OpensolrEmbeddings

emb = OpensolrEmbeddings(email="you@example.com", api_key="...", index="mysite__dense")
emb.embed_query("budget-friendly dining")   # -> 1024 floats
```

## Notes

- Vector-enabled indexes run on Opensolr's Solr 9.x environments — currently
  `us` (Chicago), `de` (Germany), `fi` (Finland). Pass `location=` to choose.
  The list is fetched live from the platform, so new regions work without a
  package upgrade — and **additional dedicated regions can be deployed on
  request** (paid add-on): [support@opensolr.com](mailto:support@opensolr.com).
- A free Opensolr account (15-day trial, no card) includes an AI quota that
  comfortably covers this README end to end:
  [opensolr.com](https://opensolr.com).
- Full platform docs: [AI & Vector Search](https://opensolr.com/opensolr-platform-user-documentation/ai-vector).

## Development

```bash
pip install -e . pytest
pytest tests/unit_tests
OPENSOLR_EMAIL=... OPENSOLR_API_KEY=... OPENSOLR_INDEX=... pytest tests/integration_tests
```

## How writing works (Data Ingestion API)

Writes go through Opensolr's [Data Ingestion API](https://opensolr.com/learn/api-data-ingestion/204/data-ingestion-api-push-documents-to-your-opensolr-index-programmatically)
— the same pipeline the Drupal and WordPress connectors use. It is
**asynchronous**: documents are queued, then embeddings, sentiment, language
and all crawler-identical derived fields are computed **server-side**, and
documents become searchable within about a minute. Progress is visible in
**Control Panel → Data Ingestion** — a per-job status board (queued /
processing / completed / failed, with processed / success / failed document
counts per job) — and via the `ingest_status` API. Each document's
identity is its `uri` (the Solr id is `md5(uri)`): pass a real URL in
metadata (`{"uri": "https://..."}`), or a deterministic one is synthesized
from your id. Re-submitting the same `uri` updates the document. Pass
`{"rtf": True, "uri": "https://.../file.pdf"}` and the server extracts the
text from PDF/DOCX/XLSX for you.

## Lexical-only mode

Don't need vectors? Pure keyword search skips the embedding call entirely —
zero AI quota, and it works on **any** Opensolr index, including non-vector
ones and older Solr versions.

## Your index schema

Documents follow the Opensolr document model (`title`, `description`, `text`,
`meta_*` custom fields). The whole schema, every field and every type suffix,
is explained in the [Index Schema Reference](https://opensolr.com/opensolr-platform-user-documentation/schema-reference).
To see your own copy: **Control Panel → click your
index → Configuration → Edit File → schema.xml**. Prefer zero-effort data
entry? Configure the **Web Crawler** in the Control Panel (Index Tools →
WebCrawler): add your site URL, validate it, and Opensolr indexes the whole
site for you.

## Grounded RAG answers

One call: hybrid retrieval picks the top hits, whose content becomes the LLM
context, and Opensolr's server-side LLM answers — the same pipeline that powers
the AI answers on our hosted search pages. No LLM key, no chain to assemble:

```python
answer = vs.ai_answer(
    "what does the refund policy say?",
    rag_docs=3,        # how many hybrid hits feed the LLM (default 3)
    rag_words=1500,    # words of text taken from each hit (default 1500)
    # instruction="Answer in German, cite the exact titles you used",  # optional
)
```


### Search tuning

Retrieval (search and RAG grounding) runs through the platform's tuned
pipeline: global defaults → your index's saved **Search Tuning** (Control
Panel → Index Settings → Search Tuning: semantic↔lexical balance, field
weights, minimum match, search mode, vector candidate pool, content quality
boost) → optional per-call overrides via `tuning`:

```
tuning={"search_mode": "keywords_required", "fw_title": 0.2,
        "mm": "strict", "vector_topk": 500, "quality_boost": 0.3}
```

Defaults match the platform's PHP configuration exactly — customize in the
Control Panel once, or per call from code.

#### Fresh Results Bias

Rank newer documents higher without hiding anything older. Every score is
multiplied by a recency curve on `creation_date` — full weight for a document
published today, about half after a year:

```python
store.similarity_search_with_score("solar inverter warranty", fresh_bias=True)
client.hybrid_search(index, query, fresh_bias=True)
client.ai_answer(index, question, tuning={"fresh_bias": 1})
```

It **re-orders and never filters**: the hit count is identical either way,
nothing old becomes unreachable, and a document with no `creation_date` simply
keeps its place instead of being pushed to the bottom. It applies to all three
retrieval shapes — vector-only, keyword-only and the fused hybrid ranking —
because the boost wraps the final score rather than one half of it. Off by
default.

This is the same control visitors get as the **Fresh** toggle beside the sort
options on the hosted Opensolr search page, so a query behaves identically here
and there.

> `fresh_bias` and `freshness_boost` are two different knobs and the names
> invite confusion. `freshness_boost` is a hard window in **days** — anything
> older is filtered out and the hit count drops. `fresh_bias` filters nothing.

## How it's tested

Every release is validated against **live Opensolr infrastructure** — no mocks:

- **Unit tests** (offline): location aliases, filter→fq mapping, query building, escaping.
- **End-to-end suite**: the full write path through the async Data Ingestion
  queue (queued → server-side enrichment → searchable), semantic / hybrid /
  lexical retrieval, metadata round-trip, filters, id round-trip (your ids
  and the Solr `md5(uri)` ids), deletes by id and by query.
- **Real-corpus validation**: searches run against a 340-document replica of
  opensolr.com's own production search index. Verified: pure-semantic hits
  with zero keyword overlap ("how do I get my data back after a disaster" →
  backup &amp; restore docs), cross-lingual queries (Romanian query → English
  content), exact-term surfacing in hybrid mode, all four hybrid modes, and
  the full alpha range 0 → 1.
- **PDF ingestion**: a real PDF ingested via `rtf:true` — server-side text
  extraction (13k+ chars), automatic content-type detection, then retrieved
  with a purely semantic query against its contents.
- **Grounded RAG answers**: `ai_answer` verified end-to-end — a question answerable only from the ingested PDF returns the correct answer, sourced from the PDF's extracted text via hybrid retrieval.

```bash
pytest tests/unit_tests
OPENSOLR_EMAIL=... OPENSOLR_API_KEY=... OPENSOLR_INDEX=... pytest tests/integration_tests
```

MIT license.
