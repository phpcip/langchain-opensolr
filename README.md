# langchain-opensolr

LangChain integration for [Opensolr](https://opensolr.com) — managed Apache Solr
with **server-side embeddings** and native **hybrid (BM25 + kNN) search**.

**See it live (real news index, hybrid + AI answer):** https://search.opensolr.com/news__dense?q=how+am+I+supposed+to+save+money%3F

**Product page:** [opensolr.com/langchain](https://opensolr.com/langchain) ·
**Platform:** [opensolr.com](https://opensolr.com) (managed Solr hosting since 2011 —
free forever, no card)

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
and grounded answers work the moment you connect. You also get the write path:
create your own index on the account, ingest into it and query it. Deletion is not
available on this shared key: no index here can be deleted or reconfigured by hand.
Whatever you create is removed automatically after 3 days.

Know what you are working with:

- **Anything you create there is deleted after 3 days.** Automatically, without warning
  or export. That includes indexes you created and every document in them.
- **The account is shared with everyone reading this.** Your index is visible to them
  and they can add documents to it, as you can to theirs. Nobody can delete or
  reconfigure an index here (that is switched off for this key), but never put anything
  real, private or client-owned in it.
- **The limits are per index, and deliberately small.** 200 MB of bandwidth and 50 MB
  of disk per index. Bandwidth is the one you will hit first: it covers a demo, a
  tutorial and a proof of concept, and it will not carry an application.

When you want an index that is private, yours and still there next week, get your own
key — [free forever, no card](https://opensolr.com/register) — and change the two
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

## Search operators

The query string understands the operators people already expect from a search box. They work
in every mode — keyword, hybrid and pure vector.

| Operator | Meaning | Example |
| --- | --- | --- |
| `"word1 word2"` | Phrase — those words together, in that order | `"machine learning"` |
| `+word` | Required — every result must contain it | `+laptop 15 inch gaming` |
| `+"word1 word2"` | Required phrase | `+"13 inch"` |
| `-word` | Excluded — drop any document containing it | `laptop -refurbished` |
| `-"word1 word2"` | Excluded phrase | `-"open box"` |

They compose: `+laptop +"13 inch" -refurbished` returns only 13-inch laptops and never a
refurbished one.

A prefixed term (`+` or `-`, word or phrase) becomes a **filter**, applied to the whole result
set. That matters as soon as a vector is involved: the semantic side of a hybrid search has no
concept of negation, so left as query text `-refurbished` would actually pull refurbished
listings *towards* the top rather than removing them. As a filter it binds every document,
whichever side of the search found it, and the exclusion is absolute.

An unprefixed phrase (`"machine learning"` with no `+` in front) is a keyword-side relevance
signal rather than a filter — use `+"machine learning"` when you need it enforced.

`+` and `-` only count at the start of a word, so `e-mail`, `covid-19` and `1+1` are searched
for literally.

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

- Vector indexes live in the **vector regions**: the Opensolr servers that run the
  web crawler, whatever their Solr version. The list is live, never written in the
  package: `client.vector_regions()` returns the regions that take new indexes right now
  (environment id, country, Solr version). `location=` takes an environment id from
  that list, or a short form such as `us`, `de` or `fi`, matched against the live
  list by environment id or country. **Additional dedicated regions can be deployed
  on request** (paid add-on): [support@opensolr.com](mailto:support@opensolr.com).
- A free Opensolr account (free forever, no card) includes an AI quota that
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
zero AI quota. It runs on any Opensolr index that has `title`, `description` and
`text` fields, including indexes outside the vector regions and older Solr
versions (see [Index requirements](#index-requirements)).

## Search by image

Search your index with a **photo** instead of a text query. Opensolr reads the
picture three ways — visual labels (what it depicts), OCR text (words printed on
it), and any barcode / QR code — and turns that into an ordinary search. Nothing
new is stored in Solr; the picture simply becomes words.

```python
# What the picture reads as (labels, OCR text, barcodes) — no search yet:
read = vs.image_to_words("photo.jpg")     # path, bytes, or base64
# {'text': 'red running shoe', 'mode': 'clip',
#  'labels': ['running shoe', 'sneaker'], 'codes': ['0123456789012']}

# Search the index with the picture:
docs = vs.search_by_image("photo.jpg", k=4)          # engine picks the best reading
docs = vs.search_by_image("photo.jpg", using="meaning")  # visual labels
docs = vs.search_by_image("photo.jpg", using="text")     # only OCR text
docs = vs.search_by_image("photo.jpg", using="code")     # exact barcode / QR match
docs = vs.search_by_image("photo.jpg", using="all")      # labels + OCR + codes
```

`search_by_image` accepts the same tuning as `similarity_search` — `hybrid`,
`mode`, `alpha`, `fresh_bias`, `filter` — so an image query runs through the
exact hybrid pipeline a text query does.

## Local files: PDFs page by page, scans through OCR

Read a file or a whole folder into LangChain `Document`s with `OpensolrLoader` —
one Document per PDF page (`metadata["page"]`, from 1), one per any other file:

```python
from langchain_opensolr import OpensolrLoader

loader = OpensolrLoader("./contracts", index="mysite__dense",
                        email="you@example.com", api_key="YOUR_OPENSOLR_API_KEY",
                        pattern="*.pdf")
docs = loader.load()          # or loader.lazy_load()
# metadata: source, page, is_ocr, language, content_type
```

Or index them straight into the store — no splitting, no loader, PDFs page by page:

```python
result = vs.add_files("./contracts", pattern="*.pdf", metadata={"team": "legal"}, wait=True)
# {"documents": [{path, source, uri, id}], "pending": [], "failed": [], "jobs": [...]}
```

- Each file is read on Opensolr's servers: its text, the text in its pictures (OCR,
  more than 100 languages), a short description of the pictures without text. Word,
  RTF, OpenDocument, text, HTML and picture files (scans) are one document each; a
  **PDF is one document per page**.
- The bytes travel once: a file the account already sent (same content) is not
  uploaded again. Pictures read by OCR count 0.5 of an AI request each; a file read
  before is free.
- Files still being read when `timeout` runs out are listed in `loader.pending` (or
  `result["pending"]`): run again to get them, nothing is sent twice. Files that could
  not be read are in `loader.failed` / `result["failed"]`.
- An indexed file's `uri` is `https://ingest.opensolr.com/<index>/<source>`, where
  `source` is its path from the folder you sent (`metadata["source"]` on every hit);
  sending it again replaces it. Its date is the file's modification time.

## PDF pages in search results

On an index where PDFs are stored page by page (files indexed as above, PDFs sent to
Data Ingestion with `rtf`, PDFs found by the Web Crawler), a search returns each PDF
**once**, as its best page. Its metadata carries `page` (the page number),
`page_url` (opens the PDF at that page, `...pdf#page=14`), `document_id`,
`more_pages` (its other matching pages, each with `page` and `page_url`) and
`matching_pages`, and its `uri`; any other hit keeps exactly the metadata it was sent with. `group_pages=False` returns the pages
as separate results.

## Searching a period

`date_from` and `date_to` (`YYYY-MM-DD`, both included, either one can be left out)
keep only the documents dated in that period (`creation_date`):

```python
docs = vs.similarity_search("rate decision", k=5, hybrid=True,
                            date_from="2026-01-01", date_to="2026-03-31")
```

They combine with every other option — filters, `fresh_bias`, hybrid or lexical.

## Index requirements

Everything this package does runs on an Opensolr index that sits in a **vector
region** and carries the **Opensolr vector schema** — the configuration every
Opensolr vector index runs. An index created through the package (`create_index`,
`create_if_missing`) has both. An index made elsewhere may not: a server outside
the vector regions, or a schema of its own. A vector region is an Opensolr server
that runs the web crawler, whatever its Solr version; the regions that take new
indexes come from the live `vector_regions` list, never from a list in the package.

**The check is automatic.** The first time the package touches an index, it asks
the platform once whether that index can do what is asked: one request per index
per client, kept for the client's life, never one per search. If the check itself
cannot run (a network error, the platform busy), nothing is blocked: the
operation runs exactly as it always did, and the check is tried again after 5
minutes. When an index differs from the Opensolr vector schema, a warning is logged
once, on the `opensolr` logger, with the reasons and the fix.

What the check gates — only what cannot work otherwise:

| Capability | Needed for | When it is missing |
|---|---|---|
| `ingestion` | Data Ingestion: adding documents and indexing files | The index is outside the vector regions: refused before a document or a byte is sent |
| `vector_search` | Vector and hybrid search | The index has no vector field of the right size: refused before any embedding is paid for |
| `pdf_pages` | Grouping the pages of a PDF in search results | Results come back ungrouped, as before |
| `keyword_search` | Keyword (lexical) search | Never blocked |

**The errors say exactly what is going on.** A blocked operation raises
`OpensolrSchemaError` (a subclass of `OpensolrError`) with `.index`, `.capability`
and `.report` (the full check), and a message with every reason and the fix:

```
Index 'shop_it' cannot take documents or files through Data Ingestion. Index shop_it
is on IT-SOLR-9-6 (Milano, Italy), which is not a vector region: Data Ingestion,
server-side embeddings and hybrid search run only on the servers of a vector region
(DE-SOLR-9 (Germany, Solr 9.00), FINLAND9 (Finland, Solr 9.60), ...). This index
cannot be made ready: create a new index in one of the vector regions (...), for
example with create_index, and send your documents there.
```

When Solr itself refuses a request, `OpensolrSolrError` carries Solr's own reason
and the index name; it is still an `httpx.HTTPStatusError`, so code that caught
that before keeps working.

```python
from langchain_opensolr import OpensolrVectorStore, OpensolrSchemaError, OpensolrSolrError

try:
    vs.add_files("./contracts")
except OpensolrSchemaError as e:
    print(e.capability, e.report["fix"]["action"])   # ingestion new_index
```

**See the details, and fix what can be fixed:**

```python
vs.check_schema()   # ready, capabilities, server, Solr version, missing / different fields, problems, fix
vs.apply_schema(force=False)   # installs the Opensolr vector schema; force=True when the index holds documents
```

`apply_schema` exists so an index of a vector region whose configuration was
changed, or never set, can take everything this package does without being
recreated: it installs the same configuration every Opensolr vector index runs,
and **replaces the index configuration**. An index that already holds documents
needs `force=True`; those documents keep the fields they were written with and
must be sent again. An index outside the vector regions cannot take it — create
a new index in a region from `vector_regions` and send your documents there. The
package never installs the schema on its own.


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

- **Unit tests** (offline): location matching against the live region list, filter→fq mapping, query building, escaping.
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
