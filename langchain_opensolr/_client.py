"""Thin REST client for the Opensolr platform APIs.

Two base URLs, by platform design:
- Management API (index list/info/create): https://opensolr.com/solr_manager/api
- AI API (embed, batch_embed, embed_and_search, ai_summary): https://api.opensolr.com/solr_manager/api

Direct Solr access (select/update) goes to the index's own host, resolved via
``get_core_info`` (``connection_url`` + HTTP basic auth).
"""

from __future__ import annotations

import re
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote

import httpx

MGMT_BASE = "https://opensolr.com/solr_manager/api"
AI_BASE = "https://api.opensolr.com/solr_manager/api"

#: Convenience aliases for Opensolr's vector-enabled environments. The
#: authoritative list is served live by the platform (``vector_regions``
#: endpoint) — new regions become valid automatically, and additional
#: dedicated regions can be deployed on request (paid): support@opensolr.com.
VECTOR_LOCATIONS: Dict[str, str] = {
    "us": "CHICAGO-96",
    "de": "DE-SOLR-9",
    "fi": "FINLAND9",
}


def resolve_location(location: str) -> str:
    """Map a friendly alias ("us"/"de"/"fi") to its environment identifier.

    Unknown values pass through unchanged — validity is decided against the
    live ``vector_regions`` list (or, ultimately, by the server), so newly
    deployed vector regions work without a package upgrade.
    """
    return VECTOR_LOCATIONS.get(location.strip().lower(), location.strip())

#: Server-side limit for one batch_embed call.
BATCH_EMBED_MAX = 50


# --------------------------------------------------------------------------- #
# Fresh Results Bias                                                            #
# --------------------------------------------------------------------------- #
# Recency as a MULTIPLICATIVE score boost on creation_date. Defined once for this
# package — every direct-to-Solr search path in every wrapper reads it from here —
# and copied byte for byte from the platform's own
# Hybrid_search::FRESH_BIAS_FUNCTION, so a query built client-side and the same
# query built server-side rank identically. Change it on the platform and change
# it here.
#
# 3.16e-11 is 1/(one year in ms): the multiplier is 1.0 for a document published
# today, 0.5 at a year old, 0.33 at two. max(0, ...) is a crash guard rather than a
# tuning choice — a creation_date in the future (bad metadata off a crawled page is
# common) makes ms() negative, and far enough negative the reciprocal divides by
# zero and Solr fails the whole query.
#
# It re-orders and never filters: numFound is unchanged, and a document with no
# creation_date is simply left unboosted rather than dropped. Off by default
# everywhere; a caller who turns it on has said they want recent documents to win,
# so it is applied at full strength with no hedging.
FRESH_BIAS_FUNCTION = "recip(max(0,ms(NOW,creation_date)),3.16e-11,1,1)"

#: Default Fresh Results Bias strength when a caller does not pass one.
FRESH_BIAS_WEIGHT_DEFAULT = 0.5


def fresh_bias_function(weight: Optional[float] = None) -> str:
    """Build the recency function for a 0.0-1.0 ``weight``.

    Mirrors ``Hybrid_search::fresh_bias_function()`` on opensolr.com, the Drupal module and
    the WordPress plugin. ``recip(ms, c, 1, 1)`` halves at ``ms = 1/c``, so the weight is a
    HALF-LIFE on a geometric scale between 365 days at 0.0 and 6 hours at 1.0:

    ==========  ===========================================================
    weight      meaning
    ==========  ===========================================================
    0.0         365-day half-life: technically on, barely visible
    0.3         41 days
    0.5         9.6 days (the default)
    0.7         2.2 days
    1.0         6 hours: date all but replaces relevance
    ==========  ===========================================================

    The fixed constant this replaces behaved like 0.0, which is why Fresh looked broken on a
    news index: a 10-day-old article kept 97% of its multiplier, nowhere near enough to
    outrank a better-matching older one.
    """
    w = FRESH_BIAS_WEIGHT_DEFAULT if weight is None else float(weight)
    w = max(0.0, min(1.0, w))
    half_life_days = 365.0 * (0.25 / 365.0) ** w
    # PHP's %g writes "1.212e-9" where Python's writes "1.212e-09". Numerically identical,
    # textually not — and this string is compared byte for byte against the platform, the
    # Drupal module and the WordPress plugin by the prompt/query parity harness. Strip the
    # padding zero so all five implementations emit the same characters.
    c = ("%.4g" % (1.0 / (half_life_days * 86400000.0))).replace("e-0", "e-").replace("e+0", "e+")
    return "recip(max(0,ms(NOW,creation_date))," + c + ",1,1)"

#: The four candidate-selection modes the {!hybrid} parser understands.
#:
#: Validated rather than trusted, because the failure is silent: `mode` is interpolated into
#: the {!hybrid} local params, and the Solr plugin does not reject a value it does not know —
#: it falls back to union. So `mode="intersectoin"`, one letter off, returned 18 hits where
#: intersection returns 2, with no error anywhere (measured 2026-08-29). A caller got nine
#: times the documents they asked for and no way to notice. The wrapper layers validated this
#: already; the client, which is the package's public API, did not.
HYBRID_MODES = ("union", "keywords_required", "meaning_required", "intersection")


#: Fields a search-operator filter is evaluated against.
#:
#: The same five the platform searches, and deliberately WITHOUT their qf boosts: a filter
#: does not score, it only cuts. The field list matters because "-word" has to mean the same
#: thing the caller expects — "drop any document containing that word" — rather than "drop it
#: only if the word is in the title".
SEARCH_OPERATOR_FIELDS = "title description uri text text_t"


def parse_operators(raw: str) -> Dict[str, Any]:
    """Split a query into its plain text and its ``+``/``-`` search operators.

    WHY (2026-09-09). ``+word`` and ``-word`` are ordinary edismax syntax, so on a pure
    keyword search they have always worked. The moment a vector leg runs beside the keyword
    one they stop meaning anything: the vector leg has never heard of them, so in ``union``
    mode every document the keyword leg just excluded comes straight back in through it. And
    exclusion fails twice over, because the raw text — minus sign included — is what gets
    embedded, and a bi-encoder has no concept of negation: ``-Ruben`` reads as one more word
    of the question and pulls the results TOWARDS Ruben. Measured on a live news index, the
    query "did juventus win that match? -Ruben" returned a Ruben article at rank 1.

    So the operators stop being query syntax and become filters — see
    :func:`apply_search_operators` — which bind every document regardless of which leg found
    it. This function is only the split.

    Rules:
      * ``+``/``-`` count as operators only at the start of a token, so ``e-mail``,
        ``covid-19`` and ``1+1`` are left alone.
      * The operand is a bare word or a ``"quoted phrase"``; a phrase keeps its quotes,
        because edismax needs them to build a phrase query.
      * An unprefixed ``"quoted phrase"`` stays in the base text — it is a relevance signal
        for the keyword leg, not a filter.
      * A lone ``+`` or ``-`` with nothing usable after it is plain text.

    Returns a dict with ``base``, ``required``, ``excluded`` and ``has_ops``.
    """
    base: List[str] = []
    required: List[str] = []
    excluded: List[str] = []
    s = raw or ""
    i, n = 0, len(s)

    while i < n:
        c = s[i]

        if c.isspace():
            base.append(c)
            i += 1
            continue

        # Consume an unprefixed phrase whole, before the operator test below can mistake a
        # '-' INSIDE it ("foo -bar") for an exclusion.
        if c == '"':
            end = s.find('"', i + 1)
            if end == -1:
                base.append(s[i:])
                break
            base.append(s[i:end + 1])
            i = end + 1
            continue

        if c in "+-" and (i == 0 or s[i - 1].isspace()):
            j = i + 1
            term = None
            nxt = j
            if j < n and s[j] == '"':
                end = s.find('"', j + 1)
                if end != -1:
                    term = s[j:end + 1]
                    nxt = end + 1
            if term is None:
                k = j
                while k < n and not s[k].isspace():
                    k += 1
                term = s[j:k]
                nxt = k
            if term.strip().strip('"'):
                (required if c == "+" else excluded).append(term)
                i = nxt
                continue
            # Empty operand ('+', '-', '+""'): the sign is just a character. Re-scan from
            # i+1 so the quote branch above can pick up the '""' properly.
            base.append(c)
            i += 1
            continue

        base.append(c)
        i += 1

    return {
        "base": " ".join("".join(base).split()),
        "required": required,
        "excluded": excluded,
        "has_ops": bool(required or excluded),
    }


def apply_search_operators(
    params: Dict[str, Any],
    parsed: Dict[str, Any],
    fields: str = SEARCH_OPERATOR_FIELDS,
) -> Dict[str, Any]:
    """Turn the operators from :func:`parse_operators` into ``fq`` entries.

    Mutates ``params`` in place and returns it.

    An ``fq`` is evaluated against the whole result set, so it binds a document no matter
    which leg surfaced it — which is the entire point. The Opensolr ``{!hybrid}`` parser
    cooperates for free: it runs both sub-queries with the request's filters, and for the
    vector leg it injects each one as a kNN ``preFilter``, so the HNSW graph is walked
    already restricted to matching documents. No candidates are spent on documents a filter
    is about to drop, and ``topN`` keeps its full depth.

    Each operand is bound BY REFERENCE (``v=$reqQ0`` / ``v=$negQ0``) rather than inlined, so
    a ``}`` or a ``{!`` in the caller's text stays text and can never become a local-param
    block — verified against live Solr: an operand of ``{!join fromIndex=other}x`` filtered
    nothing and read nothing. ``mm="100%"`` makes every clause of a multi-word operand
    mandatory.
    """
    fq = params.get("fq")
    if fq is None:
        fq = []
    elif not isinstance(fq, list):
        fq = [fq]

    for i, term in enumerate(parsed.get("required") or []):
        key = "reqQ%d" % i
        params[key] = term
        fq.append('{!edismax qf="%s" mm="100%%" v=$%s}' % (fields, key))

    for i, term in enumerate(parsed.get("excluded") or []):
        key = "negQ%d" % i
        params[key] = term
        fq.append('-{!edismax qf="%s" mm="100%%" v=$%s}' % (fields, key))

    if fq:
        params["fq"] = fq

    return params


def apply_fresh_bias(params: Dict[str, Any], weight: Optional[float] = None) -> Dict[str, Any]:
    """Wrap an already-built ``params["q"]`` so the recency curve multiplies the
    FINAL score. Mutates ``params`` in place and returns it.

    Every direct-to-Solr path in this package funnels through here — pure
    ``{!knn}``, fused ``{!hybrid}``, plain edismax and ``*:*`` alike — which is the
    whole reason the wrapper is a function instead of four copies of the same three
    lines: one implementation, one function string, nothing to drift.

    ``{!boost}`` and not an edismax ``bf``, deliberately. Under ``{!hybrid}`` a
    ``bf`` reaches only the lexical sub-query, where the plugin min-max normalizes
    it and scales it by (1-alpha), and it never touches a candidate that arrived
    through the vector leg at all; on a bare ``{!knn}`` query there is no edismax to
    read a ``bf`` in the first place. Wrapping the whole query is the one form that
    works on every shape, so every shape uses it. (The platform's lexical-only path
    does use a top-level ``bf`` — there ``defType=edismax`` is set on the request
    itself, which is the one case where edismax honours it natively.)

    The inner query moves into its own parameter and is referenced by ``v=$...``
    rather than being inlined, so a ``}`` in the user's text cannot close the
    ``{!boost}`` block and leave the remainder to be parsed as query syntax.
    """
    # A document with no creation_date evaluates recip() at its MAXIMUM, 1.0 — Solr's
    # ms(NOW, <missing>) is 0 — so an undated document is scored as if published this
    # instant and floats to the top of a "newest first" ranking. Require a date instead
    # of silently promoting the ones that have none.
    fq = params.get("fq")
    date_fq = "+creation_date:[* TO *]"
    if fq is None:
        params["fq"] = date_fq
    elif isinstance(fq, list):
        if date_fq not in fq:
            params["fq"] = fq + [date_fq]
    elif fq != date_fq:
        params["fq"] = [fq, date_fq]

    params["freshBias"] = fresh_bias_function(weight)
    params["freshBiasInner"] = params["q"]
    params["q"] = "{!boost b=$freshBias v=$freshBiasInner}"
    return params


# --------------------------------------------------------------------------- #
# The canonical AI Hints prompt                                                 #
# --------------------------------------------------------------------------- #
# ONE builder, shared by every Opensolr integration. The reference implementation
# is the platform itself — the context builder and the instruction in
# solr_manager.php, plus ai_mark_fragment() / ai_excerpt() in generic_helper.php —
# and the Drupal module, the WordPress plugin, the RAG sandbox and the Laravel
# client all reproduce it. The wording is not editorial: it was scored twice on
# 2026-08-29 against vLLM over three test sets (adversarial, realistic, broad
# topical), seven candidate phrasings, three runs each. Every clause is
# load-bearing, and the ablation notes live on build_instruction() below.
#
# Because "identical everywhere" is worth nothing as a claim in a comment, the two
# builders are PURE functions — no I/O, no network, no client instance — so a
# fixture can be driven straight through them and byte-compared against the
# platform's own output. Anything changed here has to survive that comparison
# first and the measurement second.

#: The fence written around every document. build_instruction() COUNTS these to
#: tell the model how many documents it was given, so both must use this one
#: constant or the prompt announces a number the context does not contain.
DOC_FENCE = "===== DOCUMENT "

#: RAG context defaults: how many retrieved hits reach the model, and how many
#: words of each hit's text. Four documents is what the platform measured and
#: ships — this client was sending three, i.e. a smaller context than every other
#: Opensolr integration for the same question. Both overridable per call.
DEFAULT_RAG_DOCS = 4
DEFAULT_RAG_WORDS = 1500

#: Every prompt variant was scored at 0.1, so the temperature travels with the
#: prompt: raising it invalidates the measurement the wording rests on.
DEFAULT_RAG_TEMPERATURE = "0.1"

# PHP's strip_tags(), which is what the platform runs over every fragment. Solr's
# default highlighter emits <em>, but an index with its own hl.simple.pre/post, an
# hl.tag.ellipsis, or indexed content that carries markup of its own emits anything
# at all — and the cleanup here used to remove literally "<em>" and "</em>" and
# nothing else, so every other tag reached the model as prompt text. Rules mirrored
# from PHP's state machine: "<" opens a tag only when a letter, "!", "/" or "?"
# follows it (so "5 < 6" survives), a comment goes whole even when it holds a ">",
# quoted attribute values may hold a bare ">", and an unterminated tag swallows the
# rest of the string. Entities are left alone, exactly as PHP leaves them.
_TAG_RE = re.compile(
    r"""<!--.*?(?:-->|\Z)               # comment, to its terminator or end of string
      | <[!?/A-Za-z]                    # nothing else opens a tag
        (?:"[^"]*"|'[^']*'|[^>])*       # attribute values may contain '>'
        (?:>|\Z)                        # unterminated tag eats the remainder
    """,
    re.DOTALL | re.VERBOSE,
)

# Unicode-aware, NOT re.ASCII: PHP's /u modifier sets PCRE2_UCP as well as
# PCRE2_UTF, so \s and \S there already cover U+00A0, U+2007, U+3000 and the rest
# of the Unicode spaces. Checked against PHP 8.5 rather than assumed: two words
# separated by a non-breaking space count as TWO words on the platform, so an
# ASCII-only \S here would count them as one and cut a 1500-word body in a
# different place than the platform cuts it.
_WS_RUN_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\S+")

#: A fragment ends cleanly on sentence punctuation, optionally closing a quote.
_SENTENCE_END_RE = re.compile(r'[.!?…]["\')\]]?$')

#: Exactly what PHP's trim() strips, and no more. PHP's trim() is byte-based and
#: does NOT strip U+00A0, so text_t opening on a non-breaking space keeps it and
#: stays two bytes longer against the 50-byte floor below; str.strip() would
#: quietly remove it and flip that comparison on a borderline husk.
_PHP_TRIM = " \t\n\r\0\x0b"


def _strip_tags(value: Any) -> str:
    """Remove ALL tags from a highlighter fragment, PHP strip_tags() style."""
    return _TAG_RE.sub("", str(value))


def _flat(value: Any) -> str:
    """Flatten a Solr field to text: multi-valued fields arrive as lists."""
    if isinstance(value, list):
        value = " ".join(str(x) for x in value)
    return str(value or "")


def _to_float(value: Any) -> float:
    """Score as a float, tolerating the string form Solr sometimes returns.

    Anything uncastable scores 0.0 — what PHP's (float) cast yields, and what
    "no evidence this hit is strong" should mean rather than an exception
    thrown in the middle of building a prompt.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _doc_id(doc: Dict[str, Any]) -> str:
    """Id used to look this document's highlight fragments up.

    A multi-valued id field arrives as a list; the platform takes its first
    element (reset()), so a list-shaped id still finds its fragments here.
    """
    value = doc.get("id", "")
    if isinstance(value, list):
        value = value[0] if value else ""
    return "" if value is None else str(value)


def _mark_fragment(fragment: str) -> str:
    """Mark a highlighter fragment as the cut-out excerpt it is.

    Solr slices highlight fragments mid-sentence: they begin and end wherever the window fell.
    Unmarked, they read to a model as complete sentences, and a small model stitches a fragment
    onto the paragraph that follows and treats the join as one statement. An ellipsis on the open
    end says plainly that text is missing — the same convention a search UI shows the reader.
    Kept identical to ai_mark_fragment() on the Opensolr platform and the equivalents in the
    Drupal module, the WordPress plugin and the Laravel package (2026-08-29).
    """
    fragment = _WS_RUN_RE.sub(" ", str(fragment)).strip(_PHP_TRIM)
    if not fragment:
        return ""
    # The platform tests the opening character with [\p{Lu}\p{N}]: ANY uppercase
    # letter and ANY digit, in any script. Python's re has no \p, and the ASCII-ish
    # class this used to carry ([A-ZÀ-Þ0-9]) silently failed on Greek,
    # Cyrillic, Hebrew, Arabic-Indic digits and everything else, so well-formed
    # fragments in those scripts were all marked as cut open when they were not.
    # str.isupper() and str.isnumeric() ARE those Unicode properties, at no cost:
    # the `regex` module does support \p, but it is not a dependency of any of
    # these four packages and is not worth becoming one for two character classes.
    head = fragment[1:2] if fragment[0] in '"\'([' else fragment[0]
    if not (head.isupper() or head.isnumeric()):
        fragment = "... " + fragment
    # Ends mid-sentence when there is no closing punctuation.
    if not _SENTENCE_END_RE.search(fragment):
        fragment = fragment + " ..."
    return fragment


def _excerpt(text: str, max_words: int) -> str:
    """Cut ``text`` after its ``max_words``-th word.

    It CUTS the original string and never rebuilds it from the matched words:
    " ".join(text.split()) throws away every newline, blank line and run of
    indentation, so a page reaches the model as one unbroken line with its
    paragraphs, lists and code blocks flattened. Structure is part of the meaning.
    """
    text = str(text or "")
    if not text or max_words <= 0:
        return ""
    ends = [m.end() for m in _WORD_RE.finditer(text)]
    if len(ends) <= max_words:
        return text
    return text[: ends[max_words - 1]]


def build_context(
    docs: List[Dict[str, Any]],
    hl: Dict[str, Any],
    top_n: int = DEFAULT_RAG_DOCS,
    max_words: int = DEFAULT_RAG_WORDS,
) -> str:
    """Assemble retrieved hits into the canonical LLM context.

    Pure by design: hand it ``docs`` (hits in retrieval order) and ``hl``
    (document id -> {field: [fragments]}) and it returns the string, so this port
    can be proved against the platform's own output with no account, no network
    and no client instance.

    Selection: only the first ``top_n`` hits are ever considered, and among those,
    anything scoring below half of the best score is dropped. Retrieval always
    returns a full page of hits, so a narrow question arrives with one good match
    and three unrelated articles; the model then sees that most of its context does
    not answer the question and hedges ("the content does not mention X,
    however..."). A hit carrying no score at all is KEPT — only provable weakness is
    filtered (platform rule, 2026-08-25).

    Layout, per kept document: the fence, then title, then description, then the
    matched highlight fragments, then the body. Title and description are written
    even when empty, because the model reads a fixed head per document and a
    skipped line would shift the body up into the description's slot.
    """
    docs = docs or []
    hl = hl or {}
    context = ""

    top_score = 0.0
    for doc in docs[:top_n]:
        if isinstance(doc, dict) and doc.get("score") is not None:
            top_score = max(top_score, _to_float(doc["score"]))

    for doc in docs[:top_n]:
        if not isinstance(doc, dict):
            continue
        if (top_score > 0 and doc.get("score") is not None
                and _to_float(doc["score"]) < top_score * 0.5):
            continue
        # Numbering follows the text itself — the fences already written — so the
        # count in the instruction can never drift from the context it describes.
        doc_n = str(context.count(DOC_FENCE) + 1)
        context += DOC_FENCE + doc_n + " =====\n"
        context += _flat(doc.get("title")) + "\n"
        context += _flat(doc.get("description")) + "\n"

        # Solr's own highlight fragments, ahead of the long body: these are the
        # parts the query actually matched, so the model reads the focused text
        # while it still has full attention on this document. Measured on the
        # platform, they are the difference between a one-line answer and a real
        # one. Their markup belongs to a search UI, not to a prompt.
        doc_hl = hl.get(_doc_id(doc)) or {}
        if not isinstance(doc_hl, dict):
            doc_hl = {}
        fragments: List[str] = []
        for field in ("title", "description", "text"):
            raw = doc_hl.get(field) or []
            if not isinstance(raw, list):
                raw = [raw]  # a single fragment can arrive unwrapped
            for snippet in raw:
                snippet = _mark_fragment(_strip_tags(snippet))
                if snippet:
                    fragments.append(snippet)
        if fragments:
            context += "MOST RELEVANT EXCERPTS:\n"
            for snippet in fragments:
                context += "- " + snippet + "\n"
            # Blank line closes the list, so the full text below reads as its own
            # paragraph and not as a continuation of the last fragment.
            context += "\n"

        # text_t is CONCATENATED with text, never substituted for it. On some sites
        # the JSON-LD field carries real article content; on others it holds only
        # scaffolding ("Is Accessible For Free: False", "Css Selector: ..."), and
        # preferring it there leaves the model with metadata and no article. Keeping
        # both never loses content, and the floor drops the empty husks. That floor
        # is measured in BYTES, like the platform's strlen(): a 40-character Greek or
        # Romanian husk is over 50 bytes there and under 50 characters here, so
        # counting characters would keep husks the platform drops.
        text_t = _flat(doc.get("text_t")).strip(_PHP_TRIM)
        if len(text_t.encode("utf-8")) > 50:
            body = text_t + "\n" + _flat(doc.get("text"))
        else:
            body = _flat(doc.get("text"))
        context += _excerpt(body, max_words) + "\n"
        context += "===== END OF DOCUMENT " + doc_n + " =====\n\n"
    return context


def build_instruction(context: str, query: str) -> str:
    """Wrap a context into the whole prompt — the single string sent to the model.

    Documents first, question last. What the ablations showed, so that none of it
    gets tidied up later:

    * ending with the question is load-bearing — removing the trailing "Question:"
      slot collapsed the adversarial score from 7/7 to 3/7;
    * repeating the refusal option at the END is catastrophic (7/7 -> 3/7): whatever
      sits in the last slot before generation is what the model reaches for, so the
      escape hatch belongs in the middle;
    * "do not name documents, do not comment on the ones you did not use" is what
      removed the reported bug — right answers prefixed with "Based on Document 4,
      there is no mention of...", a denial wrapped around a correct answer;
    * the two banned openings are spelled out, against the usual "never name a word
      you do not want" rule, because it was measured both ways: without them one
      answer in five still opened with "According to the documents,";
    * "find ALL of the N documents that are relevant" is what makes it synthesise
      across several articles instead of answering from whichever one it hit first
      (6/6 topical at 3.8 of 4 documents drawn on, against 5-6/6 at 3.2-3.5 before);
    * the refusal opening is PINNED to "There is no information about" so negative
      cases stay measurable — left free, the model wandered over five phrasings;
    * the closing "only if not one of them is about the question" clause is what
      keeps "use every relevant document" from turning into invention.

    The document count is read back out of the text, so it always matches what the
    model can actually see, and never drops below 1 — an empty context still has to
    read as a grammatical prompt rather than "Those were the 0 documents.".
    """
    context = str(context)
    n = str(max(1, context.count(DOC_FENCE)))
    return (
        context + '\n\n'
        + 'Those were the ' + n + ' documents.\n\n'
        + 'Now answer the question below using only facts stated in those documents. '
        + 'Find ALL of the ' + n + ' documents that are relevant to the question, even '
        + 'when the question describes the subject in completely different words than a '
        + 'document does, and answer the question based on those. Where more than one of '
        + 'them bears on the question, combine what each one adds into a single answer.\n'
        + 'Write the answer itself, formatted in Markdown for reading. Begin with one '
        + 'sentence that answers the question directly. Whenever the answer covers more '
        + 'than one development, position or fact — which is most of the time — set '
        + 'the detail out as a Markdown list, each item on its own line opening with a '
        + 'bold lead-in that names it. Keep it as prose only if there is genuinely just '
        + 'one thing to say. Be thorough: cover every distinct point the documents offer '
        + 'that bears on the question, with the concrete details — the people, places, '
        + 'numbers, dates and named events involved. Do not stop at the first thing you can '
        + 'say. Never invent generic headings such as "Overview", "Key Points" or '
        + '"Summary". Never begin with "Based on" or "According to", and never end with '
        + 'a sentence about the documents or the context. Do not name documents, do not say '
        + 'which ones you used, and do not comment on the ones you did not use.\n'
        + 'Only if not one of the ' + n + ' documents is about the question, reply with '
        + 'a single sentence that starts "There is no information about" and then names '
        + 'what they cover instead.\n\n'
        + 'Question: ' + str(query) + '\n'
        + 'Answer:'
    )


# --------------------------------------------------------------------------- #
# PDF pages and date ranges                                                     #
# --------------------------------------------------------------------------- #
# A PDF indexed from its file (Data Ingestion with rtf, the web crawler) is one Solr
# document per page: page_i is the page number, uri_id_s the id of the PDF it belongs to.
PAGE_FIELD = "page_i"
PARENT_FIELD = "uri_id_s"

#: Other matching pages returned under each PDF when the results are grouped.
GROUP_EXPAND_ROWS = 4

#: Most pages one read_document call returns.
READ_MAX_PAGES = 500

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_META_FIELD_RE = re.compile(r"[^a-z0-9_]+")


def date_range_filter(date_from: Optional[str] = None, date_to: Optional[str] = None) -> Optional[str]:
    """Solr filter on ``creation_date`` for a period, both ends included (``YYYY-MM-DD``).

    Either end may be left out. Returns ``None`` when neither is given.
    """
    import datetime

    bounds = []
    for name, value, clock in (("date_from", date_from, "T00:00:00Z"), ("date_to", date_to, "T23:59:59Z")):
        if value is None or str(value).strip() == "":
            bounds.append("*")
            continue
        text = str(value).strip()
        if not _DATE_RE.match(text):
            raise ValueError(f"{name} must be YYYY-MM-DD, got {value!r}")
        try:
            datetime.date.fromisoformat(text)
        except ValueError:
            raise ValueError(f"{name} is not a real date: {value!r}") from None
        bounds.append(text + clock)
    if bounds == ["*", "*"]:
        return None
    if "*" not in bounds and bounds[0] > bounds[1]:
        raise ValueError(f"date_from ({date_from}) is after date_to ({date_to})")
    return f"creation_date:[{bounds[0]} TO {bounds[1]}]"


def add_filters(params: Dict[str, Any], *filters: Optional[str]) -> Dict[str, Any]:
    """Append filter queries to ``params["fq"]``, keeping the ones already there."""
    extra = [f for f in filters if f]
    if not extra:
        return params
    existing = params.get("fq")
    if existing is None:
        params["fq"] = extra
    elif isinstance(existing, list):
        params["fq"] = existing + extra
    else:
        params["fq"] = [existing] + extra
    return params


def apply_page_grouping(params: Dict[str, Any]) -> Dict[str, Any]:
    """One result per PDF (its best page), with its other matching pages in Solr's expanded section.

    Documents that are not pages of a PDF stay separate results (``nullPolicy=expand``). The
    collapse sort is passed by reference so {!hybrid} ranks every page before the grouping.
    """
    add_filters(params, "{!collapse field=" + PARENT_FIELD + " nullPolicy=expand sort=$qgsort}")
    params["qgsort"] = "score desc," + PAGE_FIELD + " asc"
    params["expand"] = "true"
    params["expand.rows"] = GROUP_EXPAND_ROWS
    params["expand.sort"] = "score desc," + PAGE_FIELD + " asc"
    return params


def page_url(uri: str, page: Optional[int]) -> str:
    """The link that opens a PDF at one page."""
    if uri and page and "#" not in uri:
        return f"{uri}#page={int(page)}"
    return uri


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def page_info(solr_doc: Dict[str, Any], expanded: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Where a hit lives: ``uri``; for a page of a PDF also ``page``, ``page_url``, ``document_id``
    and, when the results were grouped, the other matching ``more_pages`` of that PDF."""
    uri = str(_first(solr_doc.get("uri")) or "")
    out: Dict[str, Any] = {"uri": uri}
    try:
        page = int(_first(solr_doc.get(PAGE_FIELD)) or 0)
    except (TypeError, ValueError):
        page = 0
    if page < 1:
        return out
    parent = str(_first(solr_doc.get(PARENT_FIELD)) or "")
    out.update(page=page, page_url=page_url(uri, page), document_id=parent)
    group = (expanded or {}).get(parent) if parent else None
    if isinstance(group, dict) and group.get("docs"):
        more = []
        for d in group["docs"]:
            try:
                n = int(_first(d.get(PAGE_FIELD)) or 0)
            except (TypeError, ValueError):
                continue
            if n > 0:
                more.append({"page": n, "page_url": page_url(str(_first(d.get("uri")) or uri), n)})
        out["more_pages"] = more
        out["matching_pages"] = 1 + int(group.get("numFound") or len(more))
    return out


def list_files(path: Union[str, Path], pattern: str = "*", recursive: bool = True) -> List[Path]:
    """The files under ``path`` (or ``path`` itself when it is a file), hidden ones left out."""
    root = Path(path).expanduser()
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"No such file or folder: {path}")
    found = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(
        f for f in found
        if f.is_file() and not any(part.startswith(".") for part in f.relative_to(root).parts)
    )


def file_source(root: Union[str, Path], file: Union[str, Path]) -> str:
    """A file's name in the index: its path from the folder that was sent, the folder's name first."""
    root_p = Path(root).expanduser().resolve()
    file_p = Path(file).expanduser().resolve()
    if root_p.is_file() or root_p == file_p:
        return file_p.name
    return (Path(root_p.name) / file_p.relative_to(root_p)).as_posix()


class OpensolrError(RuntimeError):
    """Raised when an Opensolr API call fails."""


class OpensolrClient:
    """Authenticated client for Opensolr management + AI endpoints.

    Args:
        email: Opensolr account email.
        api_key: Opensolr API key (Account > API in the control panel).
        timeout: Per-request timeout in seconds. Embedding calls run on GPU
            infrastructure and are usually fast, but cold starts happen.
    """

    def __init__(self, email: str, api_key: str, timeout: float = 120.0) -> None:
        self.email = email
        self.api_key = api_key
        self._http = httpx.Client(timeout=timeout, follow_redirects=True)
        self._core_info_cache: Dict[str, Dict[str, Any]] = {}
        # indexes whose schema has no PDF page fields: searched and read without them
        self._flat_indexes: set = set()

    # ------------------------------------------------------------------ #
    # low level                                                          #
    # ------------------------------------------------------------------ #

    def _auth_params(self) -> Dict[str, str]:
        return {"email": self.email, "api_key": self.api_key}

    #: How many times a request is retried when the connection dies before a response arrives.
    #:
    #: Long-lived keep-alive sockets get dropped by the far end now and then — measured at
    #: roughly 1 request in 60 against production. Without a retry that surfaces as a raw
    #: httpx.RemoteProtocolError out of any public method, and the sharpest edge is
    #: ingest(wait=True): its poll loop would abort a multi-minute wait on one dropped socket,
    #: even though the ingestion completed fine server-side, leaving the caller with an
    #: exception and no idea whether their documents landed (2026-08-29).
    TRANSPORT_RETRIES = 2

    def _request(self, base: str, method: str, params: Dict[str, Any]) -> Any:
        url = f"{base}/{method}"
        data = {**self._auth_params(), **params}
        resp = self._post_with_retry(url=url, data=data, label=method)
        return self._answer(method, resp)

    def _answer(self, label: str, resp: "httpx.Response") -> Any:
        if resp.status_code >= 500:
            raise OpensolrError(f"{label}: HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            body = resp.json()
        except json.JSONDecodeError as exc:
            raise OpensolrError(f"{label}: non-JSON response: {resp.text[:200]}") from exc
        if isinstance(body, dict) and body.get("status") is False:
            raise OpensolrError(f"{label}: {body.get('msg', body)}")
        return body

    def _post_with_retry(self, *, url: str, label: str, **kwargs: Any) -> "httpx.Response":
        """POST, retrying only when the connection failed BEFORE a response was produced.

        httpx.TransportError covers exactly that family — connect failures, read timeouts and
        the server hanging up mid-flight. A request that died there was never answered, so
        re-sending it cannot duplicate work. Anything that DID come back, including a 4xx or a
        5xx, is returned untouched: a 500 from the application is a real answer about a real
        attempt, and retrying it would hammer a struggling server and could repeat a write.
        """
        last: Optional[Exception] = None
        for attempt in range(self.TRANSPORT_RETRIES + 1):
            try:
                return self._http.post(url, **kwargs)
            except httpx.TransportError as exc:
                last = exc
                if attempt == self.TRANSPORT_RETRIES:
                    break
                # Short, growing pause: a dropped keep-alive reconnects immediately, while a
                # server briefly refusing connections needs a breath before the next try.
                time.sleep(0.25 * (attempt + 1))
        raise OpensolrError(
            f"{label}: connection failed after {self.TRANSPORT_RETRIES + 1} attempts: {last}"
        ) from last

    def mgmt(self, method: str, **params: Any) -> Any:
        return self._request(MGMT_BASE, method, params)

    def ai(self, method: str, **params: Any) -> Any:
        return self._request(AI_BASE, method, params)

    # ------------------------------------------------------------------ #
    # management                                                         #
    # ------------------------------------------------------------------ #

    def get_index_list(self) -> List[Dict[str, str]]:
        return self.mgmt("get_index_list")

    def get_core_info(self, index: str, refresh: bool = False) -> Dict[str, Any]:
        """Resolve an index's Solr endpoint + HTTP auth. Cached per client."""
        if not refresh and index in self._core_info_cache:
            return self._core_info_cache[index]
        body = self.mgmt("get_core_info", core_name=index)
        msg = body.get("msg") if isinstance(body, dict) else None
        if not isinstance(msg, dict) or "info" not in msg:
            raise OpensolrError(f"get_core_info({index}): unexpected response: {str(body)[:200]}")
        info = msg["info"]
        self._core_info_cache[index] = info
        return info

    def vector_regions(self) -> List[Dict[str, str]]:
        """Live list of vector-enabled environments (Solr 9.x + knn_vector +
        hybrid parser): ``[{environment, country, solr_version}, ...]``.

        Cached per client. Additional dedicated regions can be deployed on
        request (paid) — contact support@opensolr.com.
        """
        if not hasattr(self, "_vector_regions_cache"):
            body = self.mgmt("vector_regions")
            self._vector_regions_cache = body if isinstance(body, list) else []
        return self._vector_regions_cache

    def create_index(self, index: str, location: str = "us") -> Dict[str, Any]:
        """Create a vector-enabled index in a vector location.

        ``location`` is an alias ("us", "de", "fi") or a raw Opensolr
        environment identifier. Validated against the live ``vector_regions``
        list when reachable; otherwise the server has the final word.
        """
        env = resolve_location(location)
        try:
            live = {r["environment"] for r in self.vector_regions()}
        except OpensolrError:
            live = set(VECTOR_LOCATIONS.values())  # offline fallback
        if live and env not in live:
            raise ValueError(
                f"{location!r} is not a vector-enabled Opensolr location. "
                f"Currently available: {sorted(live)}. Additional regions can "
                f"be deployed on request — contact support@opensolr.com."
            )
        # create_index reads its params from the query string (GET) server-side
        resp = self._post_with_retry(label="embed", url=
            f"{MGMT_BASE}/create_index",
            params={"index_name": index, "core_type": "generic", "server_country": env},
            data=self._auth_params(),
        )
        body = resp.json()
        if isinstance(body, dict) and body.get("status") is False:
            raise OpensolrError(f"create_index: {body.get('msg', body)}")
        return body

    # ------------------------------------------------------------------ #
    # AI                                                                 #
    # ------------------------------------------------------------------ #

    def image_to_text(self, index: str, image_b64: str, top_k: int = 8) -> Dict[str, Any]:
        """Turn an image into search words via the Opensolr image_to_text API.

        The image is read three ways server-side and the answer carries all of them:
        ``text`` (the words to search with — OCR text when the picture is mostly text,
        otherwise the top visual labels), ``mode`` ("clip" or "ocr"), ``labels`` (the
        CLIP visual labels, present even in OCR mode), and ``codes`` (barcodes / QR codes).
        No image vector is stored: the picture simply becomes words the normal search runs.
        """
        return self.ai("image_to_text", index_name=index, image=image_b64, top_k=top_k)

    def embed(self, index: str, text: str, is_query: bool = False) -> List[float]:
        body = self.ai(
            "embed", index_name=index, payload=text, is_query="1" if is_query else "0"
        )
        if not isinstance(body, list) or not body:
            raise OpensolrError(f"embed: unexpected response: {str(body)[:200]}")
        return body

    def batch_embed(self, index: str, texts: List[str]) -> List[List[float]]:
        """Embed many texts. Chunks transparently at the server's batch limit."""
        out: List[List[float]] = []
        for i in range(0, len(texts), BATCH_EMBED_MAX):
            chunk = texts[i : i + BATCH_EMBED_MAX]
            resp = self._post_with_retry(label="batch_embed", url=
                f"{AI_BASE}/batch_embed",
                json={
                    **self._auth_params(),
                    "index_name": index,
                    "payloads": chunk,
                },
            )
            try:
                body = resp.json()
            except json.JSONDecodeError as exc:
                raise OpensolrError(f"batch_embed: non-JSON response: {resp.text[:200]}") from exc
            if isinstance(body, dict) and body.get("status") is False:
                raise OpensolrError(f"batch_embed: {body.get('msg', body)}")
            embeddings = body.get("embeddings") if isinstance(body, dict) else None
            if not isinstance(embeddings, list) or len(embeddings) != len(chunk):
                raise OpensolrError(f"batch_embed: unexpected response: {str(body)[:200]}")
            out.extend(embeddings)
        return out

    def ingest(self, index: str, documents: List[Dict[str, Any]], wait: bool = False, timeout: float = 180.0) -> Dict[str, Any]:
        """Queue documents through the Opensolr Data Ingestion API (async).

        Embeddings, sentiment, language detection, and all crawler-identical
        derived fields are computed server-side. Max 50 documents per call.
        Returns ``{status, msg, job_id, total_docs, doc_ids}``. With
        ``wait=True``, polls ``ingest_status`` until the job completes
        (the queue is processed every minute).
        """
        resp = self._post_with_retry(label="ingest", url=
            f"{AI_BASE}/ingest",
            json={**self._auth_params(), "core_name": index, "documents": documents},
        )
        try:
            body = resp.json()
        except json.JSONDecodeError as exc:
            raise OpensolrError(f"ingest: non-JSON response: {resp.text[:200]}") from exc
        if isinstance(body, dict) and body.get("status") is False:
            raise OpensolrError(f"ingest: {body.get('msg', body)}: {body.get('errors', '')}")
        if wait and body.get("job_id"):
            import time
            deadline = time.time() + timeout
            while time.time() < deadline:
                st = self.ingest_status(body["job_id"])
                job = st.get("job", {}) if isinstance(st, dict) else {}
                state = int(job.get("state", 0)) if str(job.get("state", "")).lstrip("-").isdigit() else 0
                if state == 1:
                    body["final_status"] = st
                    return body
                if state in (3, 4):
                    raise OpensolrError(f"ingest: job {body['job_id']} {job.get('state_label', state)}: {job.get('error')}")
                time.sleep(5)
            raise OpensolrError(f"ingest: job {body['job_id']} not completed within {timeout}s")
        return body

    def ingest_status(self, job_id: str) -> Dict[str, Any]:
        """Status of an ingestion job (also visible in the Control Panel)."""
        resp = self._post_with_retry(label="ingest_status", url=
            f"{AI_BASE}/ingest_status",
            data={**self._auth_params(), "job_id": job_id},
        )
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise OpensolrError(f"ingest_status: non-JSON response: {resp.text[:200]}") from exc

    def embed_and_search(self, index: str, query: str, rows: int = 10, **params: Any) -> Dict[str, Any]:
        """Server-side one-shot: embed the query, run the platform's tuned
        hybrid search, return ranked docs.

        Retrieval uses the same pipeline as the hosted search UI: global
        defaults, overridden by the index's saved Search Tuning (Control
        Panel → Index Settings → Search Tuning), overridden by any of these
        per-call knobs passed as extra params: ``fw_title``,
        ``fw_description``, ``fw_uri``, ``fw_text``, ``fw_text_t``,
        ``lexical_weight``, ``vector_weight``, ``vector_topk``,
        ``search_mode`` (union / keywords_required / meaning_required /
        intersection), ``quality_boost``, ``min_score``,
        ``freshness_boost``, ``fresh_bias``, ``lexical_norm_k``, ``mm``
        (flexible / balanced / strict or raw Solr mm syntax).

        ``freshness_boost`` and ``fresh_bias`` are two different knobs whose
        names invite exactly the confusion this paragraph exists to prevent.
        ``freshness_boost`` is a hard window in DAYS: anything older is
        filtered out and numFound drops. ``fresh_bias`` filters nothing — it
        multiplies each score by a recency curve on ``creation_date``, so
        recent documents win ties and near-ties while everything older stays
        reachable, and a document with no ``creation_date`` is simply left
        unboosted. Pass it as ``fresh_bias=1`` (the server also accepts
        "yes"/"true"/"on"); it is off unless asked for.

        Two more defaults are set here and are overridable the same way:
        ``in="all"`` searches every field rather than one named field — a
        RAG client wants the whole document, not just titles — and
        ``fresh="no"`` leaves the platform's date boost off, so ranking
        stays on relevance and an old-but-right document is not pushed
        below a new-but-vague one.
        """
        body = self.ai(
            "embed_and_search",
            index_name=index,
            q=query,
            rows=rows,
            **{"in": "all", "fresh": "no", **params},
        )
        return body

    # ------------------------------------------------------------------ #
    # direct Solr                                                        #
    # ------------------------------------------------------------------ #

    def solr_endpoint(self, index: str) -> Tuple[str, Optional[Tuple[str, str]]]:
        """Return (base_url, basic_auth) for the index's native Solr API."""
        info = self.get_core_info(index)
        url = info.get("connection_url")
        if not url:
            raise OpensolrError(f"No connection_url for index {index!r}")
        auth = None
        if info.get("auth_username"):
            auth = (info["auth_username"], info.get("auth_password") or "")
        return url, auth

    def solr_select(self, index: str, params: Dict[str, Any]) -> Dict[str, Any]:
        base, auth = self.solr_endpoint(index)
        resp = self._post_with_retry(label="solr_select", url=f"{base}/select", data={"wt": "json", **params}, auth=auth)
        resp.raise_for_status()
        return resp.json()

    def hybrid_search(
        self,
        index: str,
        query: str,
        rows: int = 5,
        mode: str = "union",
        alpha: float = 0.5,
        fl: str = "*,score",
        fq: Optional[str] = None,
        fresh_bias: bool = False,
        fresh_bias_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Hybrid (BM25 + kNN) search via the native ``{!hybrid}`` parser.

        The query is embedded server-side; lexical and vector scores are
        fused per document on the Solr side.

        ``fresh_bias`` multiplies the fused score by a recency curve on
        ``creation_date`` (see :data:`FRESH_BIAS_FUNCTION`), so newer documents
        rank higher. It re-orders and never filters: numFound is unchanged and a
        document with no date keeps its place. Off by default.
        """
        # Validate BEFORE embedding. Both checks are purely local, and embedding is a billed
        # GPU round-trip against the account's AI quota — paying for one to then reject the
        # caller's own typo is charging them for our own argument check.
        if mode not in HYBRID_MODES:
            raise ValueError(f"mode must be one of {HYBRID_MODES}, got {mode!r}")
        try:
            alpha_f = float(alpha)
        except (TypeError, ValueError):
            raise ValueError(f"alpha must be a number between 0 and 1, got {alpha!r}") from None
        if not 0.0 <= alpha_f <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha_f}")

        clean = query.replace("{", " ").replace("}", " ").replace('"', " ")
        vector = self.embed(index, query, is_query=True)
        compact = json.dumps(vector, separators=(",", ":"))
        params: Dict[str, Any] = {
            "q": (
                f"{{!hybrid lexical=$lexicalRaw vector=$vectorQuery "
                f"mode={mode} alpha={alpha_f} topN={max(rows, 10)}}}"
            ),
            "lexicalRaw": f'{{!edismax qf="title^100 text^1"}}{clean}',
            "vectorQuery": f"{{!knn f=embeddings topK={max(rows, 10)}}}{compact}",
            "rows": rows,
            "fl": fl,
        }
        if fq:
            params["fq"] = fq
        if fresh_bias:
            apply_fresh_bias(params, fresh_bias_weight)
        return self.solr_select(index, params)

    #: RAG context defaults — how many hybrid hits feed the LLM, and how many
    #: words of each hit's text are included. Both overridable per call, and both
    #: read from the module constants above so the client and the pure builders
    #: can never drift apart.
    RAG_DOCS = DEFAULT_RAG_DOCS
    RAG_WORDS = DEFAULT_RAG_WORDS

    def _rag_context(
        self,
        index: str,
        query: str,
        fq: Optional[str] = None,
        docs: Optional[int] = None,
        words: Optional[int] = None,
        tuning: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Retrieve the top hits, then hand them to build_context().

        Retrieval runs through the server-side ``embed_and_search`` pipeline —
        the platform's own tuned hybrid ranking (field weights, minimum-match,
        quality boosts), the same machinery behind the hosted search UI, so it
        improves automatically with the platform. When a custom ``fq`` is
        given (which that endpoint doesn't accept) — or if it fails —
        retrieval falls back to the client-side ``{!hybrid}`` query.

        This method owns the I/O and nothing else. Selection, ordering, fragment
        cleanup and layout all live in build_context(), which is pure and can
        therefore be proved byte-identical to the platform's own builder instead
        of merely described as identical.
        """
        docs = docs or self.RAG_DOCS
        words = words or self.RAG_WORDS
        hits: List[Dict[str, Any]] = []
        hl: Dict[str, Any] = {}
        if not fq:
            try:
                body = self.embed_and_search(index, query, rows=docs, **(tuning or {}))
                if isinstance(body, dict):
                    hits = body.get("results", {}).get("docs", []) or []
                    # Highlight fragments come back keyed by document id.
                    hl = body.get("results", {}).get("hl", {}) or {}
            except (OpensolrError, httpx.HTTPError):
                hits = []
        if not hits:
            # ``id`` and ``score`` are in the field list because the builder needs
            # them: ``score`` drives the relevance floor, ``id`` is the key the
            # highlight fragments come back under. Asking for the text fields alone,
            # as this used to, left every fallback hit unscored — so on this path the
            # floor could never drop anything, and a weak fourth hit went to the model.
            body = self.hybrid_search(
                index, query, rows=docs,
                fl="id,title,description,text,text_t,score", fq=fq,
            )
            hits = body.get("response", {}).get("docs", [])
        return build_context(hits, hl, top_n=docs, max_words=words)

    def ai_summary(
        self,
        index: str,
        query: str,
        filter_query: Optional[str] = None,
        rag_docs: Optional[int] = None,
        rag_words: Optional[int] = None,
        instruction: Optional[str] = None,
        tuning: Optional[Dict[str, Any]] = None,
        **params: Any,
    ) -> str:
        """Grounded RAG answer: hybrid retrieval over the index feeds the LLM.

        Retrieval runs through the platform's tuned hybrid pipeline (the same one
        behind the hosted search UI): the top ``rag_docs`` hits, first
        ``rag_words`` words of text each, become the context. The whole prompt is
        then assembled HERE — documents first, question last — and sent as the
        single ``instruction`` field.

        Nothing travels beside it. The endpoint's ``query`` and ``context`` fields
        are a backward-compatibility path: the server appends whatever it finds in
        them AFTER the instruction, under QUERY:/CONTENT: labels, which puts the
        documents last. That is the exact inversion of the layout that was
        measured — dropping the trailing question slot scored 3/7 where this
        scores 7/7 — so the instruction is the entire prompt.

        ``instruction`` stays a full override for callers who want their own
        prompt ("Answer in German", "Extract a list of people"). That branch is
        NOT the canonical prompt, so it is honoured verbatim and the question and
        retrieved context are appended to it exactly as the server has always
        appended them: same labels, same order, same guard against repeating text
        the caller already included. Only the place changed, not the result — the
        request now carries one field instead of three.

        Returns plain text.
        """
        data = {
            **self._auth_params(),
            "index_name": index,
            # "no", not "false": Api_lib::ai_summary() disables streaming on that exact string
            # and streams for anything else, so "false" was asking for a stream and getting one.
            # Nothing broke — the body arrives whole and is trimmed — but the parameter did not
            # do what its value said (2026-08-29).
            "stream": "no",
            # Every candidate wording was scored at 0.1, so the temperature is part
            # of the prompt, not a taste setting: raising it invalidates the
            # measurement the wording rests on. A caller can still override it
            # through **params, which is merged last on purpose.
            "temperature": DEFAULT_RAG_TEMPERATURE,
            **params,
        }
        try:
            context = self._rag_context(
                index, query, fq=filter_query, docs=rag_docs, words=rag_words,
                tuning=tuning,
            )
        except (OpensolrError, httpx.HTTPError):
            # Retrieval failed: still send the prompt and let the server fall back
            # to its own retrieval, rather than losing the answer entirely.
            context = ""
        if instruction:
            prompt = str(instruction)
            if query.strip() and query.strip() not in prompt:
                prompt += "\n\nQUERY:\n" + query.strip()
            if context.strip() and context.strip() not in prompt:
                prompt += "\n\nCONTENT:\n" + context.strip()
            data["instruction"] = prompt
        else:
            data["instruction"] = build_instruction(context, query)
        resp = self._post_with_retry(label="ai_summary", url=f"{AI_BASE}/ai_summary", data=data)
        if resp.status_code >= 400:
            raise OpensolrError(f"ai_summary: HTTP {resp.status_code}: {resp.text[:200]}")
        # The stream is prefixed with flush-padding whitespace — strip it.
        return resp.text.strip()

    def solr_update(self, index: str, payload: Any, commit: bool = True) -> Dict[str, Any]:
        base, auth = self.solr_endpoint(index)
        params = {"commit": "true"} if commit else {"commitWithin": "10000"}
        resp = self._post_with_retry(label="solr_update", url=
            f"{base}/update",
            params=params,
            json=payload,
            auth=auth,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # PDF pages, documents, pictures                                     #
    # ------------------------------------------------------------------ #

    def select_grouped(self, index: str, params: Dict[str, Any], group_pages: bool = True) -> Dict[str, Any]:
        """``solr_select`` with the pages of each PDF grouped under its best page.

        See :func:`apply_page_grouping`. An index whose schema has no page fields is searched
        as it is, and remembered, so it is asked once.
        """
        if not group_pages or index in self._flat_indexes:
            return self.solr_select(index, params)
        grouped = dict(params)
        if isinstance(grouped.get("fq"), list):
            grouped["fq"] = list(grouped["fq"])
        apply_page_grouping(grouped)
        try:
            return self.solr_select(index, grouped)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400 and PARENT_FIELD in exc.response.text:
                self._flat_indexes.add(index)
                return self.solr_select(index, params)
            raise

    def read_document(
        self,
        index: str,
        document: str,
        page_from: int = 1,
        page_to: Optional[int] = None,
    ) -> Dict[str, Any]:
        """The whole text of one indexed document.

        ``document`` is its id, the id it was sent with (``meta_ext_id``) or its URL. A PDF
        indexed page by page comes back as its ``pages`` in order, ``page_from`` to
        ``page_to`` (at most :data:`READ_MAX_PAGES` per call), with ``pages_indexed``; any
        other document as one ``text``.
        """
        key = str(document or "").strip()
        if not key:
            raise ValueError("document is required")
        if key.startswith(("http://", "https://")):
            key = hashlib.md5(key.rstrip("/").encode("utf-8")).hexdigest()
        first = max(1, int(page_from or 1))
        last = int(page_to) if page_to else first + READ_MAX_PAGES - 1
        if last < first:
            raise ValueError(f"page_to ({page_to}) is before page_from ({page_from})")
        last = min(last, first + READ_MAX_PAGES - 1)
        esc = key.replace("\\", "\\\\").replace('"', '\\"')
        fl = "id,uri,title,description,text,content_type,creation_date,meta_ext_id"
        docs: List[Dict[str, Any]] = []
        if index not in self._flat_indexes:
            try:
                body = self.solr_select(index, {
                    "q": f'id:"{esc}" OR {PARENT_FIELD}:"{esc}" OR meta_ext_id:"{esc}"',
                    "fq": f"(*:* -{PAGE_FIELD}:[* TO *]) OR {PAGE_FIELD}:[{first} TO {last}]",
                    "sort": f"{PAGE_FIELD} asc",
                    "rows": last - first + 1,
                    "fl": f"{fl},{PAGE_FIELD},{PARENT_FIELD}",
                })
                docs = body.get("response", {}).get("docs", [])
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 400 or (PAGE_FIELD not in exc.response.text and PARENT_FIELD not in exc.response.text):
                    raise
                self._flat_indexes.add(index)
        if index in self._flat_indexes:
            body = self.solr_select(index, {"q": f'id:"{esc}" OR meta_ext_id:"{esc}"', "rows": 1, "fl": fl})
            docs = body.get("response", {}).get("docs", [])
        if not docs:
            raise OpensolrError(f"read_document: nothing in {index} for {document!r}")

        def _text(d: Dict[str, Any]) -> str:
            t = d.get("text")
            return "\n".join(str(x) for x in t) if isinstance(t, list) else str(t or "")

        head = docs[0]
        uri = str(_first(head.get("uri")) or "")
        out: Dict[str, Any] = {
            "id": str(_first(head.get(PARENT_FIELD)) or _first(head.get("id")) or ""),
            "uri": uri,
            "title": str(_first(head.get("title")) or ""),
            "content_type": str(_first(head.get("content_type")) or ""),
            "date": str(_first(head.get("creation_date")) or ""),
        }
        pages = []
        for d in docs:
            try:
                n = int(_first(d.get(PAGE_FIELD)) or 0)
            except (TypeError, ValueError):
                n = 0
            if n > 0:
                pages.append({"page": n, "page_url": page_url(uri, n), "text": _text(d)})
        if not pages:
            out["text"] = _text(head)
            return out
        esc_parent = out["id"].replace("\\", "\\\\").replace('"', '\\"')
        total = self.solr_select(index, {"q": f'{PARENT_FIELD}:"{esc_parent}"', "rows": 0})
        out.update(
            pages_indexed=int(total.get("response", {}).get("numFound") or len(pages)),
            page_from=pages[0]["page"],
            page_to=pages[-1]["page"],
            pages=pages,
        )
        return out

    def describe_image(self, index: str, image_b64: str) -> Dict[str, Any]:
        """What a picture shows (``caption``), the text printed on it (``text``, read by OCR),
        its ``labels`` and any barcodes / QR ``codes``."""
        ans = self.image_to_text(index, image_b64, top_k=8)
        labels = [l["label"] for l in (ans.get("labels") or []) if isinstance(l, dict) and l.get("label")]
        codes: List[str] = []
        for c in ans.get("codes") or []:
            value = c.get("text") if isinstance(c, dict) else (c if isinstance(c, str) else None)
            if value:
                codes.append(value)
        mode = ans.get("mode") or ""
        ocr = str(ans.get("ocr_text") or "").strip() or (str(ans.get("text") or "").strip() if mode == "ocr" else "")
        caption = str(ans.get("caption") or "").strip() or (str(ans.get("text") or "").strip() if mode != "ocr" else "")
        return {"caption": caption, "text": ocr, "labels": labels, "codes": codes}

    # ------------------------------------------------------------------ #
    # files: read on Opensolr (text, OCR, a PDF page by page), indexed  #
    # ------------------------------------------------------------------ #

    #: Seconds between two looks at files that are still being read.
    FILE_POLL_SECONDS = 3.0

    #: Job ids one file_extract_status call accepts.
    FILE_STATUS_MAX = 20

    @staticmethod
    def _file_hash(path: Path) -> Tuple[str, int]:
        digest = hashlib.sha1()
        size = 0
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        return digest.hexdigest(), size

    def upload_file(
        self,
        index: str,
        file: Union[str, Path],
        pages: bool = False,
        text: bool = True,
    ) -> Dict[str, Any]:
        """Send one local file to be read on Opensolr: its text, its pictures through OCR, a PDF page by page.

        The bytes travel in parts, and only once: a file this account already sent (same
        content) is answered without sending it again. Returns ``job_id`` and ``state``
        (``queued``, ``processing``, ``done`` or ``failed``); once done, ``mime`` and the
        ``text``, or with ``pages=True`` the ``pages`` of a PDF (``page``, ``text``,
        ``is_ocr``, ``lang``). ``text=False`` answers the state alone. The pictures read are
        counted on the index owner's AI quota, 0.5 each.
        """
        path = Path(file).expanduser()
        sha1, size = self._file_hash(path)
        if size < 1:
            raise OpensolrError(f"file_extract: {path.name} is empty")
        form = {
            "index_name": index, "sha1": sha1, "size": size, "name": path.name,
            "pages": "1" if pages else "0", "text": "1" if text else "0",
        }
        ans = self.ai("file_extract", **form)
        stuck = 0
        with open(path, "rb") as fh:
            while ans.get("state") == "upload":
                offset = int(ans.get("received") or 0)
                part_max = int(ans.get("max_part_bytes") or 0)
                if part_max < 1 or offset >= size:
                    raise OpensolrError(f"file_extract: {path.name}: unexpected upload state {str(ans)[:200]}")
                fh.seek(offset)
                part = fh.read(min(part_max, size - offset))
                resp = self._post_with_retry(
                    label="file_extract",
                    url=f"{AI_BASE}/file_extract",
                    data={**self._auth_params(), **form, "offset": offset},
                    files={"part": (path.name, part, "application/octet-stream")},
                )
                ans = self._answer("file_extract", resp)
                if ans.get("state") == "upload" and int(ans.get("received") or 0) <= offset:
                    stuck += 1
                    if stuck >= 3:
                        raise OpensolrError(f"file_extract: {path.name}: the upload does not get past {offset} bytes")
                else:
                    stuck = 0
        return ans

    def file_status(
        self,
        index: str,
        job_ids: Sequence[str],
        pages: bool = False,
        text: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """State of files sent with :meth:`upload_file`, keyed by job id, with their text (or pages) once read."""
        ids = list(dict.fromkeys(str(j) for j in job_ids if j))
        out: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(ids), self.FILE_STATUS_MAX):
            body = self.ai(
                "file_extract_status", index_name=index, job_ids=json.dumps(ids[i : i + self.FILE_STATUS_MAX]),
                pages="1" if pages else "0", text="1" if text else "0",
            )
            results = body.get("results") if isinstance(body, dict) else None
            if not isinstance(results, dict):
                raise OpensolrError(f"file_extract_status: unexpected response: {str(body)[:200]}")
            out.update(results)
        return out

    def _wait_files(self, index: str, entries: List[Dict[str, Any]], pages: bool, text: bool, timeout: float) -> None:
        deadline = time.time() + max(0.0, float(timeout))
        while True:
            waiting: Dict[str, List[Dict[str, Any]]] = {}
            for e in entries:
                if e.get("job_id") and e.get("state") in ("queued", "processing"):
                    waiting.setdefault(e["job_id"], []).append(e)
            if not waiting or time.time() >= deadline:
                return
            time.sleep(self.FILE_POLL_SECONDS)
            for job_id, ans in self.file_status(index, list(waiting), pages=pages, text=text).items():
                for e in waiting.get(job_id, []):
                    e["state"] = ans.get("state") or "failed"
                    for key in ("text", "pages", "mime", "error"):
                        if key in ans:
                            e[key] = ans[key]

    def _send_files(self, index: str, files: Sequence[Union[str, Path]], pages: bool, text: bool) -> List[Dict[str, Any]]:
        entries = []
        for f in files:
            entry: Dict[str, Any] = {"path": str(f)}
            try:
                ans = self.upload_file(index, f, pages=pages, text=text)
                entry["job_id"] = ans.get("job_id")
                entry["state"] = ans.get("state")
                for key in ("text", "pages", "mime", "error"):
                    if key in ans:
                        entry[key] = ans[key]
            except (OpensolrError, OSError) as exc:
                entry.update(state="failed", error=str(exc))
            entries.append(entry)
        return entries

    def extract_files(
        self,
        index: str,
        files: Sequence[Union[str, Path]],
        pages: bool = True,
        timeout: float = 600.0,
    ) -> List[Dict[str, Any]]:
        """Read local files on Opensolr without indexing them (see :meth:`upload_file`).

        One entry per file: ``path``, ``job_id``, ``state`` and, once ``done``, ``mime`` plus
        the ``pages`` of a PDF (with ``pages=True``) or the ``text``; ``error`` when it could
        not be read. Files still ``queued``/``processing`` when ``timeout`` runs out can be
        asked for again: the bytes are not sent twice.
        """
        entries = self._send_files(index, files, pages=pages, text=True)
        self._wait_files(index, entries, pages=pages, text=True, timeout=timeout)
        return entries

    def ingest_files(
        self,
        index: str,
        path: Union[str, Path],
        pattern: str = "*",
        recursive: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        wait: bool = False,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """Index local files through Data Ingestion: one file, or every file of a folder.

        Each file is read on Opensolr (its text, its pictures through OCR) and a PDF is
        indexed one document per page, exactly like a PDF sent by URL with ``rtf``. Its
        ``uri`` is ``https://ingest.opensolr.com/<index>/<source>``, where ``source`` is the
        file's path from the folder sent (also in ``meta_source``), so sending it again
        replaces it; its date is the file's modification time. ``metadata`` is added to every
        document as ``meta_*`` fields. Files still being read when ``timeout`` runs out come
        back under ``pending``: call again to index them, the bytes are not sent twice.

        Returns ``{"documents": [...], "pending": [...], "failed": [...], "jobs": [...]}``;
        ``jobs`` are the Data Ingestion job ids (``wait=True`` blocks until they are done).
        """
        meta = {str(k): v for k, v in (metadata or {}).items() if isinstance(v, (str, int, float, bool))}
        root = Path(path).expanduser()
        files = list_files(root, pattern, recursive)
        entries = self._send_files(index, files, pages=False, text=False)
        for e, f in zip(entries, files):
            e["source"] = file_source(root, f)
        self._wait_files(index, entries, pages=False, text=False, timeout=timeout)

        documents: List[Dict[str, Any]] = []
        pending: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        docs: List[Dict[str, Any]] = []
        for e in entries:
            if e.get("state") in ("queued", "processing"):
                pending.append({"path": e["path"], "state": e["state"]})
                continue
            if e.get("state") != "done":
                failed.append({"path": e["path"], "error": e.get("error") or e.get("state") or "not read"})
                continue
            src = e["source"]
            uri = f"https://ingest.opensolr.com/{index}/{quote(src, safe='')}"
            name = Path(e["path"])
            doc: Dict[str, Any] = {
                "uri": uri,
                "title": (name.stem or name.name)[:250],
                "description": src,
                "rtf": True,
                "file": e["job_id"],
                "meta_ext_id": src,
                "meta_source": src,
                "meta_lc_json": json.dumps({"source": src, **meta}, ensure_ascii=False),
            }
            try:
                doc["timestamp"] = int(name.stat().st_mtime)
            except OSError:
                pass
            for key, value in meta.items():
                field = "meta_" + _META_FIELD_RE.sub("_", key.lower()).strip("_")
                if field not in ("meta_", "meta_lc_json", "meta_ext_id", "meta_source"):
                    doc[field] = str(value)
            docs.append(doc)
            documents.append({
                "path": e["path"], "source": src, "uri": uri,
                "id": hashlib.md5(uri.encode("utf-8")).hexdigest(),
            })
        jobs: List[str] = []
        for i in range(0, len(docs), 50):
            body = self.ingest(index, docs[i : i + 50], wait=wait, timeout=timeout)
            if isinstance(body, dict) and body.get("job_id"):
                jobs.append(str(body["job_id"]))
        return {"documents": documents, "pending": pending, "failed": failed, "jobs": jobs}

    def close(self) -> None:
        self._http.close()
