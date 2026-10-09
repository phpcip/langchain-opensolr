"""Opensolr document loader for LangChain: local files read on Opensolr, a PDF page by page."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from ._client import OpensolrClient, list_files

logger = logging.getLogger(__name__)


class OpensolrLoader(BaseLoader):
    """Load local files as Documents, read on Opensolr's servers.

    Each file is read there: its text, the text in its pictures (OCR, more than 100
    languages), a short description of the pictures that have no text, scanned pages
    and picture files (JPG, PNG, TIFF...) included. A PDF gives one Document per page
    (``metadata["page"]``, from 1), any other file one Document. Nothing is indexed.

    Example:
        .. code-block:: python

            from langchain_opensolr import OpensolrLoader

            loader = OpensolrLoader("./contracts", index="mysite__dense",
                                    email="you@example.com", api_key="...", pattern="*.pdf")
            for doc in loader.lazy_load():
                print(doc.metadata["source"], doc.metadata.get("page"), doc.page_content[:80])

    Args:
        path: A file or a folder.
        index: The Opensolr index the work is counted on: each picture read costs 0.5
            of an AI request; a file this account sent before is read for free.
        email: Opensolr account email.
        api_key: Opensolr API key.
        client: Optional pre-configured :class:`OpensolrClient` (overrides email/api_key).
        pattern: Which files of a folder, e.g. ``"*.pdf"``.
        recursive: Walk the sub-folders too.
        timeout: Seconds to wait for each batch of files to be read.

    Files that could not be read are skipped and listed in ``failed`` (path, error).
    Files still being read when ``timeout`` ran out are listed in ``pending``: load
    again to get them, their bytes are not sent twice.
    """

    #: Files sent, and waited for, together.
    BATCH = 20

    def __init__(
        self,
        path: Union[str, Path],
        index: str,
        email: str = "",
        api_key: str = "",
        client: Optional[OpensolrClient] = None,
        pattern: str = "*",
        recursive: bool = True,
        timeout: float = 600.0,
    ) -> None:
        if client is None:
            if not (email and api_key):
                raise ValueError("Provide either a client or email + api_key")
            client = OpensolrClient(email, api_key)
        if not index:
            raise ValueError("index is required")
        self._client = client
        self._index = index
        self.path = path
        self.pattern = pattern
        self.recursive = recursive
        self.timeout = timeout
        self.failed: List[Dict[str, str]] = []
        self.pending: List[str] = []

    def lazy_load(self) -> Iterator[Document]:
        self.failed, self.pending = [], []
        files = list_files(self.path, self.pattern, self.recursive)
        for i in range(0, len(files), self.BATCH):
            entries = self._client.extract_files(self._index, files[i : i + self.BATCH], pages=True, timeout=self.timeout)
            for entry in entries:
                yield from self._documents(entry)

    def _documents(self, entry: Dict[str, Any]) -> Iterator[Document]:
        state = entry.get("state")
        if state in ("queued", "processing"):
            self.pending.append(entry["path"])
            logger.warning("Opensolr is still reading %s; load again to get it", entry["path"])
            return
        if state != "done":
            error = str(entry.get("error") or state or "not read")
            self.failed.append({"path": entry["path"], "error": error})
            logger.warning("Opensolr could not read %s: %s", entry["path"], error)
            return
        base = {"source": entry["path"], "content_type": entry.get("mime") or ""}
        pages = entry.get("pages") or []
        if pages:
            for p in pages:
                text = str(p.get("text") or "")
                if text.strip():
                    yield Document(page_content=text, metadata={
                        **base,
                        "page": int(p.get("page") or 0),
                        "is_ocr": bool(p.get("is_ocr")),
                        "language": str(p.get("lang") or ""),
                    })
            return
        text = str(entry.get("text") or "")
        if text.strip():
            yield Document(page_content=text, metadata=base)
