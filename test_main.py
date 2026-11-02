"""Tests for main.py.

    pip install httpx            # plus the requirements listed at the top of main.py
    python -m unittest test_main -v

The embedder and vector store are in-memory fakes, so no model download, Qdrant server or GPU is needed.
PDF parsing (PyMuPDF) and the real Qdrant and sentence-transformers adapters are not covered here.
"""
import hashlib
import importlib.util
import io
import math
import re
import unittest

try:
    import main
except ImportError as exc:                      # the required libraries are not installed
    raise unittest.SkipTest(f"install the requirements listed in main.py ({exc})")

from main import (DocumentTooLarge, EmptyDocument, Hit, IngestionService, PointRecord, SearchService, Settings,
                  UnreadableDocument, UnsupportedFileType, chunk_words, detect_delimiter, extract_csv,
                  extract_xlsx, file_extension, matches_signature, read_limited, sanitize_filename)

HAVE_API_DEPS = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx", "multipart"))


class FakeEmbedder:
    """Hashes words into a small normalised vector, so texts that share words are similar."""
    name, dimension, max_tokens = "fake", 32, 256

    def __init__(self, fail_on_call=None):
        self.calls, self.fail_on_call = 0, fail_on_call

    def encode(self, texts):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("embedding failed")
        out = []
        for text in texts:
            vec = [0.0] * self.dimension
            for word in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


class FakeStore:
    def __init__(self):
        self.points, self.healthy, self.dimension = {}, True, None

    def ensure_collection(self, dimension): self.dimension = dimension
    def upsert(self, points): self.points.update({p.id: p for p in points})

    def search(self, vector, limit, filters, score_threshold):
        hits = []
        for p in self.points.values():
            if all(p.payload.get(k) == v for k, v in filters.items()):
                score = sum(a * b for a, b in zip(vector, p.vector))
                if score_threshold is None or score >= score_threshold:
                    hits.append(Hit(p.id, score, dict(p.payload)))
        return sorted(hits, key=lambda h: -h.score)[:limit]

    def _where(self, fn): return [i for i, p in self.points.items() if fn(p.payload)]
    def count_document(self, source, document_id):
        return len(self._where(lambda q: q["source"] == source and q["document_id"] == document_id))
    def delete_stale(self, source, keep):
        for i in self._where(lambda q: q["source"] == source and q["document_id"] != keep): del self.points[i]
    def delete_document(self, source, document_id):
        for i in self._where(lambda q: q["source"] == source and q["document_id"] == document_id): del self.points[i]
    def delete_source(self, source):
        ids = self._where(lambda q: q["source"] == source)
        for i in ids: del self.points[i]
        return len(ids)
    def ping(self):
        if not self.healthy: raise RuntimeError("down")
    def close(self): pass


def csv_bytes(*rows):
    return ("\n".join(rows) + "\n").encode()


SETTINGS = Settings(chunk_words=20, chunk_overlap=5, embed_batch_size=3, max_chunks_per_document=50)


class TextUtilityTests(unittest.TestCase):
    def test_chunks_overlap_and_end_exactly_at_the_text_end(self):
        text = " ".join(f"w{i}" for i in range(1, 11))
        self.assertEqual(chunk_words(text, 4, 1), ["w1 w2 w3 w4", "w4 w5 w6 w7", "w7 w8 w9 w10"])

    def test_no_redundant_tail_chunk(self):
        text = " ".join(f"w{i}" for i in range(1, 8))
        self.assertEqual(chunk_words(text, 4, 1), ["w1 w2 w3 w4", "w4 w5 w6 w7"])

    def test_empty_and_invalid(self):
        self.assertEqual(chunk_words("   \n ", 4, 1), [])
        for size, overlap in ((0, 0), (4, 4), (4, -1)):
            with self.assertRaises(ValueError):
                chunk_words("a b c", size, overlap)

    def test_filenames_are_reduced_to_a_safe_base_name(self):
        self.assertEqual(sanitize_filename("../../etc/passwd"), "passwd")
        self.assertEqual(sanitize_filename("C:\\docs\\Report.PDF"), "Report.PDF")
        self.assertEqual(sanitize_filename("a\x00b\n.csv"), "ab.csv")
        self.assertEqual(sanitize_filename(None), "unknown")
        self.assertEqual(sanitize_filename(""), "unknown")
        self.assertEqual(file_extension("Report.PDF"), "pdf")
        self.assertEqual(file_extension("noextension"), "")

    def test_signatures(self):
        self.assertTrue(matches_signature("pdf", b"%PDF-1.7 ..."))
        self.assertFalse(matches_signature("pdf", b"hello"))
        self.assertTrue(matches_signature("xlsx", b"PK\x03\x04rest"))
        self.assertFalse(matches_signature("xlsx", b"plain text"))
        self.assertTrue(matches_signature("csv", b"anything"))

    def test_read_limited(self):
        self.assertEqual(read_limited(io.BytesIO(b"abcdef").read, 10, block=2), b"abcdef")
        with self.assertRaises(DocumentTooLarge):
            read_limited(io.BytesIO(b"abcdef").read, 5, block=2)


class ExtractorTests(unittest.TestCase):
    def rows(self, data):
        return [(c.text, c.metadata) for c in extract_csv(data, SETTINGS)]

    def test_csv_rows_keep_values_as_written_and_skip_empty_cells(self):
        got = self.rows(csv_bytes("id,name,note", "007,Ann,", "008,,hello world", ",,"))
        self.assertEqual(got, [("id: 007, name: Ann", {"row": 1}), ("id: 008, note: hello world", {"row": 2})])

    def test_delimiters_are_detected(self):
        self.assertEqual(detect_delimiter("a;b;c\n1;2;3"), ";")
        self.assertEqual(detect_delimiter("a\tb\n1\t2"), "\t")
        self.assertEqual(detect_delimiter("single\nvalue"), ",")
        self.assertEqual(self.rows(csv_bytes("id;name", "1;Ann"))[0][0], "id: 1, name: Ann")

    def test_blank_header_names_and_extra_cells(self):
        self.assertEqual(self.rows(csv_bytes("a,,c", "1,2,3,4"))[0][0], "a: 1, column_2: 2, c: 3, column_4: 4")

    def test_utf8_bom_and_latin1_fallback(self):
        self.assertEqual(self.rows(b"\xef\xbb\xbfname\nCaf\xc3\xa9")[0][0], "name: Café")
        self.assertEqual(self.rows(b"name\nCaf\xe9")[0][0], "name: Café")

    def test_long_rows_are_split_to_fit_the_embedding_window(self):
        long_value = " ".join(f"w{i}" for i in range(60))
        chunks = list(extract_csv(csv_bytes("text", long_value), SETTINGS))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.metadata == {"row": 1} for c in chunks))
        self.assertTrue(all(len(c.text.split()) <= SETTINGS.chunk_words for c in chunks))

    @unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("openpyxl"), "pandas/openpyxl missing")
    def test_xlsx_all_sheets_with_sheet_and_row_metadata(self):
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Orders"
        ws.append(["Item", "Qty", "Note"]); ws.append(["Pen", 12, None]); ws.append([None, None, None]); ws.append(["Ink", 3.5, "rush"])
        wb.create_sheet("People").append(["Name"]); wb["People"].append(["Ann"])
        buf = io.BytesIO(); wb.save(buf)
        got = [(c.text, c.metadata) for c in extract_xlsx(buf.getvalue(), SETTINGS)]
        self.assertEqual(got, [("[Sheet: Orders] Item: Pen, Qty: 12", {"sheet": "Orders", "row": 2}),
                               ("[Sheet: Orders] Item: Ink, Qty: 3.5, Note: rush", {"sheet": "Orders", "row": 4}),
                               ("[Sheet: People] Name: Ann", {"sheet": "People", "row": 2})])


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.embedder, self.store = FakeEmbedder(), FakeStore()
        self.ingest = IngestionService(self.embedder, self.store, SETTINGS)
        self.search = SearchService(self.embedder, self.store)
        self.data = csv_bytes("item,qty", *[f"widget{i},{i}" for i in range(7)])

    def test_indexes_every_row_with_metadata_in_batches(self):
        result = self.ingest.ingest("stock.csv", self.data, "inventory")
        self.assertEqual((result.status, result.chunks), ("indexed", 7))
        self.assertEqual(self.embedder.calls, 3)                     # 7 chunks, batches of 3
        payload = next(iter(self.store.points.values())).payload
        self.assertEqual((payload["source"], payload["doc_type"], payload["category"]), ("stock.csv", "csv", "inventory"))
        self.assertEqual({"row", "chunk_index", "document_id", "text"} <= set(payload), True)

    def test_default_category_and_filename_cleaning(self):
        self.ingest.ingest("../../x/stock.csv", self.data)
        payload = next(iter(self.store.points.values())).payload
        self.assertEqual((payload["source"], payload["category"]), ("stock.csv", "general"))

    def test_reingesting_identical_content_is_a_no_op(self):
        self.ingest.ingest("stock.csv", self.data)
        calls = self.embedder.calls
        again = self.ingest.ingest("stock.csv", self.data)
        self.assertEqual((again.status, again.chunks, len(self.store.points)), ("unchanged", 7, 7))
        self.assertEqual(self.embedder.calls, calls)

    def test_new_version_replaces_the_old_one(self):
        first = self.ingest.ingest("stock.csv", self.data)
        second = self.ingest.ingest("stock.csv", csv_bytes("item,qty", "gadget,1", "gizmo,2"))
        self.assertNotEqual(first.document_id, second.document_id)
        self.assertEqual({p.payload["document_id"] for p in self.store.points.values()}, {second.document_id})
        self.assertEqual(len(self.store.points), 2)

    def test_different_files_do_not_interfere(self):
        self.ingest.ingest("a.csv", self.data)
        self.ingest.ingest("b.csv", self.data)
        self.assertEqual(len(self.store.points), 14)

    def test_failure_midway_leaves_nothing_behind(self):
        service = IngestionService(FakeEmbedder(fail_on_call=2), self.store, SETTINGS)
        with self.assertRaises(RuntimeError):
            service.ingest("stock.csv", self.data)
        self.assertEqual(self.store.points, {})

    def test_failed_reupload_keeps_the_previous_version(self):
        self.ingest.ingest("stock.csv", self.data)
        service = IngestionService(FakeEmbedder(fail_on_call=2), self.store, SETTINGS)
        with self.assertRaises(RuntimeError):
            service.ingest("stock.csv", csv_bytes("item,qty", *[f"other{i},{i}" for i in range(7)]))
        self.assertEqual(len(self.store.points), 7)
        self.assertTrue(all("widget" in p.payload["text"] for p in self.store.points.values()))

    def test_too_many_chunks_is_rejected_and_cleaned_up(self):
        small = Settings(chunk_words=20, chunk_overlap=5, embed_batch_size=3, max_chunks_per_document=4)
        with self.assertRaises(DocumentTooLarge):
            IngestionService(self.embedder, self.store, small).ingest("stock.csv", self.data)
        self.assertEqual(self.store.points, {})

    def test_rejects_bad_input(self):
        with self.assertRaises(UnsupportedFileType):
            self.ingest.ingest("notes.txt", b"hello")
        with self.assertRaises(UnsupportedFileType):
            self.ingest.ingest("legacy.xls", b"hello")
        with self.assertRaises(EmptyDocument):
            self.ingest.ingest("empty.csv", b"")
        with self.assertRaises(EmptyDocument):
            self.ingest.ingest("header_only.csv", csv_bytes("a,b"))
        with self.assertRaises(UnreadableDocument):
            self.ingest.ingest("fake.pdf", b"this is not a pdf")
        with self.assertRaises(UnreadableDocument):
            self.ingest.ingest("fake.xlsx", b"this is not a workbook")
        self.assertEqual(self.store.points, {})

    def test_corrupt_xlsx_with_valid_signature_is_a_client_error(self):
        if not (importlib.util.find_spec("pandas") and importlib.util.find_spec("openpyxl")):
            self.skipTest("pandas/openpyxl missing")
        with self.assertRaises(UnreadableDocument):
            self.ingest.ingest("broken.xlsx", b"PK\x03\x04 but not really a workbook")

    def test_point_ids_are_deterministic(self):
        self.ingest.ingest("stock.csv", self.data)
        ids = set(self.store.points)
        self.store.points.clear()
        IngestionService(FakeEmbedder(), self.store, SETTINGS).ingest("stock.csv", self.data)
        self.assertEqual(set(self.store.points), ids)

    # ---- search -------------------------------------------------------------
    def test_search_ranks_relevant_chunks_first(self):
        self.ingest.ingest("fruit.csv", csv_bytes("name,colour", "apple,red", "banana,yellow"))
        self.ingest.ingest("cars.csv", csv_bytes("name,colour", "sedan,blue", "truck,green"))
        hits = self.search.search("banana yellow", 3)
        self.assertEqual(hits[0].payload["text"], "name: banana, colour: yellow")
        self.assertEqual([h.score for h in hits], sorted((h.score for h in hits), reverse=True))

    def test_search_filters(self):
        self.ingest.ingest("fruit.csv", csv_bytes("name", "apple"), "food")
        self.ingest.ingest("cars.csv", csv_bytes("name", "apple truck"), "vehicles")
        self.assertEqual({h.payload["source"] for h in self.search.search("apple", 10)}, {"fruit.csv", "cars.csv"})
        self.assertEqual({h.payload["source"] for h in self.search.search("apple", 10, source="cars.csv")}, {"cars.csv"})
        self.assertEqual({h.payload["source"] for h in self.search.search("apple", 10, category="food")}, {"fruit.csv"})
        self.assertEqual(len(self.search.search("apple", 10, doc_type=".CSV")), 2)
        self.assertEqual(self.search.search("apple", 10, doc_type="pdf"), [])
        self.assertEqual(self.search.search("zzz qqq", 10, score_threshold=0.5), [])

    def test_delete_source(self):
        self.ingest.ingest("fruit.csv", csv_bytes("name", "apple", "pear"))
        self.ingest.ingest("cars.csv", csv_bytes("name", "truck"))
        self.assertEqual(self.search.delete_source("../fruit.csv"), 2)
        self.assertEqual(self.search.delete_source("fruit.csv"), 0)
        self.assertEqual(len(self.store.points), 1)


class SettingsTests(unittest.TestCase):
    def test_validation(self):
        for kwargs in ({"chunk_words": 10, "chunk_overlap": 10}, {"chunk_words": 0}, {"embed_batch_size": 0},
                       {"max_upload_bytes": 0}):
            with self.assertRaises(ValueError):
                Settings(**kwargs)

    def test_from_env(self):
        import os
        old = {k: os.environ.get(k) for k in ("MAX_UPLOAD_MB", "CHUNK_WORDS", "QDRANT_URL", "API_KEY")}
        try:
            os.environ.update(MAX_UPLOAD_MB="2", CHUNK_WORDS="50", QDRANT_URL="http://q:6333", API_KEY="k")
            s = Settings.from_env()
            self.assertEqual((s.max_upload_bytes, s.chunk_words, s.qdrant_url, s.api_key),
                             (2 * 1024 * 1024, 50, "http://q:6333", "k"))
            os.environ["CHUNK_WORDS"] = "abc"
            with self.assertRaises(ValueError):
                Settings.from_env()
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@unittest.skipUnless(HAVE_API_DEPS, "fastapi, httpx and python-multipart are required")
class ApiTests(unittest.TestCase):
    def client(self, **settings):
        from fastapi.testclient import TestClient
        base = dict(chunk_words=20, chunk_overlap=5, embed_batch_size=3, api_key="secret")
        self.store = FakeStore()
        app = main.create_app(Settings(**{**base, **settings}), embedder=FakeEmbedder(), store=self.store)
        return TestClient(app)

    H = {"X-API-Key": "secret"}

    def upload(self, c, name="data.csv", content=b"name\napple\npear\n", **kw):
        return c.post("/ingest", files={"file": (name, content, "text/csv")}, data=kw, headers=self.H)

    def test_api_key_is_required_when_configured(self):
        with self.client() as c:
            self.assertEqual(c.post("/search", json={"query": "x"}).status_code, 401)
            self.assertEqual(c.post("/search", json={"query": "x"}, headers={"X-API-Key": "no"}).status_code, 401)
            self.assertEqual(c.delete("/documents?source=a.csv").status_code, 401)
            self.assertEqual(c.get("/health").status_code, 200)               # health stays open

    def test_ingest_then_search_then_delete(self):
        with self.client() as c:
            r = self.upload(c, custom_category="food")
            self.assertEqual((r.status_code, r.json()["chunks_indexed"], r.json()["status"]), (201, 2, "indexed"))
            again = self.upload(c)
            self.assertEqual((again.status_code, again.json()["status"]), (200, "unchanged"))
            found = c.post("/search", json={"query": "apple", "limit": 1, "category": "food"}, headers=self.H).json()
            self.assertEqual(found["results"][0]["text"], "name: apple")
            self.assertNotIn("text", found["results"][0]["metadata"])
            self.assertEqual(c.delete("/documents?source=data.csv", headers=self.H).json()["deleted_chunks"], 2)

    def test_error_statuses(self):
        with self.client(max_upload_bytes=200) as c:
            self.assertEqual(self.upload(c, "notes.txt", b"hi").status_code, 415)
            self.assertEqual(self.upload(c, "x.csv", b"a\n" + b"b\n" * 500).status_code, 413)
            self.assertEqual(self.upload(c, "x.pdf", b"not a pdf").status_code, 422)
            self.assertEqual(self.upload(c, "e.csv", b"").status_code, 422)
            self.assertEqual(c.post("/search", json={"query": "   "}, headers=self.H).status_code, 422)
            self.assertEqual(c.post("/search", json={"query": "x", "limit": 0}, headers=self.H).status_code, 422)

    def test_health_reports_an_unreachable_store(self):
        with self.client() as c:
            self.store.healthy = False
            self.assertEqual(c.get("/health").status_code, 503)


if __name__ == "__main__":
    unittest.main()
