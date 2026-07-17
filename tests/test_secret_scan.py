from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.scan_corpus_secrets import scan  # noqa: E402


class SecretScanTests(unittest.TestCase):
    def test_detects_realistic_token_without_retaining_plaintext(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.md"
            token = "sk-or-v1-abcdefghijklmnopqrstuvwx"
            path.write_text(f"credential: {token}\n", encoding="utf-8")
            findings = scan([path])
            self.assertEqual(len(findings), 1)
            self.assertNotIn(token, str(findings))

    def test_allows_explicit_placeholder_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.md"
            path.write_text("api_key=replace_with_your_key\n", encoding="utf-8")
            self.assertEqual(scan([path]), [])


if __name__ == "__main__":
    unittest.main()

