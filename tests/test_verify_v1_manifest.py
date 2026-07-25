from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.freeze_v2_contract import sha256_contract_text
from scripts.verify_v1_manifest import file_matches_checkout_hash


class CheckoutHashTests(unittest.TestCase):
    def test_windows_frozen_hash_matches_linux_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_bytes(b'{\n  "status": "accepted"\n}\n')
            expected = hashlib.sha256(
                b'{\r\n  "status": "accepted"\r\n}\r\n'
            ).hexdigest()
            self.assertTrue(file_matches_checkout_hash(path, expected))

    def test_content_change_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_bytes(b'{\n  "status": "changed"\n}\n')
            expected = hashlib.sha256(
                b'{\r\n  "status": "accepted"\r\n}\r\n'
            ).hexdigest()
            self.assertFalse(file_matches_checkout_hash(path, expected))

    def test_v2_contract_text_hash_is_platform_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            lf = Path(directory) / "lf.txt"
            crlf = Path(directory) / "crlf.txt"
            lf.write_bytes(b"one\ntwo\n")
            crlf.write_bytes(b"one\r\ntwo\r\n")
            self.assertEqual(
                sha256_contract_text(lf),
                sha256_contract_text(crlf),
            )


if __name__ == "__main__":
    unittest.main()
