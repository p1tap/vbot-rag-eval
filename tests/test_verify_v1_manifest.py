from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
