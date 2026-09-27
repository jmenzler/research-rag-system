from __future__ import annotations

import unittest


class ZstdCompatibilityTests(unittest.TestCase):
    def test_abstracts_zstd_round_trip_on_supported_python(self) -> None:
        from src.citations import abstracts

        payload = b"public export compatibility"
        compressed = abstracts.zstd.compress(payload)

        self.assertEqual(abstracts.zstd.decompress(compressed), payload)


if __name__ == "__main__":
    unittest.main()
