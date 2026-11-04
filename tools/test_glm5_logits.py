import math
from pathlib import Path
import struct
import tempfile
import unittest

from tools.glm5.compare_logits import compare


class LogitComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ref = Path(self.temp.name) / "reference.bin"
        self.other = Path(self.temp.name) / "candidate.bin"

    def write(self, path, rows, flags=0, tokens=(0, 1, 2)):
        data = struct.pack("<6i", 0x474C4D50, 1, 3, 2, 1, flags)
        data += struct.pack("<3i", *tokens)
        for target, row in zip(tokens[1:], rows):
            data += struct.pack("<i3f", target, *row)
        path.write_bytes(data)

    def test_constant_shift_preserves_distribution(self):
        self.write(self.ref, [(0, 1, 2), (2, 0, 1)])
        self.write(self.other, [(5, 6, 7), (7, 5, 6)], 1)
        result = compare(self.ref, self.other)
        self.assertAlmostEqual(result["summary"]["mean_kl"], 0, places=14)
        self.assertAlmostEqual(result["summary"]["mean_tv"], 0, places=14)
        self.assertAlmostEqual(result["summary"]["mean_nll_delta"], 0, places=14)
        self.assertEqual(result["rows"][0]["max_logit_error"], 5)

    def test_low_margin_flip_and_true_token_loss(self):
        self.write(self.ref, [(0.01, 0, -2), (2, 0, -1)])
        self.write(self.other, [(0, 0.01, -2), (2, 0, -1)], 1)
        result = compare(self.ref, self.other)
        self.assertEqual(result["summary"]["low_margin_positions"], 1)
        self.assertEqual(result["summary"]["argmax_changes"], 1)
        self.assertEqual(result["summary"]["low_margin_argmax_changes"], 1)
        self.assertLess(result["rows"][0]["nll_delta"], 0)
        self.assertGreater(result["rows"][0]["kl"], 0)

    def test_large_offset_keeps_true_token_normalizer(self):
        self.write(self.ref, [(1e20, 1e20, 1e20)] * 2)
        self.write(self.other, [(0, 0, 0)] * 2, 1)
        result = compare(self.ref, self.other)
        self.assertAlmostEqual(result["summary"]["mean_reference_nll"], math.log(3), places=14)
        self.assertAlmostEqual(result["summary"]["mean_nll_delta"], 0, places=14)

    def test_conditioning_must_match(self):
        self.write(self.ref, [(0, 1, 2)] * 2)
        self.write(self.other, [(0, 1, 2)] * 2, tokens=(1, 1, 2))
        with self.assertRaisesRegex(ValueError, "conditioning"):
            compare(self.ref, self.other)

    def test_invalid_payloads_fail(self):
        self.write(self.ref, [(0, 1, 2)] * 2)
        for rows in [[(0, math.nan, 2)] * 2, [(0, math.inf, 2)] * 2]:
            self.write(self.other, rows)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                compare(self.ref, self.other)
        self.write(self.other, [(0, 1, 2)] * 2)
        self.other.write_bytes(self.other.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "truncated"):
            compare(self.ref, self.other)
        self.write(self.other, [(0, 1, 2)] * 2)
        with self.other.open("ab") as file:
            file.write(b"x")
        with self.assertRaisesRegex(ValueError, "trailing"):
            compare(self.ref, self.other)

    def test_header_shape_and_target_validation(self):
        self.write(self.ref, [(0, 1, 2)] * 2)
        self.write(self.other, [(0, 1, 2)] * 2)
        data = bytearray(self.other.read_bytes())
        struct.pack_into("<i", data, 8, 1)
        self.other.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "header"):
            compare(self.ref, self.other)
        self.write(self.other, [(0, 1, 2)] * 2)
        data = bytearray(self.other.read_bytes())
        struct.pack_into("<i", data, 36, 0)
        self.other.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "target"):
            compare(self.ref, self.other)
        for threshold in [-1, math.inf, math.nan]:
            with self.assertRaisesRegex(ValueError, "threshold"):
                compare(self.ref, self.other, threshold)


if __name__ == "__main__":
    unittest.main()
