import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "reconstruct_fair1m_tiled_annotations.py"
SPEC = importlib.util.spec_from_file_location("fair_reconstruct", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ReconstructionTests(unittest.TestCase):
    def test_valid_record_and_fully_contained_object(self):
        obj = MODULE.parse_raw_line("10 10 20 10 20 20 10 20 small_car 0", "1", 1, 100, 100)
        line = MODULE.reconstruct_tile([obj], 0, 0, 50)[0]
        self.assertTrue(line.endswith("small-car 0"))

    def test_malformed_and_unknown_records(self):
        cases = (
            ("1 2 3", "expected at least"),
            ("nan 0 1 0 1 1 0 1 van 0", "non-finite"),
            ("0 0 1 0 1 1 0 1 mystery 0", "unknown class"),
            ("0 0 1 0 2 0 3 0 van 0", "zero-area"),
        )
        for line, message in cases:
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, message):
                MODULE.parse_raw_line(line, "1", 1, 100, 100)

    def test_out_of_bounds_record(self):
        with self.assertRaisesRegex(ValueError, "outside source image"):
            MODULE.parse_raw_line("90 90 110 90 110 99 90 99 van 0", "1", 1, 100, 100)

    def test_truncated_object_is_retained_at_iof_threshold_and_ignored(self):
        obj = MODULE.parse_raw_line("0 10 10 10 10 20 0 20 van 0", "1", 1, 100, 100)
        lines = MODULE.reconstruct_tile([obj], 2, 0, 98, iof_threshold=0.7)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].endswith("van 2"))

    def test_object_below_iof_threshold_is_omitted(self):
        obj = MODULE.RawObject(((-4, 10), (6, 10), (6, 20), (-4, 20)), "van", 1)
        self.assertEqual(MODULE.reconstruct_tile([obj], 0, 0, 100, iof_threshold=0.7), [])

    def test_precision_preserves_16088_other_airplane_area(self):
        obj = MODULE.parse_raw_line(
            "16088.0000001 16088.0000001 16088.0000002 16088.0000001 "
            "16088.0000002 16088.0000002 16088.0000001 16088.0000002 "
            "other-airplane 0", "16088", 1, 20000, 20000)
        line = MODULE.reconstruct_tile([obj], 16088, 16088, 800)[0]
        coords = [float(value) for value in line.split()[:8]]
        points = list(zip(coords[0::2], coords[1::2]))
        self.assertGreater(MODULE.polygon_area(points), 0.0)
        self.assertNotIn("0 0 0 0", line)


if __name__ == "__main__":
    unittest.main()
