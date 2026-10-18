"""Validate supplied scene layouts with LIBERO-compatible basket grouping."""
import hashlib
from pathlib import Path
import unittest

from bddl.parsing import scan_tokens


ROOT = Path(__file__).resolve().parents[2]
TASKS = (
    ("1_cookies_tomato_basket.bddl", ("cookies_1", "tomato_sauce_1"),
     "b2c7ab47b89fa733c5600d366777b9945211565f7f752cd25a13506b12ed24c1"),
    ("2_butter_popcorn_basket.bddl", ("butter_1", "popcorn_1"),
     "e7e8769798c914ed89efe3150b6bcf28f36eed061b3781d4b26a26c565048a79"),
    ("3_cream_pudding_basket.bddl", ("cream_cheese_1", "chocolate_pudding_1"),
     "3fc319fa2516610aad79750edd01e1c31ab3c32a23986de471e2af232d6da4f8"),
)


class Task123DoubleBasketTest(unittest.TestCase):
    def test_both_paths_match_published_hashes_and_parse_as_double_basket(self):
        for filename, foods, expected_sha in TASKS:
            for directory in ("bddl", "evaluation_benchmark/bddl"):
                with self.subTest(file=filename, directory=directory):
                    path = ROOT / directory / filename
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_sha)
                    tokens = scan_tokens(filename=str(path))
                    self.assertEqual(tokens[0], "define")
                    sections = {node[0]: node[1:] for node in tokens[1:]}
                    objects = sections[":objects"]
                    # LIBERO overwrites earlier declarations of the same type.
                    self.assertEqual(objects.count("basket"), 1)
                    index = objects.index("basket")
                    self.assertEqual(objects[index - 3:index], ["basket_1", "basket_2", "-"])
                    self.assertIn(["on", "basket_1", "coffee_table_basket_init_region"], sections[":init"])
                    self.assertIn(["on", "basket_2", "coffee_table_basket_2_init_region"], sections[":init"])
                    regions = {node[0]: node[1:] for node in sections[":regions"]}
                    second = dict((node[0], node[1:]) for node in regions["basket_2_init_region"])
                    self.assertEqual(second[":target"], ["coffee_table"])
                    self.assertEqual(len(second[":ranges"][0][0]), 4)
                    self.assertEqual(sections[":goal"], [
                        ["and", *[["in", food, "basket_1_contain_region"] for food in foods]],
                    ])


if __name__ == "__main__":
    unittest.main()
