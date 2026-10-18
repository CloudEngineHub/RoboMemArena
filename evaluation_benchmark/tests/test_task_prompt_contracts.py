"""Check generic task wording without importing simulator or policy runtimes."""
import ast
from pathlib import Path
import unittest


class TaskPromptContractTest(unittest.TestCase):
    def test_requested_object_and_order_wording(self):
        source = Path(__file__).resolve().parents[1] / "scripts/eval_common.py"
        assignments = [node for node in ast.parse(source.read_text()).body
                       if isinstance(node, ast.Assign)]
        prompts = next(ast.literal_eval(node.value) for node in assignments
                       if any(isinstance(t, ast.Name) and t.id == "TASK_PROMPTS"
                              for t in node.targets))
        self.assertEqual(len(prompts), 26)
        for task in (4, 5):
            self.assertIn("in this order: top, middle, bottom", prompts[f"task{task}"])
            self.assertIn("Open and close each drawer", prompts[f"task{task}"])
        self.assertEqual(prompts["task11"],
                         "Put cookies into the top drawer and put chocolate into another drawer.")
        self.assertEqual(prompts["task19"],
                         "Pick and place tomato sauce, milk, and orange juice from cabinet1 to cabinet2, in that order.")
        self.assertEqual(prompts["task25"],
                         "First pick and place butter from plate1 to plate2, then pick and place cream from plate1 to plate2.")


if __name__ == "__main__":
    unittest.main()
