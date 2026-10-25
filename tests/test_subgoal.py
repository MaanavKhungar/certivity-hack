import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# These unit tests exercise the pure alignment helpers; PDF/model adapters are
# intentionally stubbed so the regression suite needs no external services.
regparse = types.ModuleType("regparse")
regparse.parse_pdf = lambda *_args, **_kwargs: None
regparse.extract_pages = lambda *_args, **_kwargs: None
regparse.paragraph_key = lambda value: value
sys.modules.setdefault("regparse", regparse)
pipeline = types.ModuleType("pipeline")
pipeline.call_json = lambda *_args, **_kwargs: None
pipeline.RUNS = Path("runs")
sys.modules.setdefault("pipeline", pipeline)

import subgoal as sg


class WordReorderingTests(unittest.TestCase):
    def test_prompt_projects_only_the_changed_semantic_atom(self):
        self.assertIn("independent semantic obligations", sg.MOD_SYSTEM)
        self.assertIn("Omit an independent sibling constraint", sg.MOD_SYSTEM)

    def test_added_prompt_preserves_semantic_anchors(self):
        for anchor in ("actor", "condition", "negation", "number", "unit", "direction", "cross-reference"):
            self.assertIn(anchor, sg.ADD_SYSTEM)
        self.assertIn('do not change "is/are" into "shall"', sg.ADD_SYSTEM)
        self.assertIn("must remain passive", sg.ADD_SYSTEM)

    def test_adjunct_reordering_is_unchanged(self):
        old = "The device shall, during operation, remain securely attached"
        new = "During operation, the device shall remain securely attached"
        self.assertTrue(sg.pure_reordering_equivalent(old, new))

    def test_added_qualifier_is_still_a_change(self):
        old = "The device shall remain securely attached"
        new = "The device shall always remain securely attached"

        self.assertFalse(sg.pure_reordering_equivalent(old, new))

    def test_changed_value_is_still_a_change(self):
        old = "The device shall operate at 12 V"
        new = "The device shall operate at 24 V"

        self.assertFalse(sg.pure_reordering_equivalent(old, new))

    def test_subject_object_reversal_is_not_equivalent(self):
        self.assertFalse(sg.pure_reordering_equivalent(
            "The vehicle shall tow the trailer", "The trailer shall tow the vehicle"))

    def test_order_sensitive_terms_stay_on_review_path(self):
        self.assertFalse(sg.pure_reordering_equivalent(
            "The warning shall occur before braking", "Before the warning braking shall occur"))

    def test_duplicate_word_count_is_not_ignored(self):
        old = "The lamp shall flash before the warning"
        new = "The lamp shall flash flash before the warning"

        self.assertFalse(sg.pure_reordering_equivalent(old, new))

    @staticmethod
    def alignment(key="1.1"):
        node = SimpleNamespace(label=key, doc="main", key=key, page_start=2)
        return {"reg": "UN RTEST", "v_new": "2.0", "old": {key: node},
                "new": {key: node}, "pages_new": []}

    def test_build_suppresses_only_confirmed_reorder_decision(self):
        old = "The device shall, during operation, remain securely attached"
        new = "During operation, the device shall remain securely attached"
        decision = {"requirement_id": "R-001", "decision": "modified", "new_paragraph": "1.1",
                    "new_text": new, "old_quote": old, "new_quote": new, "note": "reordered"}
        group = {"key": "1.1", "partners": ["1.1"], "decisions": [decision]}
        self.assertEqual(sg.build_changes(self.alignment(), [group], []), [])
        group["decisions"] = [{**decision, "new_quote": new.replace("securely", "always securely")}]
        self.assertEqual(len(sg.build_changes(self.alignment(), [group], [])), 1)

    def test_added_text_must_not_invent_normative_force(self):
        add = {"key": "1.1", "added": [{"new_text": "The service shall determine orientation.",
                "source_quote": "The orientation is to be determined by the service."}]}
        self.assertEqual(sg.build_changes(self.alignment(), [], [add]), [])

    def test_procedure_step_can_inherit_normative_lead_in(self):
        add = {"key": "1.1", "added": [{"new_text": "The procedure shall position the vehicle.",
                "source_quote": "(a) Position the vehicle."}]}
        self.assertEqual(len(sg.build_changes(self.alignment(), [], [add])), 1)

    def test_added_duplicate_of_modification_is_removed(self):
        decision = {"requirement_id": "R-002", "decision": "modified", "new_paragraph": "1.1",
                    "new_text": "The system shall stop: (a) the lamp shall flash; (b) the alarm shall sound.",
                    "old_quote": "The system shall slow.",
                    "new_quote": "The system shall stop: (a) the lamp shall flash; (b) the alarm shall sound.",
                    "note": "changed"}
        group = {"key": "1.1", "partners": ["1.1"], "decisions": [decision]}
        add = {"key": "1.1", "added": [{"new_text": "The alerts shall operate.",
               "source_quote": "(a) The lamp shall flash; (b) The alarm shall sound."}]}
        self.assertEqual([c["change"] for c in sg.build_changes(self.alignment(), [group], [add])], ["modified"])


if __name__ == "__main__":
    unittest.main()
