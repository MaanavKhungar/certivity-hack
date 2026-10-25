import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Exercise pure subgoal helpers without requiring PDF/model integrations.
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

    def test_added_qualifier_is_still_changed(self):
        old = "The device shall remain securely attached"
        new = "The device shall always remain securely attached"
        self.assertFalse(sg.pure_reordering_equivalent(old, new))

    def test_changed_value_is_still_changed(self):
        old = "The device shall operate at 12 V"
        new = "The device shall operate at 24 V"
        self.assertFalse(sg.pure_reordering_equivalent(old, new))

    def test_subject_object_reversal_is_not_assumed_equivalent(self):
        old = "The vehicle shall tow the trailer"
        new = "The trailer shall tow the vehicle"
        self.assertFalse(sg.pure_reordering_equivalent(old, new))

    def test_order_sensitive_terms_stay_on_review_path(self):
        old = "The warning shall occur before braking"
        new = "Before the warning braking shall occur"
        self.assertFalse(sg.pure_reordering_equivalent(old, new))

    def test_duplicate_word_count_is_not_ignored(self):
        old = "The lamp shall flash before the warning"
        new = "The lamp shall flash flash before the warning"
        self.assertFalse(sg.pure_reordering_equivalent(old, new))

    def test_build_suppresses_only_confirmed_reorder_decision(self):
        old = "The device shall, during operation, remain securely attached"
        new = "During operation, the device shall remain securely attached"
        node = SimpleNamespace(label="1.1", doc="main", key="1.1", page_start=2)
        alignment = {
            "reg": "UN RTEST", "v_new": "2.0", "old": {"1.1": node},
            "new": {"1.1": node}, "pages_new": [],
        }
        decision = {
            "requirement_id": "R-001", "decision": "modified",
            "new_paragraph": "1.1", "new_text": new,
            "old_quote": old, "new_quote": new, "note": "reordered",
        }
        self.assertEqual(sg.build_changes(alignment, [{
            "key": "1.1", "partners": ["1.1"], "decisions": [decision],
        }], []), [])

        changed = {**decision, "new_quote": new.replace("securely", "always securely")}
        self.assertEqual(len(sg.build_changes(alignment, [{
            "key": "1.1", "partners": ["1.1"], "decisions": [changed],
        }], [])), 1)

    def test_added_text_must_be_normative_in_the_source(self):
        node = SimpleNamespace(label="2.1", doc="main", key="2.1", page_start=3)
        alignment = {
            "reg": "UN RTEST", "v_new": "2.0", "old": {},
            "new": {"2.1": node}, "pages_new": [],
        }
        add = {"key": "2.1", "added": [{
            "new_text": "The service shall determine orientation.",
            "source_quote": "The orientation is to be determined by the service.",
        }]}
        self.assertEqual(sg.build_changes(alignment, [], [add]), [])

    def test_procedure_step_can_inherit_normative_lead_in(self):
        node = SimpleNamespace(label="2.2", doc="main", key="2.2", page_start=3)
        alignment = {
            "reg": "UN RTEST", "v_new": "2.0", "old": {},
            "new": {"2.2": node}, "pages_new": [],
        }
        add = {"key": "2.2", "added": [{
            "new_text": "The procedure shall position the vehicle.",
            "source_quote": "(a) Position the vehicle.",
        }]}
        self.assertEqual(len(sg.build_changes(alignment, [], [add])), 1)

    def test_added_text_already_covered_by_modification_is_deduplicated(self):
        node = SimpleNamespace(label="3.1", doc="main", key="3.1", page_start=4)
        alignment = {
            "reg": "UN RTEST", "v_new": "2.0", "old": {"3.1": node},
            "new": {"3.1": node}, "pages_new": [],
        }
        decision = {
            "requirement_id": "R-002", "decision": "modified",
            "new_paragraph": "3.1", "new_text": "The system shall stop: (a) the lamp shall flash; (b) the alarm shall sound.",
            "old_quote": "The system shall slow.",
            "new_quote": "The system shall stop: (a) the lamp shall flash; (b) the alarm shall sound.", "note": "changed",
        }
        add = {"key": "3.1", "added": [{
            "new_text": "The alerts shall operate.",
            "source_quote": "(a) The lamp shall flash; (b) the alarm shall sound.",
        }]}
        changes = sg.build_changes(alignment, [{
            "key": "3.1", "partners": ["3.1"], "decisions": [decision],
        }], [add])
        self.assertEqual([c["change"] for c in changes], ["modified"])


if __name__ == "__main__":
    unittest.main()
