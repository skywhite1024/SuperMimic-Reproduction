"""Check cross-stage provenance without licensed assets or inference imports."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from retarget_g1_headless import BODY_MODEL_KEY, GVHMR_PIN, verify_body_model


class ModelBindingTest(unittest.TestCase):
    def test_copied_model_is_accepted_but_changed_model_is_rejected(self):
        payload = b"test-only provenance fixture, not a body model"
        digest = hashlib.sha256(payload).hexdigest()
        receipt = {"schema_version": "gvhmr-fixed-identity-run/1.0",
                   "source_commit": GVHMR_PIN, "model_sha256": {BODY_MODEL_KEY: digest}}
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/"SMPLX_NEUTRAL.npz"
            model.write_bytes(payload)
            self.assertEqual(verify_body_model(model, receipt), digest)
            model.write_bytes(payload + b" different release")
            with self.assertRaisesRegex(ValueError, "differs from the model used"):
                verify_body_model(model, receipt)

    def test_missing_provenance_fails_before_reading_model(self):
        receipts = [
            {},
            {"schema_version": "gvhmr-fixed-identity-run/1.0", "source_commit": "other"},
            {"schema_version": "gvhmr-fixed-identity-run/1.0", "source_commit": GVHMR_PIN},
        ]
        for receipt in receipts:
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                verify_body_model(Path("nonexistent-model.npz"), receipt)


if __name__ == "__main__":
    unittest.main()
