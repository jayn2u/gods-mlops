from __future__ import annotations

import importlib
import unittest
from hashlib import sha256

from gods_mlops.datasets.manifest import canonical_json
from gods_mlops.training.contracts import locked_model


def _require(module_name: str, symbol: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise AssertionError(f"missing evaluation source contract module {module_name}") from error
    value = getattr(module, symbol, None)
    if value is None:
        raise AssertionError(f"{module_name}.{symbol} is part of the evaluation source contract")
    return value


class EvaluationManifestIdentityTests(unittest.TestCase):
    def test_evaluation_binds_dataset_manifest_hash_without_borrowing_eval_profile_config(self) -> None:
        validate = _require("gods_mlops.training.contracts", "validate_manifest_identity")
        model = locked_model("clip")
        manifest = {
            "schema_version": 1,
            "dataset_version": "dataset-2026-10-01-a1b2c3d4",
            "input_sha256": "f" * 64,
            "target": "both",
            "config": {"version": "dataset-capture-v3", "sha256": "d" * 64},
            "training": {"ready": True, "reason_codes": []},
            "evaluation": {"eligible": True, "reason_codes": []},
        }
        config = {
            "phase": "evaluation",
            "model_kind": "clip",
            "input_kind": "dataset_version",
            "input_id": manifest["dataset_version"],
            "input_sha256": sha256(canonical_json(manifest)).hexdigest(),
            "dataset_version": manifest["dataset_version"],
            "config_version": "clip-retrieval-eval-v1",
            "model_id": model.model_id,
            "model_revision": model.revision,
        }

        identity = validate(config, manifest)

        self.assertEqual(identity["phase"], "evaluation")
        self.assertEqual(identity["dataset_version"], manifest["dataset_version"])
        self.assertEqual(identity["config_version"], "clip-retrieval-eval-v1")
        with self.assertRaisesRegex(ValueError, "hash|manifest"):
            validate({**config, "input_sha256": "0" * 64}, manifest)

    def test_detr_evaluation_uses_exact_requested_split_and_not_probe_frame_cap(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("locked training image provides torch for detector runner coverage")
        select_items = _require("gods_mlops.training.detector", "_detector_items")
        items = [
            {"kind": "frame", "item_id": "train-frame", "split": "train"},
            {"kind": "frame", "item_id": "test-frame-a", "split": "test"},
            {"kind": "frame", "item_id": "test-frame-b", "split": "test"},
            {"kind": "crop", "item_id": "test-crop", "split": "test"},
        ]

        selected = select_items(
            {"items": items},
            {
                "phase": "evaluation",
                "target_phase": "evaluation",
                "evaluation_split": "test",
                "max_evaluation_frames": 1,
            },
        )

        self.assertEqual([item["item_id"] for item in selected], ["test-frame-a", "test-frame-b"])


if __name__ == "__main__":
    unittest.main()
