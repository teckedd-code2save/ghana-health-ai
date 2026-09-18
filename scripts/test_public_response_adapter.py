import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("public_adapter", Path(__file__).resolve().parents[1] / "modal/train/compare_public_response_adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class PublicAdapterTests(unittest.TestCase):
    def test_mapping_preserves_target_names(self):
        old = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
        new = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        self.assertEqual(adapter.mapped_name(old), new)
        self.assertEqual(adapter.validate_shapes({old: [64, 5376]}, {new: [64, 5376]}), {new: (64, 5376)})
        with self.assertRaises(ValueError):
            adapter.validate_shapes({old: [16, 5376]}, {new: [64, 5376]})
        with self.assertRaises(ValueError):
            adapter.validate_shapes({old: [64, 5376], new: [64, 5376]}, {new: [64, 5376]})

    def test_no_vision_or_unrelated_weights(self):
        for key in ("vision_tower.weight", "base_model.model.model.layers.0.self_attn.q_proj.weight", "../adapter"):
            with self.assertRaises(ValueError):
                adapter.mapped_name(key)

    def test_only_explicit_image_encoder_is_excluded(self):
        vision = "base_model.model.model.vision_tower.encoder.layers.0.mlp.down_proj.linear.lora_A.weight"
        text = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
        self.assertEqual(adapter.text_weights({vision: [64, 256], text: [64, 5376], "unknown": [1]}),
                         {text: [64, 5376], "unknown": [1]})


if __name__ == "__main__":
    unittest.main()
