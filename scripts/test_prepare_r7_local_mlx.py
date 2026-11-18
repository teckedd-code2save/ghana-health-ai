import unittest
from prepare_r7_local_mlx import map_key, validate_config, MODEL, REVISION


class AdapterTests(unittest.TestCase):
    def test_mapping_preserves_layer_and_projection(self):
        for letter in "AB":
            self.assertEqual(map_key(f"base_model.model.model.layers.17.linear_attn.in_proj_qkv.lora_{letter}.weight"),
                f"language_model.model.layers.17.linear_attn.in_proj_qkv.lora_{letter.lower()}")

    def test_unknown_keys_fail_instead_of_dropping_weights(self):
        for key in ("model.layers.0.lora_A.weight", "base_model.model.model.layers.32.mlp.up_proj.lora_A.weight",
                    "base_model.model.lm_head.lora_A.weight", "base_model.model.model.layers.0.mlp.up_proj.weight"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                map_key(key)

    def test_nonstandard_scaling_and_wrong_base_are_rejected(self):
        config = {"base_model_name_or_path": MODEL, "revision": REVISION, "peft_type": "LORA",
                  "r": 16, "lora_alpha": 32, "bias": "none"}
        validate_config(config)
        for change in ({"use_rslora": True}, {"rank_pattern": {"q_proj": 8}}, {"lora_alpha": 16},
                       {"revision": "main"}, {"bias": "all"}, {"modules_to_save": ["lm_head"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config({**config, **change})


if __name__ == "__main__":
    unittest.main()
