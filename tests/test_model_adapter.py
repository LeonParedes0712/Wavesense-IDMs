import unittest
from src.model_adapter import to_wavesense_probabilities


class TestModelAdapter(unittest.TestCase):

    def test_probability_mapping_and_sum(self):
        input_probs = {
            "REST": 0.05,
            "DIBUJANDO": 0.05,
            "MATH_LOAD": 0.70,
            "BRAWL_STARS": 0.10,
            "SUBWAY_SURFERS": 0.10,
        }
        res = to_wavesense_probabilities(input_probs)

        # 1. Claves exactas requeridas
        self.assertEqual(
            set(res.keys()), {"REST", "LOW_LOAD", "HIGH_LOAD", "ARTIFACT"}
        )

        # 2. Mapeos esperados
        self.assertAlmostEqual(res["REST"], 0.05, places=4)
        self.assertAlmostEqual(res["LOW_LOAD"], 0.05, places=4)
        self.assertAlmostEqual(res["HIGH_LOAD"], 0.90, places=4)
        self.assertAlmostEqual(res["ARTIFACT"], 0.0, places=4)

        # 3. Suma total debe ser 1.0
        self.assertAlmostEqual(sum(res.values()), 1.0, places=4)

    def test_artifact_override(self):
        input_probs = {"MATH_LOAD": 1.0}
        res = to_wavesense_probabilities(input_probs, signal_has_artifact=True)
        self.assertEqual(res["ARTIFACT"], 1.0)
        self.assertEqual(res["HIGH_LOAD"], 0.0)

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            to_wavesense_probabilities({})


if __name__ == "__main__":
    unittest.main()