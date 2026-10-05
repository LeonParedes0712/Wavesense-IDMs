import unittest

from src.triggering import TemporalTrigger, TriggerConfig


VALID_HIGH_LOAD = {
    "REST": 0.04,
    "LOW_LOAD": 0.08,
    "HIGH_LOAD": 0.86,
    "ARTIFACT": 0.02,
}

LOW_CONFIDENCE = {
    "REST": 0.10,
    "LOW_LOAD": 0.20,
    "HIGH_LOAD": 0.70,
    "ARTIFACT": 0.00,
}

ARTIFACT_WINDOW = {
    "REST": 0.05,
    "LOW_LOAD": 0.10,
    "HIGH_LOAD": 0.20,
    "ARTIFACT": 0.65,
}


class FakeClock:
    def __init__(self) -> None:
        self.current_time = 0.0

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


class TemporalTriggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.trigger = TemporalTrigger(clock=self.clock)

    def test_one_high_load_window_does_not_trigger(self) -> None:
        decision = self.trigger.evaluate(VALID_HIGH_LOAD)

        self.assertFalse(decision.triggered)
        self.assertEqual(decision.reason, "waiting_for_persistence")
        self.assertEqual(decision.consecutive_high_load_windows, 1)

    def test_three_consecutive_high_load_windows_trigger(self) -> None:
        self.trigger.evaluate(VALID_HIGH_LOAD)
        self.trigger.evaluate(VALID_HIGH_LOAD)
        decision = self.trigger.evaluate(VALID_HIGH_LOAD)

        self.assertTrue(decision.triggered)
        self.assertEqual(decision.reason, "sustained_high_load")
        self.assertEqual(decision.consecutive_high_load_windows, 0)

    def test_window_below_threshold_resets_streak(self) -> None:
        self.trigger.evaluate(VALID_HIGH_LOAD)
        decision = self.trigger.evaluate(LOW_CONFIDENCE)

        self.assertFalse(decision.triggered)
        self.assertEqual(decision.reason, "high_load_below_threshold")
        self.assertEqual(decision.consecutive_high_load_windows, 0)

        next_decision = self.trigger.evaluate(VALID_HIGH_LOAD)
        self.assertEqual(next_decision.consecutive_high_load_windows, 1)

    def test_artifact_blocks_and_resets_streak(self) -> None:
        self.trigger.evaluate(VALID_HIGH_LOAD)
        decision = self.trigger.evaluate(ARTIFACT_WINDOW)

        self.assertFalse(decision.triggered)
        self.assertEqual(decision.reason, "artifact_detected")
        self.assertEqual(decision.consecutive_high_load_windows, 0)

    def test_cooldown_prevents_repeated_trigger(self) -> None:
        for _ in range(3):
            self.trigger.evaluate(VALID_HIGH_LOAD)

        decision = self.trigger.evaluate(VALID_HIGH_LOAD)

        self.assertFalse(decision.triggered)
        self.assertEqual(decision.reason, "cooldown_active")
        self.assertEqual(decision.consecutive_high_load_windows, 0)
        self.assertEqual(decision.cooldown_remaining_seconds, 30.0)

    def test_can_trigger_again_after_cooldown_and_new_streak(self) -> None:
        for _ in range(3):
            self.trigger.evaluate(VALID_HIGH_LOAD)

        self.clock.advance(30.0)

        self.trigger.evaluate(VALID_HIGH_LOAD)
        self.trigger.evaluate(VALID_HIGH_LOAD)
        decision = self.trigger.evaluate(VALID_HIGH_LOAD)

        self.assertTrue(decision.triggered)
        self.assertEqual(decision.reason, "sustained_high_load")


if __name__ == "__main__":
    unittest.main()