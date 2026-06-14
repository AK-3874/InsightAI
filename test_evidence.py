import unittest

from src.threat_ai.evidence import (
    ChallengeConfig,
    FullPlatformDetector,
    build_ground_truth_challenge,
    evaluate_predictions,
)


class EvidenceHarnessTests(unittest.TestCase):
    def test_challenge_generation_is_reproducible(self):
        config = ChallengeConfig(seed=77, scenario_count=25, population=500)
        first = build_ground_truth_challenge(config)
        second = build_ground_truth_challenge(config)

        first_docs = [(doc.id, doc.text, doc.timestamp) for doc in first.documents]
        second_docs = [(doc.id, doc.text, doc.timestamp) for doc in second.documents]
        self.assertEqual(first_docs, second_docs)
        self.assertEqual(first.ground_truth, second.ground_truth)

    def test_documents_do_not_expose_scenario_type_label(self):
        challenge = build_ground_truth_challenge(ChallengeConfig(seed=3, scenario_count=10, population=200))

        for doc in challenge.documents:
            self.assertNotIn("scenario_type", doc.metadata)
            self.assertIn("scenario_id", doc.metadata)

    def test_metrics_count_recall_and_false_positives(self):
        challenge = build_ground_truth_challenge(ChallengeConfig(seed=11, scenario_count=20, population=300))
        predictions = FullPlatformDetector().predict(challenge.scenarios)
        metrics = evaluate_predictions(challenge, predictions, "Full Platform")

        positives = sum(1 for scenario in challenge.scenarios if scenario.positive)
        self.assertEqual(metrics.true_positives + metrics.false_negatives, positives)
        self.assertGreaterEqual(metrics.recall, 0.0)
        self.assertLessEqual(metrics.recall, 1.0)
        self.assertGreaterEqual(metrics.precision, 0.0)
        self.assertLessEqual(metrics.precision, 1.0)


if __name__ == "__main__":
    unittest.main()
